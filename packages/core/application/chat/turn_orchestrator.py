"""TurnOrchestrator: the pure control plane between HTTP transport and executors.

Owns the per-turn state machine: plan resolution, executor dispatch, the approval
pump, the stream commit guard, the disconnect policy (T4 invariant #3: a research
turn is never cancelled by a client disconnect) and lifecycle finalization. It
never infers policy itself — it consumes an ``ExecutionPlan``.

SSE contract: events flow to the transport VERBATIM (same frames as the legacy
router emitted), so ``apps/web`` / ``apps/desktop`` need no change.

Commit Point (design §5 step 6): the first user-visible content delta. Until then
a branch may still fall back to AGENT; after it the channel is LOCKED — an error
can only terminate the stream with a standardized event, never re-route mid-flight.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator

from core.application.chat import intent_funnel
from core.application.chat.context import ChatTurnContext
from core.application.chat.execution_plan import (
    ExecutionPlan,
    PlanKind,
    build_execution_plan,
)
from core.application.chat.executors.action import ActionExecutor
from core.application.chat.executors.agent import AgentExecutor
from core.application.chat.executors.base import (
    ChatDeps,
    ChatExecutor,
    EscalateToAgent,
    TurnRequest,
)
from core.application.chat.executors.terminal import TerminalExecutor
from core.application.chat.lifecycle import finalize_turn, handle_research_post_turn
from core.application.chat.understanding import resolve_source_facts
from core.config import settings
from core.logger import reset_log_context, set_log_context

logger = logging.getLogger(__name__)


class TurnOrchestrator:
    def __init__(self, deps: ChatDeps) -> None:
        self.deps = deps
        # The single path has only three terminal branches: a certified ACTION, a
        # decided-but-unsettled ERROR (honest message, never an Agent guess) and
        # the Agent. There is no per-lane gate to open.
        self._executors = {
            AgentExecutor.kind: AgentExecutor(),
            ActionExecutor.kind: ActionExecutor(),
            TerminalExecutor.kind: TerminalExecutor(),
        }

    # ── plan resolution ──────────────────────────────────────────────────────────
    async def resolve_plan(self, ctx: ChatTurnContext, deps: ChatDeps | None = None) -> ExecutionPlan:
        """The full Execution-Plan-Resolution funnel, in two ordered stages:

          (1) Source facts — the ORIGINAL request's source restriction
                     (``private_only`` / ``external_ok`` / web demand) is read
                     before routing. It is fact collection, never a route.
          (2) Intent Funnel — Matcher HIT certifies a table entry directly;
                     MISS/AMBIGUOUS goes through Recall + candidate aggregation +
                     cap_router (LayaChoice-v2), then argument acquisition. Every
                     non-COMPLETE outcome returns the SAME requirements untouched,
                     so ``build_execution_plan`` maps it to AGENT (fail-open) with
                     the user text byte-identical.

        ``build_execution_plan`` stays the SOLE policy mapper: ACTION eligibility
        and ``source_policy`` live there; execution authorization lives at the
        executor boundary.
        """
        requirements = resolve_source_facts(ctx.body.message)
        # Single path (ruling): the funnel cascade owns every turn's
        # routing attempt — gate + QIR cascade + argument binding live in
        # intent_funnel; the orchestrator keeps only the lifecycle and this
        # single call. The Intent Funnel returns the SAME requirements object when
        # it abstains, so the Agent keeps the turn byte-identical (fail-open).
        # An Intent Funnel admission reject is one of those abstains: it is an
        # Intent Funnel certification refusal, not a system-wide execution ban, so
        # the Agent keeps the turn and its own tool surface decides what may run.
        requirements = await intent_funnel.route(
            ctx, deps=deps, requirements=requirements,
        )
        plan = build_execution_plan(requirements)
        # Sink the turn's execution facts for the Agent:
        # * SOURCE POLICY — sandbox fencing comes from the ORIGINAL request, exactly
        #   as the control plane resolved it for this turn (Sandbox._turn_denied);
        # * Intent Funnel fallback context — why the funnel did not certify the
        #   turn (admission reject / acquisition failure / Binder or executor
        #   contract mismatch). It is routing context, never a permission: the
        #   Agent keeps its own tool registry, permissions, approvals and sandbox.
        fallback = getattr(ctx, "intent_funnel_failure", None)
        if plan.source_policy or fallback:
            agent_context = dict(ctx.agent_context or {})
            if plan.source_policy:
                agent_context["source_policy"] = plan.source_policy
            if fallback:
                agent_context["intent_funnel"] = dict(fallback)
            ctx.agent_context = agent_context
        return plan

    def executor_for(self, plan: ExecutionPlan) -> ChatExecutor:
        executor = self._executors.get(plan.kind)
        if executor is None:  # an unmapped kind must never hard-fail the turn
            logger.warning("no executor for plan kind=%s; falling back to AGENT", plan.kind)
            executor = self._executors[AgentExecutor.kind]
        return executor

    # ── non-streaming turn (/chat) ───────────────────────────────────────────────
    async def run_turn(self, ctx: ChatTurnContext) -> dict:
        plan = await self.resolve_plan(ctx, self.deps)
        if (
            plan.kind is PlanKind.AGENT
            and ctx.viewer_assembly
            and ctx.viewer_assembly.get("status") == "transcript_deferred"
        ):
            # Non-streaming transports have no channel to ask the PC client for the
            # complete transcript. Report the honest, recoverable requirement and execute
            # nothing (no LLM, no session write); the client retries on /chat/stream.
            return {
                "answer": None,
                "session_id": str(ctx.session_id),
                "user_id": str(ctx.user_id),
                "transcript_required": True,
                "transcript_request": ctx.viewer_assembly.get("transcript_request"),
            }
        if plan.capture_request is not None:
            # Non-streaming transports have no channel to ask the PC client for a
            # capture. Report the honest, recoverable requirement and execute nothing
            # (no tool, no LLM, no session write); the client retries on /chat/stream.
            return {
                "answer": None,
                "session_id": str(ctx.session_id),
                "user_id": str(ctx.user_id),
                "capture_required": True,
                "capture_request": plan.capture_request,
            }
        executor = self.executor_for(plan)
        req = TurnRequest(ctx=ctx, deps=self.deps, plan=plan)
        research_continuing = False
        logger.info(
            "chat.plan-resolved turn_kind=%s reason=%s", plan.kind.value, plan.reason
        )
        try:
            try:
                result = await executor.run(req)
            except EscalateToAgent as esc:
                # Pre-commit user-input fallback (schema/preflight): the Agent — with
                # its tools — owns the turn now, governed by the turn's SOURCE POLICY
                # from the original request. The user text is replayed byte-identical;
                # an ACTION that abstained leaves no trace in what the Agent receives.
                logger.info(
                    "chat.plan-escalated from=%s to=agent reason=%s",
                    plan.kind.value, esc.reason,
                )
                executor = self._executors[AgentExecutor.kind]
                result = await executor.run(req)
        finally:
            # /chat owns the run for the lifetime of this request. Hand the finished
            # interactive turn to the worker chain unless it hit a stop condition; the
            # single-task run slot is released only when the run is NOT handed off.
            research_continuing = await handle_research_post_turn(ctx, self.deps)
        payload = {
            "answer": result.final_answer,
            "messages": result.messages,
            "usage": result.usage,
        }
        try:
            resp = await finalize_turn(ctx, self.deps, payload, tool="chat")
        finally:
            if ctx.log_tokens is not None:
                reset_log_context(ctx.log_tokens)
        resp["messages"] = result.messages
        if research_continuing:
            resp["research_continuing"] = True
        return resp

    # ── streaming turn (/chat/stream) — mirrors the legacy gen() pump exactly ────
    async def stream_turn(
        self, ctx: ChatTurnContext, *, t_entry: float
    ) -> AsyncIterator[dict]:
        """Yield SSE frame dicts (``{"data": json}``) for one streaming turn."""
        deps = self.deps
        plan = await self.resolve_plan(ctx, deps)
        # ── Deferred client capture (funnel-certified vision ONLY) ────────────────
        # The funnel already decided ``vision``; the picture lives on the user's PC, so
        # the stream pauses here — BEFORE any executor, tool or model call — emits a
        # ``capture-request`` frame, and waits for the client to upload the page/frame
        # and POST the ticket. On success THIS plan resumes with the asset id; the
        # funnel is never re-run and no argument is extracted twice. A failed/empty
        # capture is surfaced as a recoverable ``capture-failed`` frame and retried; on
        # exhaustion the turn ends honestly with NO tool/LLM call and no session write.
        if plan.capture_request is not None and deps.capture_broker is not None:
            from core.application.chat.capture import (
                apply_capture_to_context,
                merge_payload,
                resume_plan,
            )

            capture_spec = plan.capture_request
            max_attempts = max(1, int(settings.chat_capture_max_attempts))
            resumed_plan = None
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                ticket = deps.capture_broker.open(
                    user_id=ctx.user_id, session_id=ctx.session_id
                )
                request = {
                    **capture_spec, "ticket_id": ticket.id,
                    "attempt": attempt, "max_attempts": max_attempts,
                }
                yield {"data": json.dumps(
                    {"type": "capture-request", "data": request}, ensure_ascii=False
                )}
                try:
                    submitted = await asyncio.wait_for(
                        ticket.future, timeout=settings.chat_capture_timeout_seconds
                    )
                except TimeoutError:
                    submitted = {"status": "failed", "reason": "capture_timeout"}
                finally:
                    deps.capture_broker.close(ticket.id)
                payload = merge_payload(capture_spec, submitted)
                if payload.get("status") == "ok" and payload.get("asset_id"):
                    apply_capture_to_context(ctx, payload, deps.viewer)
                    resumed_plan = resume_plan(plan, payload)
                    logger.info(
                        "chat.capture-resumed attempt=%d asset=%s",
                        attempt, payload.get("asset_id"),
                    )
                    break
                retryable = attempt < max_attempts
                yield {"data": json.dumps({
                    "type": "capture-failed",
                    "data": {
                        "reason": payload.get("reason") or "capture_failed",
                        "attempt": attempt, "max_attempts": max_attempts,
                        "retryable": retryable,
                        "question": capture_spec.get("question") or "",
                    },
                }, ensure_ascii=False)}
                if not retryable:
                    break
            if resumed_plan is None:
                yield {"data": json.dumps({"type": "done", "data": {
                    "answer": None,
                    "session_id": str(ctx.session_id),
                    "user_id": str(ctx.user_id),
                    # No session write happened, so simply resending the message is a
                    # safe recovery — never a fabricated answer from another source.
                    "capture": {"status": "failed", "retryable": True},
                }}, ensure_ascii=False)}
                return
            plan = resumed_plan
        # ── Deferred full transcript (video FULL, Agent lane) ─────────────────────
        # The question needs the whole video — the assembly already classified FULL — but
        # the deferred-data client sent no transcript. Ask NOW (never at send time) over
        # the SAME ticket/pause channel as capture, then resume the SAME turn with the
        # client's locally parsed transcript. No re-classification, no partial content,
        # and on failure no model call at all.
        if (
            plan.kind is PlanKind.AGENT
            and ctx.viewer_assembly
            and ctx.viewer_assembly.get("status") == "transcript_deferred"
        ):
            spec = ctx.viewer_assembly.get("transcript_request") or {}
            max_attempts = max(1, int(settings.chat_capture_max_attempts))
            completed = None
            attempt = 0
            while attempt < max_attempts and deps.capture_broker is not None:
                attempt += 1
                ticket = deps.capture_broker.open(
                    user_id=ctx.user_id, session_id=ctx.session_id
                )
                yield {"data": json.dumps({
                    "type": "transcript-request",
                    "data": {**spec, "ticket_id": ticket.id,
                             "attempt": attempt, "max_attempts": max_attempts},
                }, ensure_ascii=False)}
                try:
                    submitted = await asyncio.wait_for(
                        ticket.future, timeout=settings.chat_capture_timeout_seconds
                    )
                except TimeoutError:
                    submitted = {"status": "failed", "reason": "transcript_timeout"}
                finally:
                    deps.capture_broker.close(ticket.id)
                if (
                    submitted.get("status") == "ok"
                    and str(submitted.get("full_text") or "").strip()
                ):
                    completed = deps.viewer.with_transcript(
                        ctx.viewer_assembly, submitted
                    )
                    break
                retryable = attempt < max_attempts
                yield {"data": json.dumps({
                    "type": "transcript-failed",
                    "data": {
                        "reason": submitted.get("reason") or "transcript_unavailable",
                        "attempt": attempt, "max_attempts": max_attempts,
                        "retryable": retryable,
                    },
                }, ensure_ascii=False)}
                if not retryable:
                    break
            status = (completed or {}).get("status")
            if status != "injected":
                aborted = {
                    "mode": "full",
                    "status": "too_large" if status == "too_large" else "unavailable",
                    "rejected": (ctx.viewer_assembly or {}).get("rejected") or [],
                }
                yield {"data": json.dumps(
                    {"type": "viewer", "data": aborted}, ensure_ascii=False
                )}
                yield {"data": json.dumps({"type": "done", "data": {
                    "answer": None,
                    "session_id": str(ctx.session_id),
                    "user_id": str(ctx.user_id),
                    # Nothing was sent to a model and no session row was written, so
                    # resending the message is the safe recovery.
                    "transcript": {"status": "failed", "retryable": True},
                }}, ensure_ascii=False)}
                return
            ctx.viewer_assembly = completed
            ctx.agent_context = {**(ctx.agent_context or {}), "viewer": completed}
            logger.info(
                "chat.transcript-resumed chars=%d",
                len(str((completed.get("blocks") or [{}])[-1].text)),
            )
        executor = self.executor_for(plan)
        req = TurnRequest(ctx=ctx, deps=deps, plan=plan)
        # Tag every log line the stream emits with the user + session. Set inside the
        # generator (not the route) because the SSE generator runs after the handler
        # returns — the sibling pump task inherits it.
        log_tokens = set_log_context(user_id=str(ctx.user_id), session_id=str(ctx.session_id))
        # The agent may block on a human-in-the-loop approval (awaiting POST
        # /approvals/{id}), so a plain `async for` over run_stream would deadlock — the
        # stream can't advance while the approval-request frame sits unyielded. Pump the
        # stream into a queue in a sibling task, and let the ApprovalStore's sink push
        # approval frames into the same queue; this generator only ever reads the queue.
        frames: asyncio.Queue = asyncio.Queue()
        from agent.security.approvals import (  # lazy: keeps import light for tests
            ApprovalStore,
            set_request_approval,
        )

        store = ApprovalStore(
            deps.new_approval_bridge().broker,
            user_id=str(ctx.user_id),
            sink=lambda evt: frames.put_nowait(("approval", evt)),
        )
        set_request_approval(store)

        async def pump(runner: ChatExecutor):
            # NOTE: only ever await INSIDE the executor stream (never on frames). A
            # cancelled pump awaiting frames.put would swallow CancelledError in its
            # finally and strand the run_stream generator's cleanup; the queue is
            # unbounded so put_nowait never blocks, and a client disconnect lands the
            # CancelledError in the loop below (turn-cancelled logging + memory close).
            try:
                async for evt in runner.stream(
                    req, progress_sink=lambda evt: frames.put_nowait(("agent", evt))
                ):
                    frames.put_nowait(("agent", evt))
            except EscalateToAgent as esc:
                # Pre-commit only: an executor raises this before emitting any event,
                # so nothing was committed and the swap below is legal (design §5).
                logger.info(
                    "chat.stream-escalated from=%s to=agent reason=%s",
                    plan.kind.value, esc.reason,
                )
                frames.put_nowait(("escalate", esc.reason))
            finally:
                # Sentinel so the consumer below always terminates after the stream ends,
                # including on cancellation (the loop already logs turn-cancelled).
                frames.put_nowait(("agent", {"type": "done", "data": None}))

        pump_task = asyncio.create_task(pump(executor))
        final = None
        research_continuing = False
        # Stream Commit Point: set on the first user-visible content delta. Until then
        # a pre-execution block may still fall back; after it the channel is locked
        # (errors terminate the stream; transparent executor switching is forbidden).
        committed = False
        # Escalation bookkeeping: at most ONE pre-commit swap (fast path -> AGENT). The
        # abandoned pump still enqueues its done-None sentinel; the loop below must skip
        # it (stale_sentinels) or the turn would "finish" with an empty payload.
        stale_sentinels = 0
        switched = False

        def collect_done_payload() -> dict | None:
            """Disconnect path: pull the done payload the drained pump left in the queue
            (the run_stream event, not the None sentinel) when the loop never saw it."""
            try:
                while True:
                    kind, data = frames.get_nowait()
                    if (
                        kind == "agent"
                        and isinstance(data, dict)
                        and data.get("type") == "done"
                        and data.get("data") is not None
                    ):
                        return data["data"]
            except asyncio.QueueEmpty:
                return None

        # TTFT anchors: first SSE frame vs first visible content delta, both from t_entry.
        first_event_ms = first_content_ms = None
        try:
            while True:
                kind, data = await frames.get()
                if kind == "approval":
                    # Approval-request frame: forward verbatim (client POSTs /approvals/{id}).
                    yield {"data": json.dumps(data, ensure_ascii=False, default=str)}
                    continue
                if kind == "escalate":
                    # Pre-commit fast-path → AGENT swap (Phase 4). Re-pump the Agent into
                    # the SAME queue; the abandoned pump's sentinel is skipped below.
                    if committed or switched:
                        # Channel locked / already switched: never re-route mid-flight —
                        # surface a terminal error instead (design §5 Commit Point).
                        error_evt = {
                            "type": "error",
                            "data": {"message": f"escalation after commit: {data}"},
                        }
                        yield {"data": json.dumps(error_evt, ensure_ascii=False)}
                        frames.put_nowait(("agent", {"type": "done", "data": None}))
                        continue
                    switched = True
                    stale_sentinels += 1  # the escalated pump still enqueues its sentinel
                    # Pre-first-event only: the ACTION abstained before any effect, so
                    # the Agent replay is zero-pollution (original user text intact).
                    pump_task = asyncio.create_task(pump(self._executors[AgentExecutor.kind]))
                    continue
                if data["type"] == "done":
                    if stale_sentinels:
                        stale_sentinels -= 1
                        continue
                    final = data["data"]
                    break
                if first_event_ms is None:
                    first_event_ms = (time.perf_counter() - t_entry) * 1000
                if first_content_ms is None and data.get("type") == "content":
                    first_content_ms = (time.perf_counter() - t_entry) * 1000
                    committed = True  # past the Commit Point — the channel is now locked
                    logger.info(
                        "chat.stream-first-token first_event_ms=%.0f first_content_ms=%.0f",
                        first_event_ms, first_content_ms,
                    )
                yield {"data": json.dumps(data, ensure_ascii=False)}
        finally:
            set_request_approval(None)
            if ctx.research_turn:
                # Server-owned research run (T4 invariant #3): a client disconnect must
                # never cancel an active run. Let the pump drain to completion so the
                # agent's turn and its tool executions finish server-side, then still
                # finalize the turn and release the single-task run slot.
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task
                if final is None:
                    # The loop was abandoned (client disconnect) before the done frame;
                    # run the same post-turn bookkeeping even though no client is left.
                    with contextlib.suppress(Exception):
                        await finalize_turn(ctx, deps, collect_done_payload(), tool="chat_stream")
                research_continuing = await handle_research_post_turn(ctx, deps)
            else:
                # Non-research turn: the run is owned by the SSE pipe. Client disconnect
                # stops it so the turn's awaits re-raise CancelledError and unwind cleanly.
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump_task
            # Release the request-scoped log context: this generator may be closed at any
            # yield point, so the tags must not leak into the next unit of work.
            reset_log_context(log_tokens)

        # Normal completion path: the loop broke on the run's done frame.
        t_fin = time.perf_counter()
        done = await finalize_turn(ctx, deps, final, tool="chat_stream")
        logger.info(
            "chat.stream-end total_ms=%.0f finalize_ms=%.0f committed=%s",
            (time.perf_counter() - t_entry) * 1000, (time.perf_counter() - t_fin) * 1000,
            committed,
        )
        if research_continuing:
            done["research_continuing"] = True
        yield {"data": json.dumps({"type": "done", "data": done}, ensure_ascii=False)}

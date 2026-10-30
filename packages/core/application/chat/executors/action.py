"""ActionExecutor: the certified ACTION lane — direct dispatch of the tool bound to an admitted capability.

Serves turns the policy resolved to ``PlanKind.ACTION``: the Intent Funnel
certified that the turn demands EXACTLY one registered tool with fully-determined
args, so the ReAct loop is unnecessary flow — the seam calls the SAME
``ToolRuntime.execute`` the Agent would (approval waterfall, guards, sandbox, ACL
pipeline all inherited).

Stage order encodes the side-effect boundary isolation:

  0. FINAL EXECUTION AUTHORIZATION — before anything can run, resolve the
     authoritative live Registry view and re-confirm that this capability is
     allowed to execute from the Chat plane RIGHT NOW: registered, enabled and
     active, admitted by the Intent Funnel's capability rule (a hidden
     capability may be recalled/offered but never executed), bound to the tool
     being dispatched. The verdict is fail-closed: an unavailable/unreadable
     registry refuses. This is deliberately independent of the model's
     selection, any routing-time stamp, or caller-supplied action fields.
  0.1 preserved route/execute TOCTOU checks — Registry fingerprint drift, entry
     lifecycle, tool binding, the live ``ToolRuntime.schemas()`` roster, and the
     kind gate. A drift is C3 TERMINAL.
  0.5 routing-stamped ``binding_integrity`` (C2) terminates before the schema gate.
  1. existence + schema gate (live roster, then :func:`validate_action`) —
     pre-execution, side-effect-free; a malformed REQUEST (argument shape)
     escalates and the Agent owns the clarification.
  2. seam presence — not wired ⇒ C2 terminal (never re-plan around a wiring fault).
  3. the seam call is the SIDE-EFFECT BOUNDARY. Before it: anything the seam can
     PROVE did not execute raises :class:`ActionPreflightFailure` ⇒ escalate.
     After the tool body was entered, any other exception means the state is
     UNKNOWN — one honest terminal message, never a blind Agent replay (that is
     how duplicate folders get made).
  4. ``{"ok": False, "reason"}`` = a DECIDED denial (approval refused, policy
     guard) — no side effect happened; surfaced honestly as a terminal answer.

Everything before the first yielded event is pre-commit; the events after
(content → internal done, user+assistant rows persisted) mirror the former
DIRECT lane. The lane is deterministic EXCEPT for the query capabilities in
:data:`~core.application.chat.executors.answer_synthesis.SYNTHESIS_TOOLS`, whose
tool output is retrieved material rather than an answer: a successful dispatch of
one of those gets exactly ONE synthesis completion (:mod:`answer_synthesis`)
before it is returned. Every failure, denial and non-query tool keeps the
deterministic contract.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from core.application.chat.actions import (
    ActionIntegrityFailure,
    ActionPreflightFailure,
    ActionSchemaError,
    validate_action,
)
from core.application.chat.execution_plan import PlanKind
from core.application.chat.executors.answer_synthesis import (
    SYNTHESIS_TOOLS,
    synthesize_answer,
)
from core.application.chat.executors.base import (
    ChatEvent,
    EscalateToAgent,
    ExecutorResult,
    ProgressSink,
    TurnRequest,
)
from core.application.chat.executors.execution_authorization import (
    REASON_CAPABILITY_INACTIVE,
    REASON_CAPABILITY_NOT_ALLOWLISTED,
    REASON_TOOL_BINDING_MISMATCH,
    authorize_execution,
)
from core.application.chat.intent_funnel.failure import (
    EXECUTION_NOT_STARTED,
    KIND_CONTRACT_MISMATCH,
    KIND_EXECUTION_PREFLIGHT,
    REASON_EXECUTOR_PARAMETER_MISMATCH,
    REASON_EXECUTOR_PREFLIGHT,
    STAGE_EXECUTOR,
    build_failure,
    publish_failure,
)

logger = logging.getLogger(__name__)

# Honest terminal messages (no LLM on this branch — the words ARE the answer).
_STATE_UNKNOWN = (
    "That operation could not be confirmed — it may or may not have been applied, and "
    "I did not retry it automatically to avoid doing it twice. Please check the result "
    "and try again only if it is missing."
)
_DENIED_PREFIX = "Could not complete that request: "
# C2 (internal binding/runtime integrity) and C3 (registry drift / governance) are
# decided TERMINAL outcomes: an honest message, no seam, and NEVER an escalation —
# the Agent must not be used as a recovery channel for system faults.
_TERMINAL_INTEGRITY = (
    "That request references an action that is not available in the current system "
    "configuration, so nothing was done. Please contact the administrator if this "
    "persists."
)
_TERMINAL_STALE_ROUTE = (
    "The capability resolved for this request is no longer active, so nothing was "
    "done. Please phrase the request again."
)
_TERMINAL_NOT_PERMITTED = (
    "That action is not permitted on this channel, so nothing was done. Please "
    "phrase the request again."
)

# Structured refusal reasons emitted alongside the terminal message.
REASON_REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"
REASON_REGISTRY_VERSION_MISMATCH = "REGISTRY_VERSION_MISMATCH"
REASON_KIND_DISABLED = "KIND_DISABLED"
REASON_TOOL_NOT_ON_ROSTER = "TOOL_NOT_ON_ROSTER"
REASON_TOOL_ROSTER_UNAVAILABLE = "TOOL_ROSTER_UNAVAILABLE"
REASON_BINDING_INTEGRITY = "BINDING_INTEGRITY"


@dataclass(frozen=True)
class DispatchOutcome:
    """The terminal text plus an optional structured refusal reason.

    ``refusal_reason`` is set only when the dispatch was refused before the seam;
    a normal tool result (success, denial, state-unknown) has ``None`` here.
    """

    message: str
    refusal_reason: str | None = None


def _refusal(reason: str, *, stamped: bool = False) -> DispatchOutcome:
    """Map a machine-readable refusal code to the honest terminal text.

    ``stamped`` preserves the historical route/execute split: a funnel-certified
    turn (registry fingerprint present) that names a tool missing from the live
    roster is a STALE ROUTE ("phrase the request again"); a hand-built action with
    no routing stamp is a system INTEGRITY fault. The structured reason is the same.
    """
    if reason == REASON_TOOL_NOT_ON_ROSTER and not stamped:
        return DispatchOutcome(_TERMINAL_INTEGRITY, reason)
    if reason in (REASON_CAPABILITY_INACTIVE, REASON_REGISTRY_VERSION_MISMATCH,
                  REASON_KIND_DISABLED, REASON_TOOL_NOT_ON_ROSTER):
        return DispatchOutcome(_TERMINAL_STALE_ROUTE, reason)
    if reason == REASON_CAPABILITY_NOT_ALLOWLISTED:
        return DispatchOutcome(_TERMINAL_NOT_PERMITTED, reason)
    # Unknown capability / binding drift / registry or roster unavailable are system
    # integrity faults: the request names something this process cannot honour.
    return DispatchOutcome(_TERMINAL_INTEGRITY, reason)


def _turn_facts(ctx):
    """The turn's structured facts for the Agent-facing record, or ``None``.

    The executor may see a duck-typed test double without a request body; the
    record stays honest in that case (an empty context summary) rather than
    fabricating one.
    """
    if getattr(ctx, "body", None) is None:
        return None
    from core.application.chat.intent_funnel.contract import TurnFacts

    return TurnFacts.of(ctx)


def _schema_mismatch(args, schema) -> dict:
    """Evidence for a rejected ``validate_action`` call (reporting only).

    Mirrors the gate's own checks so the Agent is told exactly what failed:
    missing required slots, non-string values, over-length values, and args the
    tool schema does not declare. It never rewrites or drops a value.
    """
    if not isinstance(args, dict):
        return {"missing": [], "invalid": {"args": "not_a_mapping"}, "extra": []}
    missing: list[str] = []
    invalid: dict[str, str] = {}
    for slot, raw_spec in (schema or {}).items():
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        required = bool(spec.get("required", True))
        max_len = int(spec.get("max_len") or 0)
        value = args.get(slot)
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                missing.append(str(slot))
            continue
        if not isinstance(value, str):
            invalid[str(slot)] = "not_a_string"
            continue
        if max_len and len(value.strip()) > max_len:
            invalid[str(slot)] = "max_len"
    extra = sorted(set(args) - set((schema or {}).keys()))
    return {"missing": missing, "invalid": invalid, "extra": extra}


def _tool_roster(deps) -> dict[str, dict[str, dict]] | None:
    """tool -> {slot: {"max_len": int, "required": bool}} projected from the
    live ``ToolRuntime.schemas()`` — the one tool existence/schema truth since
    the ruling (max_len 0 = the schema states no length bound). A
    schema with no ``required`` list is treated as fully required (fail
    closed). None when no runtime is reachable — callers fail closed honestly."""
    runtime = getattr(getattr(deps, "agent", None), "runtime", None)
    if runtime is None:
        return None
    out: dict[str, dict[str, dict]] = {}
    for s in runtime.schemas():
        params = s.get("parameters") or {}
        props = params.get("properties") or {}
        req = params.get("required")
        required_all = req is None
        out[str(s["name"])] = {
            str(k): {
                "max_len": int((v or {}).get("maxLength") or 0),
                "required": bool(required_all or (k in (req or []))),
                # Reporting-only fields: ``validate_action`` reads max_len and
                # required, while the Agent-facing contract record needs the real
                # type (notably array ``paths``) to explain a mismatch.
                "type": str((v or {}).get("type") or "string"),
                "items": (v or {}).get("items") or {},
            }
            for k, v in props.items()
        }
    return out


class ActionExecutor:
    kind = PlanKind.ACTION

    async def _dispatch(self, req: TurnRequest) -> DispatchOutcome:
        """Run the staged dispatch; returns the terminal outcome. Raises
        :class:`EscalateToAgent` ONLY on proven pre-execution user-input failure."""
        action = req.plan.action or {}
        tool = action.get("tool")
        args = action.get("args")
        cap_id = str(action.get("capability_id") or "")

        # Tool existence/schema truth (ruling): the LIVE
        # ``ToolRuntime.schemas()`` roster — the legacy DIRECT_TOOLS table is
        # no longer consulted anywhere on the dispatch path.
        roster = _tool_roster(req.deps)

        # 0. FINAL EXECUTION AUTHORIZATION. This is the last gate before the side
        #    effect and the ONLY one that answers "may this Chat process run this
        #    capability". It reads the authoritative live Registry, not the action
        #    payload: a model pick, a routing stamp, or a hand-built plan is not
        #    authority. Failure to establish capability admission is a refusal, never an
        #    allow (fail-closed).
        try:
            from core.application.chat.intent_funnel.registry import active_view
            from core.application.chat.intent_funnel.registry.entry import STATUS_ACTIVE
            from core.application.chat.intent_funnel import funnel as funnel_mod

            view = await active_view(session_factory=req.deps.session_factory)
        except Exception:  # noqa: BLE001 - unreadable registry ⇒ refuse
            logger.exception(
                "chat.action authorization: registry unavailable (capability=%s)", cap_id)
            return _refusal(REASON_REGISTRY_UNAVAILABLE)
        if view is None:
            logger.warning(
                "chat.action authorization: empty registry view (capability=%s)", cap_id)
            return _refusal(REASON_REGISTRY_UNAVAILABLE)

        try:
            decision = authorize_execution(view, capability_id=cap_id, tool=tool)
        except Exception:  # noqa: BLE001 - unconfirmable permission ⇒ refuse
            logger.exception(
                "chat.action authorization failed (capability=%s tool=%s)", cap_id, tool)
            return _refusal(REASON_REGISTRY_UNAVAILABLE)
        if not decision.allowed:
            logger.warning(
                "chat.action refused reason=%s capability=%s tool=%s",
                decision.reason, cap_id, tool,
            )
            return _refusal(str(decision.reason))

        entry = decision.entry

        # 0.1 Preserved route/execute TOCTOU re-validation: same Registry content
        #     fingerprint as the one stamped at routing time, entry still enabled
        #     and active, same tool binding, tool present on the live ToolRuntime
        #     roster, and the kind gate still open. Any drift is C3 TERMINAL: a
        #     routed turn never executes on blind trust. (The lifecycle/binding
        #     checks above are also enforced by the authorization decision; they
        #     stay here as defense in depth for the routing stamp contract.)
        funnel_fp = action.get("funnel_registry_version")
        if funnel_fp is not None and view.fingerprint != str(funnel_fp):
            logger.warning(
                "chat.action route-stale fingerprint(capability=%s stamped=%s active=%s)",
                cap_id, funnel_fp, view.fingerprint,
            )
            return _refusal(REASON_REGISTRY_VERSION_MISMATCH)
        if entry is None or not entry.enabled or entry.status != STATUS_ACTIVE:
            return _refusal(REASON_CAPABILITY_INACTIVE)
        if entry.tool_binding != tool:
            return _refusal(REASON_TOOL_BINDING_MISMATCH)
        if roster is None:
            logger.error(
                "chat.action integrity: live ToolRuntime roster unavailable (tool=%s)", tool)
            return _refusal(REASON_TOOL_ROSTER_UNAVAILABLE)
        if str(tool or "") not in roster:
            logger.warning(
                "chat.action route-stale roster (capability=%s tool=%s)", cap_id, tool)
            return _refusal(REASON_TOOL_NOT_ON_ROSTER,
                            stamped=funnel_fp is not None)
        if not funnel_mod.kind_enabled(entry.intent_kind):
            logger.warning(
                "chat.action route-stale kind (capability=%s kind=%s)",
                cap_id, entry.intent_kind,
            )
            return _refusal(REASON_KIND_DISABLED)

        # 0.5 Routing-stage binding integrity (stage-2 C2, stamped in
        #     resolve_plan): the route promised a capability whose binding the
        #     action table does not honor. Decided TERMINAL — the Agent must not
        #     re-plan around a system inconsistency. Checked BEFORE the schema gate
        #     because such an action intentionally carries no args.
        if action.get("binding_integrity"):
            logger.error(
                "chat.action binding-integrity tool=%s: %s",
                tool, action.get("binding_integrity"),
            )
            return _refusal(REASON_BINDING_INTEGRITY)

        # 1. final schema gate, BEFORE the seam — malformed ⇒ nothing executed.
        #    Argument-shape problems are the user-input class: escalate and let the
        #    Agent clarify. A missing roster fails closed through the schema error.
        try:
            validated = validate_action(str(tool or ""),
                                        args if isinstance(args, dict) else {},
                                        tool_schemas=roster)
        except ActionSchemaError as exc:
            # A proven pre-body user-input failure. Hand the Agent the structured
            # contract evidence, not just an error string: nothing was executed,
            # so acting on the request now cannot duplicate a side effect.
            mismatch = _schema_mismatch(
                args, roster.get(str(tool or ""), {}))
            publish_failure(req.ctx, build_failure(
                kind=KIND_CONTRACT_MISMATCH,
                stage=STAGE_EXECUTOR,
                reason=REASON_EXECUTOR_PARAMETER_MISMATCH,
                capability_id=cap_id,
                tool_binding=str(tool or ""),
                query=str(getattr(req.ctx, "user_text", "") or ""),
                args=args if isinstance(args, dict) else {},
                missing=mismatch["missing"],
                invalid=mismatch["invalid"],
                extra=mismatch["extra"],
                required=roster.get(str(tool or ""), {}),
                facts=_turn_facts(req.ctx),
                detail=f"the runtime schema gate rejected the arguments: {exc}",
                original_error=str(exc),
                execution_state=EXECUTION_NOT_STARTED,
            ))
            raise EscalateToAgent(f"action schema: {exc}") from exc

        # 2. seam presence. A missing seam is an internal wiring fault (C2), NOT a
        #    user-input problem — terminate honestly; never re-plan through the Agent.
        if req.deps.run_tool is None:
            logger.error("chat.action integrity: run_tool seam not wired (tool=%s)", tool)
            return DispatchOutcome(_TERMINAL_INTEGRITY)

        # 3. the side-effect boundary.
        try:
            result = await req.deps.run_tool(validated["tool"], validated["args"], req.ctx)
        except ActionIntegrityFailure as exc:
            # C2: registry/runtime inconsistency (unknown tool, arg-schema drift).
            # Provably pre-body, but system faults must not launder through the
            # Agent as a retry mechanism.
            logger.error("chat.action integrity failure tool=%s: %s", validated["tool"], exc.reason)
            return DispatchOutcome(_TERMINAL_INTEGRITY)
        except ActionPreflightFailure as exc:
            # Proved: the body never ran. The Agent may clarify / re-plan. This is
            # NOT a parameter-contract mismatch (authorization / runtime preflight
            # can fail for reasons the user's arguments cannot fix), so it carries
            # its own kind and reason.
            publish_failure(req.ctx, build_failure(
                kind=KIND_EXECUTION_PREFLIGHT,
                stage=STAGE_EXECUTOR,
                reason=REASON_EXECUTOR_PREFLIGHT,
                capability_id=cap_id,
                tool_binding=str(validated["tool"]),
                query=str(getattr(req.ctx, "user_text", "") or ""),
                args=validated["args"],
                required=roster.get(str(validated["tool"]), {}),
                facts=_turn_facts(req.ctx),
                detail=exc.reason,
                original_error=exc.reason,
                execution_state=EXECUTION_NOT_STARTED,
            ))
            raise EscalateToAgent(f"preflight: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001 - STATE UNKNOWN by contract
            # STATE UNKNOWN is terminal and NEVER falls back to the Agent, so no
            # "safe to retry" context is published. A blind Agent replay here is
            # exactly how duplicate folders/terms are created.
            logger.warning("chat.action state-unknown tool=%s: %r", validated["tool"], exc)
            return DispatchOutcome(_STATE_UNKNOWN)

        # 4. decided denial (terminal, no side effect) or success.
        if result.get("ok") is False:
            return DispatchOutcome(
                _DENIED_PREFIX + str(result.get("reason") or "the operation was denied."))
        raw = str(result.get("output") or "Done.")
        # Query tools returned MATERIAL, not an answer. Synthesize one answer from
        # (question + material) — only after a successful dispatch, and a failed or
        # empty synthesis falls back to `raw` so the retrieved data is never lost.
        if validated["tool"] in SYNTHESIS_TOOLS:
            answer = await synthesize_answer(req.deps, req.ctx, validated["tool"], raw)
            if answer:
                return DispatchOutcome(answer)
        return DispatchOutcome(raw)

    # ── stream / run: identical terminal shape (any query synthesis already ran
    #    in _dispatch, so both entry points return the same final text) ────────────
    async def stream(
        self, req: TurnRequest, *, progress_sink: ProgressSink
    ) -> AsyncIterator[ChatEvent]:
        # Everything up to the first yield is pre-commit: an escalation here is legal.
        outcome = await self._dispatch(req)
        message = outcome.message
        ctx = req.ctx
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in (ctx.history or [])
            if isinstance(m.get("content"), str) and m.get("role") in ("user", "assistant")
        ]
        await ctx.session_memory.append_message("user", ctx.user_text)
        await ctx.session_memory.append_message("assistant", message)
        yield {"type": "content", "data": message}
        done = {
            "answer": message,
            "messages": [*history, {"role": "user", "content": ctx.user_text},
                         {"role": "assistant", "content": message}],
            "usage": {},
            "error": None,
            "cost_usd": None,
        }
        if outcome.refusal_reason:
            done["refusal_reason"] = outcome.refusal_reason
        yield {"type": "done", "data": done}

    async def run(self, req: TurnRequest) -> ExecutorResult:
        outcome = await self._dispatch(req)
        message = outcome.message
        ctx = req.ctx
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in (ctx.history or [])
            if isinstance(m.get("content"), str) and m.get("role") in ("user", "assistant")
        ]
        await ctx.session_memory.append_message("user", ctx.user_text)
        await ctx.session_memory.append_message("assistant", message)
        messages = [
            *history,
            {"role": "user", "content": ctx.user_text},
            {"role": "assistant", "content": message},
        ]
        return ExecutorResult(
            messages=messages, final_answer=message, usage={},
            refusal_reason=outcome.refusal_reason,
        )

"""Executor contracts: what the orchestrator hands a capability, and what comes back.

``ChatEvent`` is the same ``{"type": ..., "data": ...}`` dict the SSE transport has
always emitted (agent events, approvals, viewer short-circuit, done) — executors
produce events, the orchestrator forwards them VERBATIM so the frontend contract
(web + desktop) never changes.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from core.application.chat.execution_plan import ExecutionPlan, PlanKind

if TYPE_CHECKING:  # avoid an import cycle: context.py TYPE_CHECKING-imports base
    from core.application.chat.context import ChatTurnContext

# The wire payload of one SSE data frame — deliberately a plain dict (legacy shape).
ChatEvent = dict[str, Any]

ProgressSink = Callable[[ChatEvent], None]


@dataclass
class ExecutorResult:
    """The non-streaming result, shaped like the loop's ``AgentResult`` so lifecycle
    bookkeeping (finalize_turn reads ``answer``/``usage``/``messages``) is uniform."""

    messages: list[dict]
    final_answer: str
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    cost_usd: float | None = None
    # Structured fail-closed verdict for an execution-boundary refusal (ACTION lane).
    refusal_reason: str | None = None


@dataclass(frozen=True)
class ViewerDeps:
    """The pure assembly functions from ``apps/api/viewer_context.py`` (injected so
    this package never imports the api layer)."""

    build_blocks: Callable[..., dict]
    validate_citations: Callable[..., tuple[list, list]]
    citation_map: Callable[..., dict]
    snapshot: Callable[..., dict]
    # Renders the injected reference blocks into the grounded-prompt section (the
    # viewer fast path reuses the SAME renderer the Agent path's DYNAMIC_SUFFIX uses).
    render_reference: Callable[..., str] = lambda blocks: ""
    # Does THIS turn explicitly name the on-screen document (chat_assets §6.3)? Decides
    # whether to mount a ``reference`` attachment for the viewer's open file. Default
    # False so a core-only wiring never attaches merely because a viewer is open.
    requested_reference: Callable[..., bool] = lambda message, viewer: False
    # Merge a deferred client capture (page/frame screenshot + prepared text) into the
    # turn assembly on resume. Identity by default so a core-only wiring never rewrites
    # the assembly; the API layer injects the real pure function.
    with_capture: Callable[..., dict] = lambda assembly, capture: assembly
    # Merge a deferred full transcript (video FULL, client-parsed subtitles) into the
    # turn assembly on resume. Identity by default for the same reason as ``with_capture``.
    with_transcript: Callable[..., dict] = lambda assembly, payload: assembly


@dataclass
class ChatDeps:
    """Everything the control plane needs from the host process, resolved per request.

    The router wires these from its own module globals so existing test seams
    (monkeypatched ``get_agent`` / ``_log_usage`` / ``SessionLocal`` / …) keep
    working unchanged.
    """

    session_factory: Any
    queue: Any                                   # core.infrastructure.jobs.TaskQueue
    drive: Any                                   # core.application.drive_service.DriveService
    agent: Any                                   # AgentKernel instance
    llm: Any
    embedder: Callable[[], Any]
    viewer: ViewerDeps
    new_approval_bridge: Callable[[], Any]
    persist_turn_meta: Callable[[str | None, str, Any], Awaitable[None]]
    log_usage: Callable[..., Awaitable[None]]
    resolve_research: Callable[..., tuple]
    # The ACTION direct-dispatch seam: await run_tool(tool, args, ctx) ->
    #   {"ok": True, "output": str}   — registered tool executed, deterministic result;
    #   {"ok": False, "reason": str}  — a DECIDED, terminal denial (provable non-execution
    #                                    that the Agent clarifying would not change:
    #                                    an approval denial, a policy/sandbox guard);
    # raises ActionPreflightFailure  — ONLY when the seam can GUARANTEE the side effect
    #                                    did not happen (schema/未注册/"preflight:" errors
    #                                    the Agent may take over and clarify);
    # raises anything else           — the tool body was ENTERED and the state is
    #                                    UNKNOWN: the executor must terminate honestly
    #                                    and NEVER re-run the action via the Agent
    #                                    (no duplicate folder / duplicate term).
    # The host wires it through the SAME ToolRuntime.execute the Agent uses (approval,
    # guards, ACL pipeline inherited) — the fast path is flow control, never a second
    # capability registry. None = not wired → the ACTION branch degrades to the Agent.
    run_tool: Callable[[str, dict, Any], Awaitable[dict]] | None = None
    # Phase 4 acquisition seam: resolve the Argument Path Router's inputs for ONE
    # decided capability (declaration / evidence / system values+sources), i.e. an
    # ``argument_acquisition.inputs.AcquisitionInputs`` per capability id. None =
    # the cascade builds the PRODUCTION provider per turn from the live Registry
    # (``argument_acquisition.provider.for_turn``); a non-None value (tests) wins.
    # The cascade reads it duck-typed, so legacy deps objects (and deps=None) keep
    # working unchanged.
    acquisition_inputs: Any = None
    # Capability-selection seam for TESTS ONLY: a ``CapabilitySelector`` (the
    # deterministic StubSelector or a fake) that the cascade prefers over the
    # production cap_router service. None = the deployed service
    # (``cap_router.default_selector()``). There is no configuration switch and
    # no second selection model: production always uses the service, and a
    # selector failure exits to the Agent.
    cap_router_selector: Any = None
    # Phase 4 extraction seam: ``await extract(*, query, entry, model_slots,
    # bundle) -> (values, source)`` — the MODEL-owned argument extractor for one
    # already-decided capability (current implementation: Qwen, wired in
    # ``apps/api/routers/chat.py``). None = not wired → a MODEL acquisition need
    # exits ``ACQUISITION_MODEL_PENDING`` to the Agent (keeps stub/legacy deps and
    # existing tests green). It is turn-independent: the cascade calls it at the
    # acquisition hop with the turn's context bundle.
    argument_extractor: Any = None
    # Page-scope seam (file capabilities): ``await resolver(*, query, facts,
    # history, capability, asset_id) -> PageScope``. It asks the ONLINE LLM for
    # the sentence's reading scope (all / single / range) and returns a typed
    # verdict; the capability handlers map it to ``pages``. None = not wired ->
    # the handlers keep their legacy viewer-window rule. Document identity is
    # NOT part of this seam: ``asset_id`` still comes from TurnFacts.
    page_scope_resolver: Any = None
    # Deferred-capture seam (funnel-certified vision): the in-process broker that
    # pauses ONE streaming turn until the PC client uploads the page/frame capture.
    # None = not wired → a capture-required plan degrades honestly (the ACTION then
    # lacks its asset and the executor's schema gate decides), never a silent wait.
    capture_broker: Any = None


class EscalateToAgent(Exception):
    """Pre-commit fallback signal (design §5 Commit Point): an executor raises this
    BEFORE emitting any user-visible event to hand the turn back to the Agent.
    After the first content delta the channel is locked and this must never be raised.

    The ACTION executor uses it only for a provably pre-body user-input failure
    (schema / preflight); an authorization refusal is a terminal C2/C3 outcome and
    must never reach this path.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class TurnRequest:
    """One executor invocation: resolved context + the plan branch it must serve."""

    ctx: ChatTurnContext
    deps: ChatDeps
    plan: ExecutionPlan


class ChatExecutor(Protocol):
    """A capability branch. ``stream`` yields ChatEvents; ``run`` is the non-stream
    equivalent returning an AgentResult-shaped object.

    Executors must not emit the final ``done`` event — the orchestrator assembles it
    in :mod:`core.application.chat.lifecycle`.
    """

    kind: PlanKind

    def stream(self, req: TurnRequest, *, progress_sink: ProgressSink) -> AsyncIterator[ChatEvent]: ...

    async def run(self, req: TurnRequest) -> Any: ...

"""ExecutionPlan: the control plane's single execution contract + pure policy.

This module OWNS ``ExecutionPlan`` and the pure mapping function
``build_execution_plan(requirements)``. The mapping is in-memory only — no I/O, no
capability probing, no authorization (those live in Pre-flight and the
executors). ``turn_orchestrator`` is a pure consumer of the result.

Single path (ruling): ACTION is the only certified fast lane — a certified
``requested_action`` maps to ACTION, anything else maps to AGENT (fail-open).
The unshipped DIRECT / VIEWER / LOCAL_RAG / COMPOSITE lanes were deleted with
L0, their only requirement producer: no non-ACTION turn can reach HIGH
confidence any more, so keeping those branches would have been dead code.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.application.chat.understanding import Confidence, Signal, TurnRequirements

# Tools whose whole job IS the web: a web-demanding turn may only certify these
# (see _is_action_eligible). Names match the Registry's tool_binding values.
_WEB_FAMILY_TOOLS = frozenset({"web_search", "search_social"})


class PlanKind(str, Enum):
    ACTION = "action"
    AGENT = "agent"


@dataclass(frozen=True)
class ExecutionPlan:
    """What the orchestrator must do for this turn — the sole scheduling input.

    ``requires_memory`` only marks HYDRATION ELIGIBILITY; the authoritative recall
    decision stays with ``MemoryService.should_recall()``. ``action`` carries the
    normalized Action Request (ACTION kind only); final validation/authz happen in
    the executor, not here. ``reason`` is the decision trace for telemetry.

    ``source_policy`` records the ORIGINAL request's source restriction (``None``
    or ``"private_only"``); it is derived from user intent, NEVER from a fast-path
    failure, and is what fences the Agent's tools after escalation.
    """

    kind: PlanKind
    requires_memory: bool = False
    action: dict | None = None
    reason: str = ""
    source_policy: str | None = None
    # Deferred client capture (ACTION only): the funnel already certified the tool,
    # but its picture must be captured by the PC client. The orchestrator pauses the
    # stream, then resumes the SAME plan with the uploaded asset — never a re-route.
    capture_request: dict | None = None


def _source_policy(requirements) -> str | None:
    """Map the source FACTS to the policy that fences the escalated Agent.

    ``private_only`` is the only fence the current single path can produce: the
    original request explicitly restricts the turn to the user's own knowledge
    base. Everything else leaves the normal permission/approval funnel in charge —
    an ACTION abstention or failure never creates a web ban by itself.
    """
    if requirements.private_only and requirements.needs_web is not Signal.HIGH:
        return "private_only"
    return None


def _plan(requirements, kind: PlanKind, **kw) -> ExecutionPlan:
    return ExecutionPlan(kind=kind, source_policy=_source_policy(requirements), **kw)


def _agent(requirements, reason: str) -> ExecutionPlan:
    return _plan(requirements, PlanKind.AGENT, requires_memory=True, reason=reason)


def build_execution_plan(requirements: TurnRequirements) -> ExecutionPlan:
    """Pure mapping: TurnRequirements -> ExecutionPlan.

    Static rules:
      * anything not HIGH-confidence, any AMBIGUOUS/ABSTAIN or capability conflict
        -> AGENT (fallback);
      * only a fully certified ACTION is eligible for direct dispatch;
      * side-effect actions with unregistered templates -> AGENT or explicit error.

    An Intent Funnel admission reject leaves ``requested_action`` unset and the
    original requirements byte-identical, so this mapper sends the turn to AGENT
    like every other Intent Funnel abstain. Admission is an Intent Funnel rule,
    not a system-wide execution ban; the Agent's own roster/permissions govern it.
    """
    if requirements.confidence is not Confidence.HIGH:
        return _agent(requirements, f"confidence={requirements.confidence.value} -> agent")

    # ── ACTION (typed direct dispatch over the registered tool registry) ──────────
    # A certified single registered tool with fully-determined args (funnel cascade).
    # An unregistered/ambiguous/under-parameterized turn never gets
    # ``requested_action`` set, so it maps to AGENT (fail-open) — the policy mapper
    # validates NOTHING (schema gate + authz live in the executor and the host seam).
    if _is_action_eligible(requirements):
        return _plan(
            requirements, PlanKind.ACTION, action=dict(requirements.requested_action),
            requires_memory=False,
            capture_request=requirements.capture_request,
            reason="certified single registered tool, args fully determined -> action",
        )

    return _agent(requirements, "no certified action for this requirement set -> agent")


def _is_action_eligible(requirements: TurnRequirements) -> bool:
    """The ACTION guard: the funnel certified a registered-tool direct request.
    ``requested_action`` is ONLY ever set by the funnel's ``certified()``, so
    presence is the full contract; no second registry check here (the executor's
    execution-boundary authorization and schema gate own that).

    Web demand (E2E-matrix ruling): the blanket web veto moved off the funnel
    ENTRY and lands HERE, where the certified tool is known — a web-demanding turn
    may take the ACTION lane only when the certified tool is itself web-family
    (``web_search``/``search_social``). A composite like "新建文件夹并查新闻"
    certifies a NON-web tool while ``needs_web`` stays HIGH, so it still exits to
    the Agent — the single-capability contract is preserved exactly where it was
    before, just evaluated with more information."""
    action = requirements.requested_action
    web_admissible = (requirements.needs_web is Signal.LOW
                      or (action or {}).get("tool") in _WEB_FAMILY_TOOLS)
    return (
        requirements.needs_action is Signal.HIGH
        and action is not None
        and web_admissible
        and not requirements.needs_memory
    )

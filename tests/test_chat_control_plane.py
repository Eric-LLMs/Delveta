"""Control-plane contracts under the single-path ruling.

The router-level SSE compatibility is already covered end-to-end by
test_viewer_chat (TestClient + fake kernel asserts frame shapes). What is pinned
here is the pure policy surface:

* ``TurnRequirements`` defaults to an abstain;
* the default policy (all experimental lanes off) keeps routing on AGENT — the
  formal path certifies ACTION only, everything else rides its own lane gate;
* the pure mapper still maps a HIGH-confidence requirement set to its lane under
  that lane's gate; only the funnel's Matcher/Recall cascade can produce such a
  set now that L0's signal engine is gone;
* the orchestrator's registry resolves the right branch and degrades unmapped kinds
  to the agent branch, never hard-failing.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.application.chat.execution_plan import (
    ExecutionPlan,
    PlanKind,
    PolicyContext,
    build_execution_plan,
)
from core.application.chat.understanding import (
    Confidence,
    Signal,
    TurnRequirements,
)


def test_default_requirements_abstain():
    req = TurnRequirements()
    assert req.confidence is Confidence.ABSTAIN
    assert req.needs_private is Signal.LOW
    assert req.needs_memory is False


def test_default_policy_keeps_undemanded_turns_on_agent():
    # Neutral (abstain) input maps to AGENT; a HIGH-confidence input with no
    # certified action and no lane gate open ALSO lands on AGENT — the experimental
    # lanes stay dark by default (Option A, single-path ruling).
    neutral = build_execution_plan(TurnRequirements(), PolicyContext())
    assert neutral.kind is PlanKind.AGENT

    strong = TurnRequirements(
        needs_private=Signal.LOW, needs_web=Signal.LOW,
        needs_viewer=Signal.LOW, needs_action=Signal.LOW,
        confidence=Confidence.HIGH,
    )
    # Default policy (DIRECT gate closed) → still AGENT (per-kind gating).
    assert build_execution_plan(strong, PolicyContext()).kind is PlanKind.AGENT


def test_direct_kind_maps_only_under_direct_gate():
    clean = TurnRequirements(confidence=Confidence.HIGH)  # all needs default LOW
    both_on = PolicyContext(direct_fast_path_enabled=True)
    assert build_execution_plan(clean, both_on).kind is PlanKind.DIRECT
    # A demand present disqualifies DIRECT even with the gate on.
    demanding = TurnRequirements(
        confidence=Confidence.HIGH, needs_web=Signal.HIGH,
    )
    assert build_execution_plan(demanding, both_on).kind is PlanKind.AGENT


# ── Phase 3 VIEWER policy mapping ────────────────────────────────────────────────

def _viewer_req(**over):
    kw = {"needs_viewer": Signal.HIGH, "confidence": Confidence.HIGH}
    kw.update(over)
    return TurnRequirements(**kw)


def test_viewer_kind_maps_under_viewer_gate_only():
    all_on = PolicyContext(
        direct_fast_path_enabled=True, viewer_fast_path_enabled=True,
    )
    plan = build_execution_plan(_viewer_req(), all_on)
    assert plan.kind is PlanKind.VIEWER and plan.requires_viewer is True
    # Viewer gate OFF → a viewer-demand turn (needs_viewer HIGH) is not DIRECT either,
    # so it falls through to AGENT.
    only_direct = PolicyContext(direct_fast_path_enabled=True)
    assert build_execution_plan(_viewer_req(), only_direct).kind is PlanKind.AGENT


def test_viewer_demand_with_other_capability_stays_agent():
    all_on = PolicyContext(
        direct_fast_path_enabled=True, viewer_fast_path_enabled=True,
    )
    # The viewer must be the SOLE demand — a co-occurring private/web/memory need keeps
    # the turn on the Agent (which owns read_document / rag / web / recall).
    for over in (
        {"needs_private": Signal.HIGH},
        {"needs_web": Signal.HIGH},
        {"needs_action": Signal.HIGH},
        {"needs_memory": True},
    ):
        assert build_execution_plan(_viewer_req(**over), all_on).kind is PlanKind.AGENT, over


# ── Phase 4 LOCAL_RAG policy mapping ───────────────────────────────────────────────

def _rag_req(**over):
    kw = {"needs_private": Signal.HIGH, "confidence": Confidence.HIGH}
    kw.update(over)
    return TurnRequirements(**kw)


def _all_on():
    return PolicyContext(
        direct_fast_path_enabled=True,
        viewer_fast_path_enabled=True, retrieval_fast_path_enabled=True,
    )


def test_retrieval_kind_maps_under_retrieval_gate_only():
    plan = build_execution_plan(_rag_req(), _all_on())
    assert plan.kind is PlanKind.LOCAL_RAG and plan.requires_retrieval is True
    # Everything else ON but the retrieval gate closed → still AGENT (per-kind gating).
    no_rag = PolicyContext(
        direct_fast_path_enabled=True, viewer_fast_path_enabled=True,
    )
    assert build_execution_plan(_rag_req(), no_rag).kind is PlanKind.AGENT


def test_private_plus_other_demand_stays_agent_fail_closed():
    # A co-occurring web demand never mixes public with private on the fast path —
    # the Agent arbitrates. Same for viewer/action/memory co-demands.
    for over in (
        {"needs_web": Signal.HIGH},
        {"needs_viewer": Signal.HIGH},
        {"needs_action": Signal.HIGH},
        {"needs_memory": True},
    ):
        assert build_execution_plan(_rag_req(**over), _all_on()).kind is PlanKind.AGENT, over


def test_reason_is_traceable():
    assert "agent" in build_execution_plan(TurnRequirements(), PolicyContext()).reason


def test_l0_signal_engine_is_gone():
    # L0 retirement: the module owns source facts + the requirements contract only.
    # No lexical or context-based route certification may reappear here.
    from core.application.chat import understanding

    assert not hasattr(understanding, "resolve_requirements")
    assert not hasattr(understanding, "_viewer_ground_eligible")
    assert not hasattr(understanding, "_lex_action")
    assert not hasattr(understanding, "_memory_trigger")


# ── orchestrator registry ─────────────────────────────────────────────────────────


def _ctx(*, message="hi"):
    from core.application.chat.context import ChatTurnContext
    return ChatTurnContext(
        body=SimpleNamespace(message=message, attach=None),
        user=None, user_id="u", guest_token=None, log_user=None, tier="free",
        notice=None, model=None, base_url=None, api_key=None, business_name=None,
        credential_id=None, user_text=message, owned_asset_id=None, inline_image=None,
        viewer_assembly=None, research_turn=False, effective_handoff=None,
    )


def test_orchestrator_resolves_agent_plan_and_executor():
    from core.application.chat.executors.agent import AgentExecutor
    from core.application.chat.executors.direct import DirectExecutor
    from core.application.chat.executors.retrieval import RetrievalExecutor
    from core.application.chat.turn_orchestrator import TurnOrchestrator

    ctx = _ctx(message="hi")
    orch = TurnOrchestrator(deps=None)  # no deps: funnel fails open, resolution touches no deps
    import asyncio
    plan = asyncio.run(orch.resolve_plan(ctx))  # default lanes dark → AGENT (async since QIR stage 1)
    assert plan.kind is PlanKind.AGENT
    assert isinstance(orch.executor_for(plan), AgentExecutor)
    # DIRECT / VIEWER / LOCAL_RAG are now registered branches...
    assert isinstance(orch.executor_for(ExecutionPlan(kind=PlanKind.DIRECT)), DirectExecutor)
    assert isinstance(orch.executor_for(ExecutionPlan(kind=PlanKind.LOCAL_RAG)), RetrievalExecutor)
    # ...and Phase 5 registered ACTION + COMPOSITE alongside them.
    from core.application.chat.executors.action import ActionExecutor
    from core.application.chat.executors.composite import CompositeExecutor
    assert isinstance(orch.executor_for(ExecutionPlan(kind=PlanKind.ACTION)), ActionExecutor)
    assert isinstance(orch.executor_for(ExecutionPlan(kind=PlanKind.COMPOSITE)), CompositeExecutor)
    # Fail-Closed is STRUCTURAL: no WEB executor is registered anywhere, so a private
    # turn's EscalateToAgent hand-back can only land on the Agent — the control plane
    # physically cannot route LOCAL_RAG -> WEB, and the Agent's own web use stays a
    # visible agent-level decision exactly as before Phase 4. WEB is also the example
    # of an unmapped kind: executor_for must degrade it to AGENT, never raise.
    assert isinstance(orch.executor_for(ExecutionPlan(kind=PlanKind.WEB)), AgentExecutor)

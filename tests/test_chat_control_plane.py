"""Control-plane contracts under the single-path ruling.

The router-level SSE compatibility is covered end-to-end elsewhere. What is pinned
here is the pure policy surface:

* ``TurnRequirements`` defaults to an abstain;
* only an Intent-Funnel-certified ``requested_action`` maps to ACTION; every other
  requirement set — including HIGH-confidence non-action demands — maps to AGENT;
* the orchestrator registry contains exactly ACTION + AGENT; an
  unmapped kind degrades to AGENT, never a hard fail.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.application.chat.execution_plan import (
    ExecutionPlan,
    PlanKind,
    build_execution_plan,
)
from core.application.chat.understanding import (
    Confidence,
    Signal,
    TurnRequirements,
)

ACTION = {
    "tool": "create_folder",
    "args": {"name": "x"},
    "capability_id": "cap-folder",
    "funnel_registry_version": "live1-deadbeef",
}


def test_default_requirements_abstain():
    req = TurnRequirements()
    assert req.confidence is Confidence.ABSTAIN
    assert req.needs_private is Signal.LOW
    assert req.needs_memory is False


def test_uncertified_turns_stay_on_agent():
    neutral = build_execution_plan(TurnRequirements())
    assert neutral.kind is PlanKind.AGENT

    strong = TurnRequirements(
        needs_private=Signal.LOW, needs_web=Signal.LOW,
        needs_viewer=Signal.LOW, needs_action=Signal.LOW,
        confidence=Confidence.HIGH,
    )
    assert build_execution_plan(strong).kind is PlanKind.AGENT


def test_certified_action_maps_to_action():
    req = TurnRequirements(
        confidence=Confidence.HIGH, needs_action=Signal.HIGH,
        requested_action=ACTION,
    )
    plan = build_execution_plan(req)
    assert plan.kind is PlanKind.ACTION
    assert plan.action == ACTION
    assert plan.action is not ACTION          # a copy, never the live requirements dict


def test_web_demand_admits_only_web_family_actions():
    news = {**ACTION, "tool": "web_search"}
    req = TurnRequirements(
        confidence=Confidence.HIGH, needs_action=Signal.HIGH,
        needs_web=Signal.HIGH, requested_action=ACTION,
    )
    assert build_execution_plan(req).kind is PlanKind.AGENT
    assert build_execution_plan(
        TurnRequirements(
            confidence=Confidence.HIGH, needs_action=Signal.HIGH,
            needs_web=Signal.HIGH, requested_action=news,
        )
    ).kind is PlanKind.ACTION


def test_memory_eligible_action_stays_on_agent():
    req = TurnRequirements(
        confidence=Confidence.HIGH, needs_action=Signal.HIGH,
        needs_memory=True, requested_action=ACTION,
    )
    assert build_execution_plan(req).kind is PlanKind.AGENT


def test_private_only_source_policy_rides_the_plan():
    req = TurnRequirements(
        confidence=Confidence.HIGH, needs_action=Signal.HIGH,
        private_only=True, requested_action=ACTION,
    )
    assert build_execution_plan(req).source_policy == "private_only"


def test_reason_is_traceable():
    assert "agent" in build_execution_plan(TurnRequirements()).reason
    req = TurnRequirements(
        confidence=Confidence.HIGH, needs_action=Signal.HIGH,
        requested_action=ACTION,
    )
    assert "action" in build_execution_plan(req).reason


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
    from core.application.chat.executors.action import ActionExecutor
    from core.application.chat.executors.agent import AgentExecutor
    from core.application.chat.turn_orchestrator import TurnOrchestrator

    ctx = _ctx(message="hi")
    orch = TurnOrchestrator(deps=None)  # no deps: Intent Funnel fails open, resolution touches no deps
    import asyncio
    plan = asyncio.run(orch.resolve_plan(ctx))  # uncertified -> AGENT
    assert plan.kind is PlanKind.AGENT
    assert isinstance(orch.executor_for(plan), AgentExecutor)
    assert isinstance(orch.executor_for(ExecutionPlan(kind=PlanKind.ACTION)), ActionExecutor)
    # Only the two formal branches exist; an unmapped kind degrades to AGENT.
    assert set(orch._executors) == {PlanKind.ACTION, PlanKind.AGENT}
    unmapped = SimpleNamespace(kind="legacy_lane")
    assert isinstance(orch.executor_for(unmapped), AgentExecutor)

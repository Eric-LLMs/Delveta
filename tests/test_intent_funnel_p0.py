"""P0 intent_funnel package tests: contracts + the orchestration's seams.

The live-table ruling deleted the legacy QIR lane; the
single-path ruling deleted the dark-launch rollout gates: these tests pin the
fail-open shapes (absent deps -> route returns the SAME object), the deps-only
gate, and the node contracts the single-hop chain runs on. Every wired turn
reaches the Matcher; no entry rule may re-route it first.
"""
import types

from core.application.chat import intent_funnel
from core.application.chat.intent_funnel import contract
from core.application.chat.intent_funnel.funnel import funnel_live, route
from core.application.chat.understanding import Signal, TurnRequirements
from core.config import settings

# ── route(): fail-open stays byte-identical ──────────────────────────────────────


async def test_route_without_deps_returns_same_requirements_object(monkeypatch):
    # Single-path ruling: the only route-level fail-open door left is missing
    # deps (control plane not wired) -> route must be identity.
    req = TurnRequirements()
    ctx = types.SimpleNamespace(body=types.SimpleNamespace(message="create a folder x"))
    out = await route(ctx, deps=None, requirements=req)
    assert out is req


async def test_route_delegates_when_gate_live(monkeypatch):
    monkeypatch.setattr(intent_funnel.funnel, "funnel_live", lambda d: True)

    sentinel = TurnRequirements(needs_action=Signal.HIGH)

    async def fake_cascade(ctx, deps, requirements):
        return sentinel

    monkeypatch.setattr(intent_funnel.funnel, "_cascade", fake_cascade)
    ctx = types.SimpleNamespace(body=types.SimpleNamespace(message="create a folder x"))
    out = await route(ctx, deps=object(), requirements=TurnRequirements())
    assert out is sentinel


async def test_route_skips_cascade_when_fail_open(monkeypatch):
    monkeypatch.setattr(intent_funnel.funnel, "funnel_live", lambda d: False)

    async def boom_cascade(ctx, deps, requirements):  # must never run
        raise AssertionError("fail-open exit but cascade executed")

    monkeypatch.setattr(intent_funnel.funnel, "_cascade", boom_cascade)
    req = TurnRequirements()
    ctx = types.SimpleNamespace(body=types.SimpleNamespace(message="hi"))
    assert await route(ctx, deps=object(), requirements=req) is req


async def test_route_has_no_entry_veto(monkeypatch):
    # L0 retirement: nothing in the requirements can short-circuit the cascade.
    # The old L0-certified early return and the turn_veto gate are gone — the
    # funnel_cascade always gets the turn when deps are wired.
    seen = []

    async def fake_cascade(ctx, deps, requirements):
        seen.append(requirements)
        return requirements

    monkeypatch.setattr(intent_funnel.funnel, "_cascade", fake_cascade)
    req = TurnRequirements(requested_action={"tool": "t", "args": {}})
    ctx = types.SimpleNamespace(body=types.SimpleNamespace(message="create a folder x"))
    assert await route(ctx, deps=object(), requirements=req) is req
    assert seen == [req]


# ── funnel_live(): the fail-open door ───────────────────────────────────────────


def test_gate_is_deps_only(monkeypatch):
    # Single-path ruling: no rollout switch or entry veto can close the gate any
    # more — it is open whenever deps are wired; absent deps is the only door,
    # and it fails open to the Agent.
    assert funnel_live(deps=object()) is True
    assert funnel_live(deps=None) is False


# ── contracts ────────────────────────────────────────────────────────────────────


def test_bound_arguments_adapter_maps_legacy_return():
    complete = contract.BoundArguments.of({"name": "资料"})
    assert complete.state == contract.BIND_COMPLETE
    assert complete.is_complete and not complete.is_missing
    assert complete.args == {"name": "资料"}
    missing = contract.BoundArguments.of(None)
    assert missing.state == contract.BIND_MISSING
    assert missing.is_missing and not missing.is_complete
    assert missing.args is None


def test_reason_code_constants_avoid_bare_ambiguous():
    # 8.10: prefixed codes only — the legacy Confidence.AMBIGUOUS stays untouched.
    assert contract.MATCH_AMBIGUOUS == "MATCH_AMBIGUOUS"
    assert contract.MATCH_AMBIGUOUS != "AMBIGUOUS"
    m = contract.MatchResult(state=contract.MATCH_AMBIGUOUS, candidates=("a", "b"))
    assert m.capability_id is None  # ambiguous: Matcher never picks (8.1)


def test_contracts_are_frozen():
    for cls in (
        contract.MatchResult, contract.Candidate, contract.RecallResult,
        contract.BoundArguments, contract.TurnFacts,
    ):
        assert cls.__dataclass_params__.frozen, cls.__name__
    # chain ruling: the Decision node is gone from the contract, and
    # with it the second-hop recheck entry point.
    assert not hasattr(contract, "DecisionResult")
    # single-path ruling: the legacy ToolIntentModel package, its verdict type
    # and the selector backend switch are all deleted — one production lane.
    from core.application.chat.intent_funnel import cap_router
    from core.application.chat.intent_funnel import funnel as funnel_mod
    from core.config import settings
    assert not hasattr(contract, "ToolIntentVerdict")
    assert not hasattr(funnel_mod, "tool_intent")
    assert not hasattr(cap_router, "BACKENDS")
    assert not hasattr(settings, "chat_cap_router_backend")
    # naming ruling: "Model A" was a placeholder and Judge/Decision
    # were historical.
    bad = ("judge", "model_a", "modela", "decision")
    assert not [n for n in dir(contract) if any(b in n.lower() for b in bad)]
    # live-table ruling: the QIR adapter lane is deleted outright —
    # no IntentVerdict/AgentFallback survivors, the funnel certifies by
    # returning an ACTION TurnRequirements or the fail-open original.
    assert not hasattr(contract, "IntentVerdict")
    assert not hasattr(contract, "AgentFallback")


async def test_orchestrator_calls_funnel_once_with_requirements(monkeypatch):
    """The P0 shape: orchestrator keeps the lifecycle, funnel owns the routing."""
    from core.application.chat.turn_orchestrator import TurnOrchestrator

    calls = []

    async def fake_route(ctx, *, deps, requirements):
        calls.append(requirements)
        return requirements

    monkeypatch.setattr(intent_funnel, "route", fake_route)
    orch = TurnOrchestrator(deps=None)
    ctx = types.SimpleNamespace(
        body=types.SimpleNamespace(message="hello"),
        agent_context=None,
    )
    # Single-path ruling: resolve_plan always resolves + routes — no gate to open.
    await orch.resolve_plan(ctx, deps=None)
    assert len(calls) == 1
    assert isinstance(calls[0], TurnRequirements)

"""Boundary fault injection at the real dispatch seam (Phase 4-B).

Each case injects ONE fault at a REAL seam of the A-1 chain and pins the SAFE
degradation the boundary must produce: an honest terminal state, a provable
zero/one side effect (observed through ``EffectRecorder``), and never a blind
replay. The executor's terminal texts are imported verbatim from
``executors/action.py`` so an assertion cannot drift from production wording.

The last three cases (F1, F2, F2′), added in 4-B as **TRUE RED**, are the TARGET
contracts that 4-C hardened. They were plain failing assertions (never ``xfail``)
so the defect stayed visible; they now pass against the hardened runtime and stay
as regression locks — an edit to the fix that re-opens the gap turns them red again.

  E1  route→dispatch registry fingerprint drift  → _TERMINAL_STALE_ROUTE, 0 effect
  E2  routing-stage binding-integrity marker      → _TERMINAL_INTEGRITY, 0 effect
  E4  unwired dispatch seam (run_tool is None)    → _TERMINAL_INTEGRITY, 0 effect
  R2  no approver bound                           → decided DENY, 0 effect
  R3  a raising guard                             → fail-closed DENY, 0 effect
  R4  a raising pre-execute hook                  → blocked before the body, 0 effect
  R5  a mutating tool blocked post-execute        → _STATE_UNKNOWN, exactly ONE effect
  R6  a raising output renderer                   → _STATE_UNKNOWN, exactly ONE effect
  R7  a raising tools/result observer             → isolated; the turn is unaffected
  F1  approval chain raises                        → decided DENY (fixed safe text), 0 effect
  F2  a MUTATING body overruns its deadline        → _STATE_UNKNOWN, 0 effect, no retry
  F2′ a NON-MUTATING body overruns its deadline    → denied terminal (NOT _STATE_UNKNOWN)
"""
from __future__ import annotations

import asyncio

import pytest
from agent.engine.decisions import PostToolDecision
from core.application.chat.executors.action import (
    _DENIED_PREFIX,
    _STATE_UNKNOWN,
    _TERMINAL_INTEGRITY,
    _TERMINAL_STALE_ROUTE,
)

from tests.e2e.harness import (
    CAP_BY_ID,
    break_render,
    build_stack,
    drop_run_tool,
    hang_body,
    inject_guard,
    inject_observer,
    inject_waterfall,
    no_approver,
    raise_from_broker,
    sse_for,
    stamp_certified,
)

# Anti-hang watchdog for the F2 body-hang case: a hanging tool body must terminate
# within the runtime's own timeout contract; this bound only stops the SUITE from
# hanging forever (it is not the pass criterion).
_BODY_HANG_WATCHDOG_S = 0.3


# ── E1: route→dispatch TOCTOU drift ⇒ STALE terminal, zero effect ──────────────────
async def test_e1_registry_fingerprint_drift_is_terminal(monkeypatch):
    """The Registry content changed between routing and dispatch ⇒ the executor refuses
    the stamp and returns the STALE terminal; the seam is never entered."""
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)
    stamp_certified(monkeypatch, funnel_registry_version="stale-fp-e2e")

    res = await sse_for(stack, cap.message)

    assert res.answer == _TERMINAL_STALE_ROUTE
    assert stack.recorder.call_count == 0            # the dispatching seam was never reached
    assert stack.spy.web_queries == []
    assert stack.port.steps == 0


# ── E2: routing-stage binding-integrity marker ⇒ INTEGRITY terminal, zero effect ────
async def test_e2_binding_integrity_is_terminal(monkeypatch):
    """A routing-stage binding-integrity stamp is a C2 system fault ⇒ honest terminal,
    seam never entered, and NEVER laundered through the Agent as a retry channel."""
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)
    stamp_certified(monkeypatch, binding_integrity="e2e-injected-drift")

    res = await sse_for(stack, cap.message)

    assert res.answer == _TERMINAL_INTEGRITY
    assert stack.recorder.call_count == 0
    assert stack.spy.web_queries == []
    assert stack.port.steps == 0


# ── E4: unwired dispatch seam ⇒ INTEGRITY terminal, zero effect ────────────────────
async def test_e4_unwired_run_tool_is_terminal(monkeypatch):
    """``deps.run_tool is None`` is an internal wiring fault (C2) ⇒ honest terminal,
    not a user-input escalation."""
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)
    drop_run_tool(monkeypatch)

    res = await sse_for(stack, cap.message)

    assert res.answer == _TERMINAL_INTEGRITY
    assert stack.recorder.call_count == 0
    assert stack.spy.web_queries == []
    assert stack.port.steps == 0


# ── R2: no approver bound ⇒ the ASK degrades to a decided DENY, zero effect ─────────
async def test_r2_missing_approver_degrades_to_deny(monkeypatch):
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)
    no_approver(stack.runtime)

    res = await sse_for(stack, cap.message)

    assert (res.answer or "").startswith(_DENIED_PREFIX)
    assert stack.spy.folders_created == []           # the body never ran
    assert stack.recorder.call_count == 1            # the blocked attempt was recorded …
    assert stack.recorder.calls[0]["is_error"] is True   # … as an error, not a fake success
    assert stack.port.steps == 0


# ── R3: a raising guard ⇒ fail-closed DENY, zero effect ─────────────────────────────
async def test_r3_raising_guard_fails_closed(monkeypatch):
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)

    async def _boom_guard(exec_):
        raise RuntimeError("guard boom (fault injection)")

    inject_guard(stack.runtime, _boom_guard)

    res = await sse_for(stack, cap.message)

    assert (res.answer or "").startswith(_DENIED_PREFIX)
    assert stack.spy.folders_created == []
    assert stack.recorder.calls[0]["is_error"] is True
    assert stack.port.steps == 0


# ── R4: a raising pre-execute hook ⇒ blocked BEFORE the body, zero effect ───────────
async def test_r4_raising_pre_execute_hook_blocks_before_the_body(monkeypatch):
    """A broken pre-execute hook is pre-body: the body must not run. For a non-mutating
    tool the honest degradation is an escalation (the Agent may clarify) — never a
    side effect from an uncontrolled middleware."""
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)

    async def _boom_pre(exec_, next_):
        raise RuntimeError("pre-execute boom (fault injection)")

    inject_waterfall(stack.runtime, "tools/pre-execute", _boom_pre)

    res = await sse_for(stack, cap.message)

    assert stack.spy.web_queries == []               # the body never ran
    assert stack.recorder.calls[0]["is_error"] is True
    assert stack.port.steps >= 1                     # the Agent took the turn
    assert res.approvals == []


# ── R5: a mutating tool blocked post-execute ⇒ _STATE_UNKNOWN, ONE effect ───────────
async def test_r5_mutating_post_block_is_state_unknown(monkeypatch):
    """The body ran, then a post-execute block fired ⇒ the effect exists, so the honest
    terminal is STATE UNKNOWN and the executor must NOT replay through the Agent."""
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)

    async def _block(exec_, result, next_):
        return PostToolDecision.block("blocked by test (fault injection)")

    inject_waterfall(stack.runtime, "tools/post-execute", _block)

    res = await sse_for(stack, cap.message)

    assert res.answer == _STATE_UNKNOWN
    assert len(stack.spy.folders_created) == 1       # exactly ONE effect, never two
    assert stack.port.steps == 0                     # the Agent was never re-run


# ── R6: a raising output renderer ⇒ _STATE_UNKNOWN, ONE effect ──────────────────────
async def test_r6_raising_render_is_state_unknown(monkeypatch):
    """The body completed, then its renderer raised ⇒ the effect exists ⇒ STATE UNKNOWN."""
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)
    break_render(stack.runtime, "create_folder")

    res = await sse_for(stack, cap.message)

    assert res.answer == _STATE_UNKNOWN
    assert len(stack.spy.folders_created) == 1
    assert stack.port.steps == 0


# ── R7: a raising tools/result observer ⇒ isolated; the turn is unaffected ──────────
async def test_r7_raising_result_observer_is_isolated(monkeypatch):
    """Observers are read-only: a raising ``tools/result`` observer must be logged and
    swallowed, never disturbing the turn (this is the channel EffectRecorder rides)."""
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)

    def _boom_observer(payload):
        raise RuntimeError("observer boom (fault injection)")

    inject_observer(stack.runtime, "tools/result", _boom_observer)

    res = await sse_for(stack, cap.message)

    assert stack.recorder.call_count == 1            # the recorder still observed the result
    assert stack.recorder.calls[0]["is_error"] is False
    assert stack.spy.web_queries == [cap.message]    # the real body ran exactly once
    assert res.answer                             # the turn completed normally


# ══ F1/F2: the 4-C-hardened contracts (were TRUE RED in 4-B, now regression locks) ══
async def test_f1_raising_approval_chain_degrades_to_a_decided_denial(monkeypatch):
    """The approval decision stage is ISOLATED: a raising approval chain fails closed.

    ``ToolRuntime.execute`` isolates every decision stage ("a raising hook can never
    escape the loop"); 4-C brought the approval resolution
    (``_resolve_ask`` → ``ApprovalBridge`` → ``ApprovalStore.request`` → ``broker.register``)
    under the same isolation. The approval runs BEFORE the tool body, so a raise is
    provably side-effect-free: the honest degradation is a DECIDED DENIAL, never the
    STATE UNKNOWN text (which would falsely imply the operation might have been applied).
    """
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)
    raise_from_broker(stack.broker)

    res = await sse_for(stack, cap.message)

    # CONTRACT: approval is pre-body ⇒ decided denial (NOT the UNKNOWN text).
    assert (res.answer or "").startswith(_DENIED_PREFIX), (
        f"a raising approval chain produced {res.answer!r}; the pre-body approval fault "
        f"must degrade to a decided denial ({_DENIED_PREFIX!r}), not STATE UNKNOWN."
    )
    assert res.answer != _STATE_UNKNOWN
    assert "fault injection" not in (res.answer or "")   # never leak the raw exception text
    assert stack.spy.folders_created == []           # zero side effect
    assert stack.port.steps == 0


async def test_f2_mutating_body_overrun_is_state_unknown(monkeypatch):
    """A MUTATING tool whose body overran its deadline ⇒ _STATE_UNKNOWN, no retry.

    The body was ENTERED then cancelled, so the write may or may not have landed — the
    honest terminal is STATE UNKNOWN and the executor must NEVER auto-retry (the same
    doctrine as a post-write failure). The watchdog is the test-layer anti-hang guard
    (the suite must never hang), not the pass criterion.
    """
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)
    hang_body(stack.runtime, "create_folder")

    try:
        res = await asyncio.wait_for(sse_for(stack, cap.message), timeout=_BODY_HANG_WATCHDOG_S)
    except TimeoutError:
        pytest.fail(
            "a mutating tool body overran its deadline and never returned — the runtime "
            "has no tool-body timeout. Contract: cancel within the runtime deadline and "
            "surface an honest terminal, not an unbounded wait."
        )

    assert res.answer == _STATE_UNKNOWN
    assert stack.spy.folders_created == []           # the body never completed a write
    assert stack.port.steps == 0                     # the Agent was never re-run


async def test_f2_non_mutating_body_overrun_is_not_state_unknown(monkeypatch):
    """A NON-MUTATING (read) tool whose body overran ⇒ an ordinary terminal failure.

    A read tool has no write side effect, so its timeout must NOT be reported as STATE
    UNKNOWN — it is a plain, decided failure surfaced honestly (and the Agent is not
    re-run to re-hang).
    """
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)
    hang_body(stack.runtime, "web_search")

    try:
        res = await asyncio.wait_for(sse_for(stack, cap.message), timeout=_BODY_HANG_WATCHDOG_S)
    except TimeoutError:
        pytest.fail(
            "a non-mutating tool body overran its deadline and never returned — the "
            "runtime has no tool-body timeout."
        )

    assert res.answer != _STATE_UNKNOWN               # a read timeout is never UNKNOWN
    assert (res.answer or "").startswith(_DENIED_PREFIX)
    assert stack.spy.web_queries == []               # the body never completed
    assert stack.port.steps == 0

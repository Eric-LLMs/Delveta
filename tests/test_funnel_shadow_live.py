"""Shadow-live A/B (2026-09-27): the full cascade observes REAL chat traffic in
a fire-and-forget task while the Agent path stays the sole execution route.

The invariants these tests pin:
* flag OFF (default) = byte-identical dark launch — no task, no AB lines;
* flag ON = the turn is still owned by the Agent (plan never sees the shadow),
  the verdict pair (``funnel_ab_shadow`` + ``funnel_ab_turn``) is emitted,
  and a broken observation can ONLY show up as log lines (fail-safe);
* research/handoff turns are excluded from the denominator by ruling;
* ``cascade_shadow`` persistence is additive: the default replay path stays
  byte-identical (no event row), ``persist_event=True`` writes inside the
  execution_mode=shadow pin with the turn_key join anchor in trace_json.
"""
from __future__ import annotations

import asyncio
import logging
import types

import pytest
from core.application.chat import turn_orchestrator as tor
from core.application.chat.intent_funnel import funnel
from core.application.chat.intent_funnel import observability as obs_mod
from core.application.chat.intent_funnel.contract import (
    REASON_NO_CANDIDATE,
    REASON_REGISTRY_UNAVAILABLE,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry import entry as T
from core.application.chat.intent_funnel.registry.entry import (
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import (
    Complexity,
    Confidence,
    Signal,
    TurnRequirements,
)
from core.config import settings
from core.infrastructure.request_context import get_request_execution_mode


def _entry(cid, *, tool="create_folder", standard=""):
    q = T.QueryRecord(id="q1", query=standard, language=derive_language(standard))
    return T.CapabilityEntry(
        capability_id=cid, tool_binding=tool, description="d",
        standard_queries=(q,), request_query_examples=("做个事",),
    )


def _view(entries):
    return RegistryLiveView(
        fingerprint=content_fingerprint(list(entries)), entries=tuple(entries)
    )


def _req():
    return TurnRequirements(complexity=Complexity.LOW, confidence=Confidence.LOW,
                            needs_web=Signal.LOW, needs_memory=False)


def _ctx(msg="新建文件夹", **over):
    ns = types.SimpleNamespace(
        body=types.SimpleNamespace(message=msg, attach=None, viewer=None),
        owned_asset_id=None, research_turn=False, effective_handoff=None,
        session_id="s-1", user_id=1, history=[], agent_context=None,
        funnel_shadow_task=None, funnel_turn_key="",
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _orch(deps=None):
    return tor.TurnOrchestrator(deps if deps is not None
                                else types.SimpleNamespace(session_factory=None))


@pytest.fixture()
def _no_fast_paths(monkeypatch):
    """The A/B ruling: shadow-live is independent of every execution gate."""
    monkeypatch.setattr(settings, "chat_fast_paths_enabled", False)
    monkeypatch.setattr(settings, "chat_funnel_enabled", False)
    monkeypatch.setattr(settings, "chat_matcher_mode", "off")


# ── resolve_plan hook ───────────────────────────────────────────────────────────

async def test_flag_off_is_byte_identical_dark(monkeypatch, caplog):
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", False)
    ctx = _ctx()
    plan = await _orch().resolve_plan(ctx)
    assert ctx.funnel_shadow_task is None and ctx.funnel_turn_key == ""
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("funnel_ab_" in m for m in msgs)
    assert plan.kind.value == "agent"


async def test_flag_on_spawns_observation_but_agent_plan_unchanged(
    monkeypatch, caplog, _no_fast_paths
):
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", True)
    view = _view([_entry("cap-a", standard="新建文件夹")])

    async def fake_active(**kw):
        return view

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active
    )
    monkeypatch.setattr(
        "core.application.chat.intent_funnel.recall.load_index",
        lambda *a, **k: None,  # MISS lane only; HIT never touches it
    )
    ctx = _ctx("新建文件夹")
    plan = await _orch().resolve_plan(ctx)
    assert plan.kind.value == "agent"  # the shadow has zero routing authority
    assert ctx.funnel_turn_key and ctx.funnel_shadow_task is not None
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        await ctx.funnel_shadow_task
    line = next(r.getMessage() for r in caplog.records
                if "funnel_ab_shadow" in r.getMessage())
    assert f"turn_key={ctx.funnel_turn_key}" in line
    assert "matcher=HIT:cap-a" in line and "final_route=" in line
    # 8.14: the pin is released when the observation ends
    assert get_request_execution_mode() == "production"


async def test_research_and_handoff_turns_are_excluded(monkeypatch, _no_fast_paths):
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", True)
    for over in ({"research_turn": True}, {"effective_handoff": {"kind": "research"}}):
        ctx = _ctx(**over)
        await _orch().resolve_plan(ctx)
        assert ctx.funnel_shadow_task is None and ctx.funnel_turn_key == ""


async def test_vetoed_turn_logs_ineligible_without_the_cascade(
    monkeypatch, caplog, _no_fast_paths
):
    """Non-pure text (attachment/control payload) is vetoed by the shared
    guardrail BEFORE the cascade — the AB pair still gets a shadow line so the
    denominator stays auditable."""
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", True)

    async def boom(**kw):  # must never run: the veto short-circuits
        raise AssertionError("cascade must not run on a vetoed turn")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", boom
    )
    ctx = _ctx("x" * 64)  # is_pure_user_text guard, see sanitization rules
    ctx.body.message = "[Attached: doc.pdf] Call the `read_document` tool…"
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        plan = await _orch().resolve_plan(ctx)
        await ctx.funnel_shadow_task
    assert plan.kind.value == "agent"
    line = next(r.getMessage() for r in caplog.records
                if "funnel_ab_shadow" in r.getMessage())
    assert "status" not in line  # a clean veto line, not an error line
    assert "fallback_reason=input_not_pure_text" in line


async def test_shadow_fault_is_fail_quiet(monkeypatch, caplog, _no_fast_paths):
    """ANY exception inside the observation task must surface as one info line
    and never touch the turn (the fail-safe promise of shadow-live)."""
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", True)

    async def boom(*a, **k):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(funnel, "cascade_shadow", boom)
    ctx = _ctx()
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        plan = await _orch().resolve_plan(ctx)
        await ctx.funnel_shadow_task  # must not raise
    assert plan.kind.value == "agent"
    assert any("status=error" in r.getMessage() and "provider exploded" in r.getMessage()
               for r in caplog.records)


# ── shadow requirements (A/B fix 2026-09-27, item 4/3): the web/memory entry
# vetoes must be OBSERVABLE on a dark launch. Before the fix the shadow saw the
# neutral default requirements, so these turns were all mis-attributed to
# NO_CANDIDATE at the recall stage. ────────────────────────────────────────────

async def _shadow_line_for(monkeypatch, caplog, msg):
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", True)
    ctx = _ctx(msg)
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        plan = await _orch().resolve_plan(ctx)
        await ctx.funnel_shadow_task
    line = next(r.getMessage() for r in caplog.records
                if "funnel_ab_shadow" in r.getMessage())
    return plan, ctx, line


async def test_mid_sentence_cjk_web_turn_is_entry_vetoed_in_shadow(
    monkeypatch, caplog, _no_fast_paths
):
    """Item 4 (sim #14): "重要新闻" sits mid-sentence — the old "\\b(…|新闻)\\b"
    never matched CJK, and the neutral shadow requirements hid the veto twice
    over. Both legs fixed: the sentence classifies needs_web=HIGH and the
    shadow line reports the DESIGN exit, not a fake NO_CANDIDATE."""
    plan, ctx, line = await _shadow_line_for(
        monkeypatch, caplog, "2026年9月AI行业有什么重要新闻?")
    assert "deepest_stage=entry" in line
    assert "fallback_reason=turn_demands_web_or_memory" in line
    # ⑦ the Agent is completely unaffected: its own turn stays byte-identical
    assert plan.kind.value == "agent" and ctx.body.message == "2026年9月AI行业有什么重要新闻?"


async def test_anaphoric_summary_turn_is_contextual_fallback_not_a_takeover(
    monkeypatch, caplog, _no_fast_paths
):
    """Item 3 (sim #9): "把上面的内容总结一下" points at the CONVERSATION — the
    sentence carries no workable target, so the memory-deixis veto must stop
    the funnel BEFORE any cascade (never a summary capability match)."""
    plan, ctx, line = await _shadow_line_for(
        monkeypatch, caplog, "把上面的内容总结一下")
    assert "deepest_stage=entry" in line
    assert "fallback_reason=turn_demands_web_or_memory" in line
    assert plan.kind.value == "agent"


async def test_deictic_input_suspects_are_entry_vetoed_in_shadow(
    monkeypatch, caplog, _no_fast_paths
):
    """Shadow-A/B follow-up (suspects a403c4b341e1 / d795e47fe617): the two
    empty-schema standard-query sentences named their input object nowhere but
    an absent screen — the shadow must report the DESIGN exit at entry with the
    dedicated reason, and the Agent turn stays untouched."""
    for msg in ("把这份笔记做成思维导图", "把这个术语加入我的词汇库"):
        async def boom(**kw):  # the veto must short-circuit before the cascade
            raise AssertionError("cascade must not run on an input-vetoed turn")

        monkeypatch.setattr(
            "core.application.chat.intent_funnel.registry.active_view", boom
        )
        plan, ctx, line = await _shadow_line_for(monkeypatch, caplog, msg)
        assert "deepest_stage=entry" in line
        assert "fallback_reason=referenced_input_absent" in line
        assert "would_route=f" in line
        assert plan.kind.value == "agent"


async def test_plain_qa_turn_still_reaches_the_recall_lane(
    monkeypatch, caplog, _no_fast_paths
):
    """The veto guard must stay narrow: no web/memory cue -> the shadow runs
    the real cascade (registry read -> here: UNAVAILABLE view is enough to
    prove the cascade, not the entry branch)."""
    async def none_view(**kw):
        return None

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", none_view
    )
    _plan, _ctx, line = await _shadow_line_for(
        monkeypatch, caplog, "什么是思维导图?简单说说")
    assert "deepest_stage=registry" in line
    assert "fallback_reason=REGISTRY_UNAVAILABLE" in line


async def test_shadow_yields_when_production_cascade_owns_the_turn(
    monkeypatch, caplog
):
    """A/B hygiene fix: with the real gates LIVE the cascade already traces
    the turn — no shadow task, no AB pair, one funnel decision per turn."""
    monkeypatch.setattr(settings, "chat_funnel_shadow_live", True)
    monkeypatch.setattr(settings, "chat_fast_paths_enabled", True)
    monkeypatch.setattr(settings, "chat_funnel_enabled", True)
    monkeypatch.setattr(settings, "chat_action_fast_path_enabled", True)
    ctx = _ctx("新建文件夹")
    plan = await _orch().resolve_plan(ctx)
    assert plan.kind.value in ("agent", "action")
    assert ctx.funnel_shadow_task is None and ctx.funnel_turn_key == ""


# ── the funnel_ab_turn line: Agent-side outcome derivation ──────────────────────

async def test_ab_turn_line_counts_this_turns_calls_only(monkeypatch, caplog):
    monkeypatch.setattr(logging, "getLogger", logging.getLogger)  # no-op guard
    ctx = _ctx()
    ctx.funnel_turn_key = "k1"
    ctx.history = [{"role": "user", "content": "old"},
                   {"role": "assistant", "content": "old answer"}]
    messages = ctx.history + [
        {"role": "user", "content": "新建文件夹"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t", "function": {"name": "create_folder",
                                                 "arguments": "{}"}}]},
        {"role": "tool", "content": "ok", "tool_call_id": "t"},
        {"role": "assistant", "content": "已创建"},
    ]
    plan = types.SimpleNamespace(kind=types.SimpleNamespace(value="agent"))
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        tor._ab_turn_line(ctx, plan, agent_ms=1234.0, status="ok",
                          messages=messages, usage={"total_tokens": 900})
    line = next(r.getMessage() for r in caplog.records
                if "funnel_ab_turn" in r.getMessage())
    assert "turn_key=k1" in line and "plan_kind=agent" in line
    assert "llm_calls=2" in line          # history assistant rows are NOT counted
    assert "agent_tools=create_folder" in line
    assert "total_tokens=900" in line


async def test_ab_turn_line_reads_the_kernel_tool_call_shape(caplog):
    """The real loop records {id, name, arguments} at the TOP level (not the
    OpenAI {function:{name}} nesting) — the classifier's true_hit evidence
    depends on agent_tools resolving names, never "?"."""
    ctx = _ctx()
    ctx.funnel_turn_key = "k2"
    ctx.history = []
    messages = [
        {"role": "user", "content": "帮我总结一下这份文档"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t", "name": "read_document",
                         "arguments": "{}"}]},
        {"role": "tool", "content": "ok", "tool_call_id": "t"},
    ]
    plan = types.SimpleNamespace(kind=types.SimpleNamespace(value="agent"))
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        tor._ab_turn_line(ctx, plan, agent_ms=1.0, status="ok",
                          messages=messages, usage=None)
    line = next(r.getMessage() for r in caplog.records
                if "funnel_ab_turn" in r.getMessage())
    assert "agent_tools=read_document" in line


async def test_ab_turn_line_silent_without_turn_key(caplog):
    ctx = _ctx()
    plan = types.SimpleNamespace(kind=types.SimpleNamespace(value="agent"))
    with caplog.at_level(logging.INFO, logger="core.application.chat.turn_orchestrator"):
        tor._ab_turn_line(ctx, plan, agent_ms=1.0, status="ok",
                          messages=[], usage=None)
    assert not any("funnel_ab_turn" in r.getMessage() for r in caplog.records)


# ── cascade_shadow persistence seam ─────────────────────────────────────────────

async def test_cascade_shadow_default_still_writes_no_event_row(monkeypatch):
    """Existing offline replays (synthetic_workload) must stay log-only."""
    calls = []
    monkeypatch.setattr(obs_mod, "persist_event",
                        lambda *a, **k: calls.append(1))
    ctx = _ctx("hello")

    async def none_view(**kw):
        return None

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", none_view
    )
    res = await funnel.cascade_shadow(ctx, deps=types.SimpleNamespace(
        session_factory=object()))
    assert res["execution_mode"] == "shadow"
    assert res["fallback_reason"] == REASON_REGISTRY_UNAVAILABLE
    assert calls == []


async def test_cascade_shadow_persists_with_turn_key_inside_the_pin(monkeypatch):
    seen = {}

    async def spy_persist(deps, ctx, trace, trace_json=None):
        seen["mode"] = get_request_execution_mode()
        seen["trace_json"] = trace_json
        seen["fallback"] = trace["fallback"]

    monkeypatch.setattr(obs_mod, "persist_event", spy_persist)

    async def none_view(**kw):
        return None

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", none_view
    )
    ctx = _ctx("hello")
    await funnel.cascade_shadow(
        ctx, deps=types.SimpleNamespace(session_factory=object()),
        persist_event=True, turn_key="k9",
    )
    assert seen["mode"] == "shadow"          # written INSIDE the 8.14 pin
    assert seen["trace_json"]["turn_key"] == "k9"
    assert "query" not in seen["trace_json"]  # the no-query rule holds
    assert seen["fallback"] == REASON_REGISTRY_UNAVAILABLE

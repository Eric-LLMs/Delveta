"""Chat-plane exposure hiding for cap-artifact + cap-bash (ruling).

One goal, two independent surfaces — hidden from CHAT, kept on the AGENT plane:

1. INTENT FUNNEL gate — ``settings.chat_funnel_hidden_capabilities`` ships
   ``cap-edit-file,cap-artifact,cap-bash``. The capability-admission predicate
   ``is_intent_funnel_capability_allowed`` drops them from the Matcher exact
   index AND the Intent Funnel's ``entries_by_id`` (so no card is built and
   certification is refused).
2. RECALL / CARD surface — the Recall corpus is loaded straight from the LIVE
   query tables, which carry NO admission predicate, so a hidden capability's
   sentences can still surface from ``recall.recall``. Those hits KEEP their
   cards (``funnel_cardable``), matching the option distribution the selector was
   trained on; capability admission is enforced on the DECISION instead — a pick
   outside ``is_intent_funnel_capability_allowed`` becomes
   ``INTENT_FUNNEL_ALLOWLIST_REJECT`` before argument extraction and the turn
   falls open to the Agent. A hidden capability therefore never certifies and
   never executes through the Intent Funnel, but it is no longer invisible to the
   selector — and the Agent keeps its own independent tool surface.

The AGENT plane is untouched: ``agent_hidden_tools`` does NOT name artifact/bash,
so the chat-process kernel still registers ``bash`` with its destructive marker and
WRITE/NETWORK permissions, and the destructive deny/guard + approval chain is
unchanged. The Registry row stays enabled+active; no tool / Runtime / Binder edit.
"""
from __future__ import annotations

import logging
import re
import types

import pytest
from agent.engine.runtime import ToolRuntime
from agent.tools.fs_tools import register_fs_tools
from agent.tools.tool_gateway import ToolCatalog
from agent.tools.tool_permissions import ToolPermission
from core.application.chat.intent_funnel import candidate_aggregation as agg_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import recall as recall_mod
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    MATCH_MISS,
    REASON_ACQUISITION_MODEL_PENDING,
    REASON_INTENT_FUNNEL_ALLOWLIST_REJECT,
    REASON_VERSION_MISMATCH,
    Candidate,
    MatchResult,
    RecallResult,
)
from core.application.chat.intent_funnel.matcher.index import build_index
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
    is_intent_funnel_capability_allowed,
)
from core.application.chat.understanding import TurnRequirements
from core.config import Settings, settings

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"
_HIDDEN = "cap-edit-file,cap-artifact,cap-bash"


class _NoSandbox:  # bash_tool only stores it; no execution in this file
    pass


# ── 1. the Intent Funnel predicate + shipped default ────────────────────────────


def _entry(cid: str, tool: str, *, query: str = "做一些事情") -> CapabilityEntry:
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description=f"does {cid}",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=query,
                                      language=derive_language(query)),),
        parameters={"name": {"type": "string", "required": True,
                             "max_len": 120, "description": "arg"}},
        enabled=True, status="active",
    )


def test_shipped_default_names_all_three_hidden_capabilities():
    # the shipped default (no env override) hides edit-file, artifact and bash.
    assert set(settings.chat_funnel_hidden_capabilities.split(",")) == {
        "cap-edit-file", "cap-artifact", "cap-bash"}


def test_recall_threshold_default_is_082():
    # The shipped default must be the value the four-arm experiment chose.
    # Read the field default, not ``settings``: a local .env / env var must not
    # be able to mask a wrong default.
    assert Settings.model_fields["chat_funnel_min_score"].default == 0.82


def test_recall_threshold_env_override(monkeypatch):
    """CHAT_FUNNEL_MIN_SCORE overrides the default (case-insensitive field env
    name); with the override removed the shipped default applies again."""
    monkeypatch.setenv("CHAT_FUNNEL_MIN_SCORE", "0.88")
    assert Settings(_env_file=None).chat_funnel_min_score == 0.88
    monkeypatch.delenv("CHAT_FUNNEL_MIN_SCORE", raising=False)
    assert Settings(_env_file=None).chat_funnel_min_score == 0.82


def test_hidden_capabilities_are_not_admitted_by_the_intent_funnel(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)
    for cid, tool in (("cap-artifact", "artifact"), ("cap-bash", "bash"),
                      ("cap-edit-file", "edit_file")):
        e = _entry(cid, tool)
        assert e.enabled and e.status == "active"     # Registry row untouched
        assert not is_intent_funnel_capability_allowed(e)  # …but invisible to the Intent Funnel
    assert is_intent_funnel_capability_allowed(_entry("cap-read-file", "read_file"))


def test_hidden_capabilities_absent_from_matcher_exact_index(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)
    view = RegistryLiveView(fingerprint="fp-hidden", entries=(
        _entry("cap-artifact", "artifact", query="编译项目"),
        _entry("cap-bash", "bash", query="跑一条命令"),
        _entry("cap-read-file", "read_file", query="读一下这个文件"),
    ))
    indexed = {cid for ids in build_index(view).values() for cid in ids}
    assert indexed == {"cap-read-file"}               # no artifact/bash in the index


# ── 2. the RECALL closure (the real gap): single-point orchestrator filter ───────


def _ctx(message="随便说点什么"):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
        user_id="u", viewer_assembly=None, agent_context=None, log_tokens=None,
        # Fields the real AgentExecutor reads when it forwards the turn.
        user_text=message, session_memory=None, model=None, base_url=None, api_key=None,
        disable_thinking=False,
    )


def _deps():
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object())


def _fallback(caplog) -> str:
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    return re.search(r"fallback_reason=(\S+)", lines[0]).group(1)


class _Rec:
    def __init__(self) -> None:
        self.aggregated_ids: list[set[str]] = []


def _wire(monkeypatch, *, recall_candidates, entries, hidden):
    view = RegistryLiveView(fingerprint=content_fingerprint(list(entries)),
                            entries=tuple(entries))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_MISS)

    async def fake_load(sf):
        return types.SimpleNamespace(version="corpus-test")

    async def fake_recall(index, message, *, embedder, min_score):
        return RecallResult(candidates=tuple(recall_candidates))

    real_agg = agg_mod.aggregate_by_capability

    def spy_agg(candidates):
        rec.aggregated_ids.append({c.capability_id for c in candidates})
        return real_agg(candidates)

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(recall_mod, "load_index", fake_load)
    monkeypatch.setattr(recall_mod, "recall", fake_recall)
    monkeypatch.setattr(agg_mod, "aggregate_by_capability", spy_agg)
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", hidden, raising=False)
    # The deterministic test double stands in for the ONE deployed selector.
    from core.application.chat.intent_funnel.cap_router.backends.stub import (
        StubSelector,
    )
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: StubSelector())
    monkeypatch.setattr(settings, "chat_funnel_min_score", 0.60)
    monkeypatch.setattr(settings, "chat_funnel_margin", 0.06)
    return rec


async def _route(ctx, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=_deps(), requirements=req)


async def test_hidden_recall_hit_is_carded_and_its_pick_becomes_an_intent_funnel_allowlist_reject(
        monkeypatch, caplog):
    """A hidden capability recalled at the TOP score KEEPS its card (the selector
    must see the training distribution), but the pick is converted into an Intent
    Funnel REJECT before certification — never Intent-Funnel-executed, never
    re-routed to another card, and the turn falls open to the Agent."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entries = [_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")]
    rec = _wire(monkeypatch,
                recall_candidates=[Candidate("cap-bash", 0.95, origin="recall"),
                                   Candidate("cap-a", 0.70, origin="recall")],
                entries=entries, hidden=_HIDDEN)
    req = TurnRequirements()
    out = await _route(_ctx(), req)

    assert rec.aggregated_ids == [{"cap-bash", "cap-a"}]   # both are carded
    # The stub picks cap-bash (margin 0.25). It is on a card but not admitted
    # by ``is_intent_funnel_capability_allowed``, so the business layer turns
    # the pick into the REJECT exit. The ORIGINAL requirements come back
    # byte-identical, so the normal plan mapping sends the turn to the Agent.
    assert out is req and _fallback(caplog) == REASON_INTENT_FUNNEL_ALLOWLIST_REJECT
    # Same fallback rule as a selector NONE/REJECT: the untouched requirements
    # map to AGENT, never to a whole-request prohibition.
    from core.application.chat.execution_plan import PlanKind, build_execution_plan
    assert build_execution_plan(out).kind is PlanKind.AGENT
    # No capability was certified, and nobody re-picked the lower-scored card.
    assert out.requested_action is None


async def test_hidden_pick_full_path_resolves_agent_plan_and_hands_over_the_reason(
        monkeypatch, caplog):
    """The complete fallback path through the real control plane: a hidden pick is
    rejected (``INTENT_FUNNEL_ALLOWLIST_REJECT``), ``resolve_plan`` sees the
    ORIGINAL requirements, maps them to AGENT, and ``run_turn`` actually invokes
    AgentExecutor — never a refusal executor or a whole-request prohibition.

    The refusal is also CONTEXT, not only telemetry: the control plane sinks the
    reject into ``agent_context`` and the real AgentExecutor forwards that dict as
    ``deps.agent.run(context=…)`` — the exact channel the chat prompt section reads.
    """
    from core.application.chat import turn_orchestrator as turn_mod
    from core.application.chat.execution_plan import PlanKind
    from core.application.chat.executors.agent import AgentExecutor
    from core.application.chat.turn_orchestrator import TurnOrchestrator

    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    _wire(monkeypatch,
          recall_candidates=[Candidate("cap-bash", 0.95, origin="recall"),
                             Candidate("cap-a", 0.70, origin="recall")],
          entries=[_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")],
          hidden=_HIDDEN)

    calls = []

    class _Result:
        final_answer = "agent owns the turn"
        messages: list = []
        usage: dict = {}

    class _Agent:
        """The kernel seam: records the context the executor forwards."""

        async def run(self, message, history, *, session_memory=None, model=None,
                      base_url=None, api_key=None, context=None):
            calls.append({"message": message, "context": context})
            return _Result()

    async def fake_finalize(ctx, deps, payload, *, tool):
        return dict(payload)

    async def fake_post_turn(ctx, deps):
        return False

    monkeypatch.setattr(turn_mod, "finalize_turn", fake_finalize)
    monkeypatch.setattr(turn_mod, "handle_research_post_turn", fake_post_turn)

    deps = types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(), agent=_Agent())
    orch = TurnOrchestrator(deps)
    resp = await orch.run_turn(_ctx())

    assert resp["answer"] == "agent owns the turn"
    assert len(calls) == 1                       # AgentExecutor.run was dispatched
    # The Agent receives the ORIGINAL user text plus the funnel's routing fact.
    assert calls[0]["message"] == "随便说点什么"
    record = calls[0]["context"]["intent_funnel"]
    assert record["kind"] == "ADMISSION_REJECT"
    assert record["stage"] == "admission"
    assert record["reason"] == REASON_INTENT_FUNNEL_ALLOWLIST_REJECT
    assert record["capability_id"] == "cap-bash"
    assert record["tool_binding"] == "bash"
    # …and the plan really went through the AGENT branch.
    assert _fallback(caplog) == REASON_INTENT_FUNNEL_ALLOWLIST_REJECT
    plan = await orch.resolve_plan(_ctx())
    assert plan.kind is PlanKind.AGENT
    assert isinstance(orch.executor_for(plan), AgentExecutor)


async def test_admitted_pick_enters_acquisition_and_records_no_reject(
        monkeypatch, caplog):
    """Case B (control): a pick INSIDE the Intent Funnel allowlist passes the
    admission gate and enters argument acquisition normally; no reject record is
    written, so no routing note is armed for the Agent."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    _wire(monkeypatch,
          recall_candidates=[Candidate("cap-a", 0.95, origin="recall"),
                             Candidate("cap-bash", 0.60, origin="recall")],
          entries=[_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")],
          hidden=_HIDDEN)
    ctx = _ctx()
    req = TurnRequirements()

    out = await _route(ctx, req)

    # cap-a is admitted, so the turn proceeds PAST admission into acquisition
    # (which then abstains: this fixture declares no acquisition inputs). The
    # absence of the reject reason proves the gate did not fire.
    assert out is req
    assert _fallback(caplog) != REASON_INTENT_FUNNEL_ALLOWLIST_REJECT
    # The later acquisition abstain may publish its own record; what must NOT
    # appear is an admission-reject record for the admitted pick.
    record = getattr(ctx, "intent_funnel_failure", None) or {}
    assert record.get("kind") != "ADMISSION_REJECT"
    assert record.get("reason") != REASON_INTENT_FUNNEL_ALLOWLIST_REJECT


async def test_intent_funnel_allowlist_reject_never_enters_argument_acquisition_or_binder(
        monkeypatch, caplog):
    """Capability admission (``is_intent_funnel_capability_allowed``) is the SOLE
    rule: the extractor, the Argument Path Router and the Binder are never
    reached for a rejected pick."""
    from core.application.chat.intent_funnel import binder as binder_mod
    from core.application.chat.intent_funnel import orchestrator as orch_mod

    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    calls: list[str] = []

    async def boom_acquisition(*a, **kw):
        calls.append("acquisition")
        raise AssertionError("argument acquisition must not run")

    def boom_binder(*a, **kw):
        calls.append("binder")
        raise AssertionError("the Binder must not run")

    monkeypatch.setattr(orch_mod, "_acquisition_hop", boom_acquisition)
    monkeypatch.setattr(binder_mod, "validate", boom_binder)
    _wire(monkeypatch,
          recall_candidates=[Candidate("cap-bash", 0.95, origin="recall"),
                             Candidate("cap-a", 0.70, origin="recall")],
          entries=[_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")],
          hidden=_HIDDEN)
    out = await _route(_ctx(), TurnRequirements())
    assert calls == []
    assert out.requested_action is None
    assert _fallback(caplog) == REASON_INTENT_FUNNEL_ALLOWLIST_REJECT


async def test_hidden_recall_hits_reach_the_cap_router_and_can_be_rejected(
        monkeypatch, caplog):
    """Hidden hits are ordinary candidates now: they enter aggregation and the
    cap_router list, and a winning hidden pick is rejected — not silently
    replaced by a lower-scored card that IS admitted."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entries = [_entry("cap-bash", "bash"), _entry("cap-artifact", "artifact"),
               _entry("cap-a", "create_folder"), _entry("cap-b", "create_folder"),
               _entry("cap-c", "create_folder")]
    rec = _wire(monkeypatch,
                recall_candidates=[Candidate("cap-bash", 0.99, origin="recall"),
                                   Candidate("cap-artifact", 0.50, origin="recall"),
                                   Candidate("cap-a", 0.90, origin="recall"),
                                   Candidate("cap-b", 0.70, origin="recall"),
                                   Candidate("cap-c", 0.60, origin="recall")],
                entries=entries, hidden=_HIDDEN)
    req = TurnRequirements()
    out = await _route(_ctx(), req)

    assert rec.aggregated_ids == [
        {"cap-bash", "cap-artifact", "cap-a", "cap-b", "cap-c"}]
    assert out is req and _fallback(caplog) == REASON_INTENT_FUNNEL_ALLOWLIST_REJECT


async def test_candidate_outside_the_intent_funnel_allowlist_reaches_the_selector(
        monkeypatch, caplog):
    """The selector's INPUT is the card surface, not capability admission: a
    hidden capability is offered to the model, and only its pick is rejected."""
    from core.application.chat.intent_funnel.cap_router.backends.stub import (
        StubSelector,
    )
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entries = [_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")]
    seen: list[set[str]] = []

    class _SpySelector:
        async def select(self, query, candidates, *, entries_by_id, facts=None):
            seen.append({c.capability_id for c in candidates})
            return await StubSelector().select(
                query, candidates, entries_by_id=entries_by_id, facts=facts)

    _wire(monkeypatch,
          recall_candidates=[Candidate("cap-bash", 0.95, origin="recall"),
                             Candidate("cap-a", 0.70, origin="recall")],
          entries=entries, hidden=_HIDDEN)
    monkeypatch.setattr(cap_router_mod, "default_selector",
                        lambda: _SpySelector())

    req = TurnRequirements()
    out = await _route(_ctx(), req)

    assert seen == [{"cap-bash", "cap-a"}]   # the hidden capability is carded
    assert out is req                        # the Agent keeps the turn
    assert _fallback(caplog) == REASON_INTENT_FUNNEL_ALLOWLIST_REJECT


async def test_control_without_hiding_the_same_hit_is_aggregated(monkeypatch, caplog):
    """Sanity/control: with NOTHING hidden the very same cap-bash hit DOES enter
    the pool — proving the closure is the gate, not an empty recall."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entries = [_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")]
    rec = _wire(monkeypatch,
                recall_candidates=[Candidate("cap-bash", 0.95, origin="recall"),
                                   Candidate("cap-a", 0.70, origin="recall")],
                entries=entries, hidden="")
    await _route(_ctx(), TurnRequirements())
    assert rec.aggregated_ids == [{"cap-bash", "cap-a"}]   # both survive without hiding


async def test_chat_cannot_certify_a_hidden_capability(monkeypatch, caplog):
    """Even a Matcher HIT on a hidden capability is refused: the capability is not
    a chat-plane candidate, so ``entries_by_id`` has no entry and the orchestrator
    exits to the Agent instead of certifying."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    view = RegistryLiveView(fingerprint="fp", entries=(
        _entry("cap-bash", "bash"), _entry("cap-a", "create_folder")))

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):              # forged HIT on the hidden cap
        return MatchResult(state=MATCH_HIT, capability_id="cap-bash",
                           matched_literal="跑一条命令")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)

    req = TurnRequirements()
    out = await _route(_ctx(), req)
    assert out is req and _fallback(caplog) == REASON_VERSION_MISMATCH


# ── 3. the AGENT plane keeps artifact/bash, with the security chain intact ───────


def test_agent_hidden_tools_does_not_name_bash_or_artifact():
    names = {t.strip() for t in settings.agent_hidden_tools.split(",") if t.strip()}
    assert "bash" not in names
    assert "artifact" not in names


def test_agent_plane_still_registers_bash(tmp_path):
    rt = ToolRuntime()
    register_fs_tools(rt, tmp_path, sandbox=_NoSandbox(),
                      exclude={t.strip() for t in settings.agent_hidden_tools.split(",")
                               if t.strip()})
    assert rt.get("bash") is not None                   # visible to the LLM Agent
    assert "- bash:" in ToolCatalog(rt).render_index()  # in the prompt catalog


def test_bash_destructive_and_permission_are_unchanged(tmp_path):
    rt = ToolRuntime()
    register_fs_tools(rt, tmp_path, sandbox=_NoSandbox())
    bash = rt.get("bash")
    assert bash.destructive is True                     # destructive marker intact
    assert {ToolPermission.WRITE, ToolPermission.NETWORK} <= set(bash.permissions)


# ── 4. edit-file behavior does not regress ───────────────────────────────────────


def test_edit_file_still_hidden_and_neighbors_routable(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)
    assert not is_intent_funnel_capability_allowed(_entry("cap-edit-file", "edit_file"))
    assert is_intent_funnel_capability_allowed(_entry("cap-read-file", "read_file"))


@pytest.mark.parametrize("cid", ["cap-artifact", "cap-bash"])
def test_registry_row_is_left_enabled_and_active(cid, monkeypatch):
    """The Registry row is NOT disabled by the exposure change (page/PC/worker/
    admin keep it): only the chat-plane view drops it."""
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)
    e = _entry(cid, "artifact" if cid == "cap-artifact" else "bash")
    assert e.enabled is True and e.status == "active"
    assert not is_intent_funnel_capability_allowed(e)

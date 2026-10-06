"""Chat-plane exposure hiding for cap-artifact + cap-bash (ruling).

One goal, two independent surfaces — hidden from CHAT, kept on the AGENT plane:

1. FUNNEL gate — ``settings.chat_funnel_hidden_capabilities`` ships
   ``cap-edit-file,cap-artifact,cap-bash``. The predicate ``chat_plane_candidate``
   drops them from the Matcher exact index AND the funnel ``entries_by_id`` (so no
   card is built and certification is refused).
2. RECALL closure — the Recall corpus is loaded straight from the LIVE query
   tables, which carry NO exposure predicate, so a hidden capability's sentences
   can still surface from ``recall.recall``. The orchestrator drops any recall hit
   whose capability is not a current chat-plane candidate, so a hidden capability
   never enters the candidate pool, aggregation, the cap_router list, or
   certification. No Recall SQL change — this is the single closure point.

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
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import recall as recall_mod
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    MATCH_MISS,
    REASON_ACQUISITION_MODEL_PENDING,
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
    chat_plane_candidate,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"
_HIDDEN = "cap-edit-file,cap-artifact,cap-bash"


class _NoSandbox:  # bash_tool only stores it; no execution in this file
    pass


# ── 1. the FUNNEL predicate + shipped default ────────────────────────────────────


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


def test_hidden_capabilities_are_not_chat_plane_candidates(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)
    for cid, tool in (("cap-artifact", "artifact"), ("cap-bash", "bash"),
                      ("cap-edit-file", "edit_file")):
        e = _entry(cid, tool)
        assert e.enabled and e.status == "active"     # Registry row untouched
        assert not chat_plane_candidate(e)            # …but invisible to the funnel
    assert chat_plane_candidate(_entry("cap-read-file", "read_file"))


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
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")
    monkeypatch.setattr(settings, "chat_funnel_min_score", 0.60)
    monkeypatch.setattr(settings, "chat_funnel_margin", 0.06)
    return rec


async def _route(ctx, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=_deps(), requirements=req)


async def test_hidden_recall_hit_never_enters_the_candidate_pool(monkeypatch, caplog):
    """A hidden capability recalled at the TOP score is dropped before aggregation;
    only the legit hit remains, and the funnel cannot select the hidden one."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entries = [_entry("cap-bash", "bash"), _entry("cap-a", "create_folder")]
    rec = _wire(monkeypatch,
                recall_candidates=[Candidate("cap-bash", 0.95, origin="recall"),
                                   Candidate("cap-a", 0.70, origin="recall")],
                entries=entries, hidden=_HIDDEN)
    req = TurnRequirements()
    out = await _route(_ctx(), req)

    assert rec.aggregated_ids == [{"cap-a"}]          # cap-bash dropped pre-aggregation
    # cap-a (K=1, business-direct) → ARP needs its MODEL slot → Agent, NOT a
    # VERSION_MISMATCH from selecting the hidden cap-bash.
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_MODEL_PENDING


async def test_hidden_recall_hits_absent_from_the_cap_router_list(monkeypatch, caplog):
    """With hidden hits mixed among 3 legit ones, aggregation/router see exactly
    the 3 legit candidates — the hidden ones never reach the selector."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entries = [_entry("cap-bash", "bash"), _entry("cap-artifact", "artifact"),
               _entry("cap-a", "create_folder"), _entry("cap-b", "create_folder"),
               _entry("cap-c", "create_folder")]
    rec = _wire(monkeypatch,
                recall_candidates=[Candidate("cap-bash", 0.99, origin="recall"),
                                   Candidate("cap-artifact", 0.98, origin="recall"),
                                   Candidate("cap-a", 0.90, origin="recall"),
                                   Candidate("cap-b", 0.70, origin="recall"),
                                   Candidate("cap-c", 0.60, origin="recall")],
                entries=entries, hidden=_HIDDEN)
    req = TurnRequirements()
    out = await _route(_ctx(), req)

    assert rec.aggregated_ids == [{"cap-a", "cap-b", "cap-c"}]
    # cap-a selected (clear margin) → MODEL slot pending; never the hidden caps.
    assert out is req and _fallback(caplog) == REASON_ACQUISITION_MODEL_PENDING


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
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")

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
    assert not chat_plane_candidate(_entry("cap-edit-file", "edit_file"))
    assert chat_plane_candidate(_entry("cap-read-file", "read_file"))


@pytest.mark.parametrize("cid", ["cap-artifact", "cap-bash"])
def test_registry_row_is_left_enabled_and_active(cid, monkeypatch):
    """The Registry row is NOT disabled by the exposure change (page/PC/worker/
    admin keep it): only the chat-plane view drops it."""
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", _HIDDEN, raising=False)
    e = _entry(cid, "artifact" if cid == "cap-artifact" else "bash")
    assert e.enabled is True and e.status == "active"
    assert not chat_plane_candidate(e)

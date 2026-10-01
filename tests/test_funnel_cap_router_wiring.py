"""Phase 3 (2026-10-01) — cap_router selection-lane wiring (transition seam).

Pins the LOCKED Phase 3 business rules:

* ``backend=off``  -> the FULL legacy chain, byte-identical (HIT and MISS/AMB);
* ``backend=stub`` -> the NEW lane: MISS/AMBIGUOUS selects via cap_router (stub);
* a MATCH_HIT on the NEW lane is NEVER re-selected — neither via cap_router nor
  via the legacy ``select_and_extract`` — because the Matcher already pinned the
  capability. With no Argument Path Router yet it safely falls back to the Agent
  with ``CAP_ROUTER_HIT_DEFERRED`` (a Phase 3 compatibility limitation);
* NONE -> ``CAP_ROUTER_NONE``; a selector that cannot serve ->
  ``CAP_ROUTER_UNAVAILABLE``; both exit to the Agent, never to a Qwen fallback;
* the Phase 3 stub extracts NO arguments, so a schema'd capability exits
  ``BIND_MISSING`` — the deliberate limit of the phase, NOT a cap_router failure;
* the CapabilityRoute -> ToolIntentVerdict bridge is a marked TRANSITION SHIM.

Aggregation is NOT modified; the spies below only observe it.
"""
from __future__ import annotations

import logging
import re
import types

from core.application.chat.intent_funnel import candidate_aggregation as agg_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel import funnel
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import recall as recall_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.contract import (
    MATCH_AMBIGUOUS,
    MATCH_HIT,
    MATCH_MISS,
    REASON_BIND_MISSING,
    REASON_CAP_ROUTER_HIT_DEFERRED,
    REASON_CAP_ROUTER_NONE,
    REASON_CAP_ROUTER_UNAVAILABLE,
    TOOL_INTENT_CONFIDENT,
    Candidate,
    MatchResult,
    RecallResult,
    ToolIntentVerdict,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"
MSG = "create a folder named zeta"


def _entry(cid):
    q = "create a folder"
    return CapabilityEntry(
        capability_id=cid, tool_binding="create_folder", description=f"does {cid}",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=q,
                                      language=derive_language(q)),),
        parameters={"name": {"type": "string", "required": True,
                             "max_len": 120, "description": "folder name"}},
    )


def _ctx(message=MSG):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, research_turn=False, effective_handoff=None,
        session_id="",
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
    """Spy recorder for every node the wired cascade could touch."""

    def __init__(self):
        self.legacy = []      # legacy select_and_extract calls
        self.factory = []     # cap_router.selector_for(backend) calls
        self.recall = []      # recall.load_index calls
        self.aggregate = []   # candidate_aggregation.aggregate_by_capability calls


_ONE = [Candidate("cap-a", 0.9, origin="recall", matched_example="create a folder")]


def _wire(monkeypatch, *, state, recall_candidates=_ONE) -> _Rec:
    entries = (_entry("cap-a"), _entry("cap-b"))
    view = RegistryLiveView(
        fingerprint=content_fingerprint(list(entries)), entries=entries)
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        # SYNC on purpose: production matcher.match() is a synchronous call
        # (orchestrator.py: `mres = matcher.match(...)`); an async fake would
        # return a coroutine and crash on `mres.state`.
        if state == MATCH_HIT:
            return MatchResult(state=MATCH_HIT, capability_id="cap-a",
                               matched_literal="create a folder")
        if state == MATCH_AMBIGUOUS:
            return MatchResult(state=MATCH_AMBIGUOUS, candidates=("cap-a", "cap-b"))
        return MatchResult(state=MATCH_MISS)

    async def fake_load(sf):
        rec.recall.append(1)
        return types.SimpleNamespace(version="corpus-test")

    async def fake_recall(index, message, *, embedder, min_score):
        return RecallResult(candidates=tuple(recall_candidates))

    real_agg = agg_mod.aggregate_by_capability

    def spy_agg(candidates):
        rec.aggregate.append(1)
        return real_agg(candidates)

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(recall_mod, "load_index", fake_load)
    monkeypatch.setattr(recall_mod, "recall", fake_recall)
    monkeypatch.setattr(agg_mod, "aggregate_by_capability", spy_agg)
    monkeypatch.setattr(settings, "chat_funnel_min_score", 0.82)
    monkeypatch.setattr(settings, "chat_funnel_margin", 0.06)

    async def fake_select_and_extract(query, candidates, *, entries_by_id,
                                      llm=None, facts=None):
        rec.legacy.append((query, len(candidates)))
        return ToolIntentVerdict(TOOL_INTENT_CONFIDENT, "cap-a", arguments=None)

    monkeypatch.setattr(tool_intent_mod, "select_and_extract", fake_select_and_extract)

    real_factory = cap_router_mod.selector_for

    def spy_factory(backend):
        rec.factory.append(backend)
        return real_factory(backend)

    monkeypatch.setattr(cap_router_mod, "selector_for", spy_factory)
    return rec


async def _run(monkeypatch, caplog, *, state, backend, recall_candidates=_ONE):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    rec = _wire(monkeypatch, state=state, recall_candidates=recall_candidates)
    monkeypatch.setattr(settings, "chat_cap_router_backend", backend)
    req = TurnRequirements()
    out = await funnel.route(_ctx(), deps=_deps(), requirements=req)
    return out, req, rec, _fallback(caplog)


# ── backend=off: the FULL legacy chain is untouched ──────────────────────────────


async def test_off_miss_uses_legacy_hop(monkeypatch, caplog):
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS, backend="off")
    assert rec.legacy and not rec.factory
    assert out is req and fb == REASON_BIND_MISSING


async def test_off_hit_uses_legacy_hop(monkeypatch, caplog):
    # off keeps the OLD HIT path (legacy select_and_extract) — the rollback lane.
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_HIT, backend="off")
    assert rec.legacy and not rec.factory
    assert not rec.recall            # HIT never recalls, even in the old chain
    assert out is req


# ── backend=stub: MISS/AMBIGUOUS -> Recall -> Aggregation -> cap_router ───────────


async def test_stub_miss_routes_through_cap_router(monkeypatch, caplog):
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS, backend="stub")
    assert rec.factory == ["stub"]   # cap_router consulted
    assert rec.recall and rec.aggregate
    assert not rec.legacy            # the legacy hop is NOT used on the new lane
    # stub extracts no args -> schema'd cap exits BIND_MISSING (phase limit)
    assert out is req and fb == REASON_BIND_MISSING


async def test_stub_ambiguous_routes_through_cap_router(monkeypatch, caplog):
    out, req, rec, fb = await _run(
        monkeypatch, caplog, state=MATCH_AMBIGUOUS, backend="stub")
    assert rec.factory == ["stub"] and rec.recall and rec.aggregate
    assert not rec.legacy
    # ambiguous seeds are untrusted -> the stub answers NONE
    assert out is req and fb in (REASON_CAP_ROUTER_NONE, REASON_BIND_MISSING)


# ── backend=stub + HIT: capability decided, no re-selection, safe fallback ────────


async def test_stub_hit_is_deferred_without_reselection(monkeypatch, caplog):
    # The Matcher pinned the capability; Phase 3 must NOT re-select it (neither
    # cap_router nor legacy select_and_extract) and must short-circuit BEFORE
    # Recall / Aggregation.
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_HIT, backend="stub")
    assert not rec.factory           # no cap_router selection
    assert not rec.legacy            # no legacy capability selection
    assert not rec.recall            # no Recall
    assert not rec.aggregate         # no Candidate Aggregation
    assert out is req and fb == REASON_CAP_ROUTER_HIT_DEFERRED


# ── NONE / unavailable fallbacks ─────────────────────────────────────────────────


async def test_stub_none_exits_cap_router_none(monkeypatch, caplog):
    # a too-close two-candidate race (both trusted): the stub never resolves it
    close = [Candidate("cap-a", 0.90, origin="recall"),
             Candidate("cap-b", 0.89, origin="recall")]
    out, req, rec, fb = await _run(
        monkeypatch, caplog, state=MATCH_MISS, backend="stub",
        recall_candidates=close)
    assert rec.factory == ["stub"] and not rec.legacy
    assert out is req and fb == REASON_CAP_ROUTER_NONE


async def test_undeployed_backend_exits_unavailable(monkeypatch, caplog):
    # "laya" is a valid enum value but has no selector yet -> never falls through
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS, backend="laya")
    assert rec.factory == ["laya"] and not rec.legacy
    assert out is req and fb == REASON_CAP_ROUTER_UNAVAILABLE


async def test_selector_that_raises_exits_unavailable(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    rec = _wire(monkeypatch, state=MATCH_MISS)

    class _Boom:
        async def select(self, query, candidates, *, entries_by_id, facts=None):
            raise cap_router_mod.CapabilityRouterUnavailable("transport down")

    monkeypatch.setattr(cap_router_mod, "selector_for", lambda backend: _Boom())
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")
    req = TurnRequirements()
    out = await funnel.route(_ctx(), deps=_deps(), requirements=req)
    assert out is req and _fallback(caplog) == REASON_CAP_ROUTER_UNAVAILABLE
    assert not rec.legacy


# ── marked transition seams (code + shim) ────────────────────────────────────────


def test_hit_compatibility_limitation_is_documented():
    import core.application.chat.intent_funnel.orchestrator as orch

    with open(orch.__file__, encoding="utf-8") as fh:
        src = (orch.__doc__ or "") + fh.read()
    assert "PHASE 3 COMPATIBILITY LIMITATION" in src
    assert "Phase 4" in src


def test_transition_shim_is_marked_for_removal_and_used():
    import core.application.chat.intent_funnel.orchestrator as orch
    import core.application.chat.intent_funnel.selection_transition as shim

    with open(shim.__file__, encoding="utf-8") as fh:
        shim_src = (shim.__doc__ or "") + fh.read()
    assert "TRANSITION SHIM" in shim_src and "REMOVE IN PHASE 4" in shim_src
    with open(orch.__file__, encoding="utf-8") as fh:
        assert "selection_transition" in fh.read()   # actually wired, not dead

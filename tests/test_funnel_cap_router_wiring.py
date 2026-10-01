"""Phase 3/4 (2026-10-01) — cap_router selection + Argument Path Router wiring.

Pins the LOCKED business rules:

* ``backend=off``  -> the FULL legacy chain, byte-identical (HIT and MISS/AMB);
* ``backend=stub`` -> the NEW lane: MISS/AMBIGUOUS selects via cap_router (stub),
  then the decided capability feeds the Argument Path Router;
* a MATCH_HIT on the NEW lane goes STRAIGHT to the Argument Path Router — it is
  NEVER re-selected (no Recall, no Aggregation, no cap_router, no legacy
  ``select_and_extract``), because the Matcher already pinned the capability;
* NONE -> ``CAP_ROUTER_NONE``; a selector that cannot serve ->
  ``CAP_ROUTER_UNAVAILABLE``; both exit to the Agent, never to a Qwen fallback;
* the phase-3 transition shim is GONE: ``selection_transition`` was deleted and
  ``CAP_ROUTER_HIT_DEFERRED`` no longer exists — the ARP replaces both.

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
    REASON_ACQUISITION_UNDECLARED,
    REASON_BIND_MISSING,
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
    # cap_router selected cap-a -> the ARP: no acquisition declaration is wired
    # (deps carries no acquisition_inputs) -> acquisition-undeclared -> Agent.
    assert out is req and fb == REASON_ACQUISITION_UNDECLARED


async def test_stub_ambiguous_routes_through_cap_router(monkeypatch, caplog):
    out, req, rec, fb = await _run(
        monkeypatch, caplog, state=MATCH_AMBIGUOUS, backend="stub")
    assert rec.factory == ["stub"] and rec.recall and rec.aggregate
    assert not rec.legacy
    # ambiguous seeds are untrusted -> the stub answers NONE (the ARP is never reached)
    assert out is req and fb == REASON_CAP_ROUTER_NONE


# ── backend=stub + HIT: capability decided -> ARP directly, no re-selection ───────


async def test_stub_hit_goes_to_arp_without_reselection(monkeypatch, caplog):
    # The Matcher pinned the capability; the new lane must NOT re-select it
    # (neither cap_router nor legacy select_and_extract) and must short-circuit
    # BEFORE Recall / Aggregation. It goes STRAIGHT to the Argument Path Router:
    # with no acquisition declaration wired the ARP answers undeclared -> Agent.
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_HIT, backend="stub")
    assert not rec.factory           # no cap_router selection
    assert not rec.legacy            # no legacy capability selection
    assert not rec.recall            # no Recall
    assert not rec.aggregate         # no Candidate Aggregation
    assert out is req and fb == REASON_ACQUISITION_UNDECLARED


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


# ── Phase 4: the ARP is wired; the Phase 3 shim is deleted ───────────────────────


def test_hit_is_wired_straight_to_the_argument_path_router():
    import core.application.chat.intent_funnel.orchestrator as orch

    with open(orch.__file__, encoding="utf-8") as fh:
        src = (orch.__doc__ or "") + fh.read()
    assert "_acquisition_hop" in src and "path_router" in src   # actually wired, not dead
    # the Phase 3 compatibility limitation and its reason are gone entirely
    assert "CAP_ROUTER_HIT_DEFERRED" not in src
    assert "PHASE 3 COMPATIBILITY LIMITATION" not in src


def test_transition_shim_is_deleted_and_route_never_laundered():
    import core.application.chat.intent_funnel.orchestrator as orch
    from core.application.chat.intent_funnel import contract

    # the shim module no longer exists
    try:
        import core.application.chat.intent_funnel.selection_transition  # noqa: F401
    except ImportError:
        pass
    else:
        raise AssertionError("selection_transition must be deleted in Phase 4")
    # the transitional reason is removed from the vocabulary
    assert not hasattr(contract, "REASON_CAP_ROUTER_HIT_DEFERRED")
    with open(orch.__file__, encoding="utf-8") as fh:
        assert "selection_transition" not in fh.read()   # no residue wiring

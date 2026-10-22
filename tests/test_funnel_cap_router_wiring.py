"""Phase 3/4 — cap_router selection + Argument Path Router wiring.

Pins the LOCKED business rules:

* there is exactly ONE production selector (the deployed cap_router service);
  tests inject a double through ``ChatDeps.cap_router_selector`` — there is no
  backend switch and no legacy selection lane;
* a MATCH_HIT goes STRAIGHT to the Argument Path Router — it is NEVER re-selected
  (no Recall, no Aggregation, no cap_router), because the Matcher already pinned
  the capability;
* the selector is consulted for EVERY non-empty candidate set (K >= 1): there is
  no K=1 business-direct path and no K=2 ineligibility. K=1 hands over the single
  card, K=2 hands over both, K >= 4 is truncated to the top-3; only K=0 skips the
  selector entirely (``NO_CANDIDATE``);
* NONE -> ``CAP_ROUTER_NONE``; REJECT -> ``CAP_ROUTER_REJECT``; a selector that
  cannot serve -> ``CAP_ROUTER_UNAVAILABLE``; all exit to the Agent, never to a
  Qwen fallback;
* the production acquisition provider is built per turn when ``deps`` carries
  none, so a decided capability WITH parameters is declared and its MODEL slots
  need the extractor (absent here -> ``ACQUISITION_MODEL_PENDING`` -> Agent).

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
from core.application.chat.intent_funnel.cap_router.backends.stub import StubSelector
from core.application.chat.intent_funnel.contract import (
    MATCH_AMBIGUOUS,
    MATCH_HIT,
    MATCH_MISS,
    REASON_ACQUISITION_MODEL_PENDING,
    REASON_CAP_ROUTER_NONE,
    REASON_CAP_ROUTER_REJECT,
    REASON_CAP_ROUTER_UNAVAILABLE,
    REASON_NO_CANDIDATE,
    Candidate,
    MatchResult,
    RecallResult,
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


def _deps(selector=None):
    """``selector=None`` exercises the PRODUCTION path (the deployed service)."""
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        cap_router_selector=selector)


def _fallback(caplog) -> str:
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    return re.search(r"fallback_reason=(\S+)", lines[0]).group(1)


class _Rec:
    """Spy recorder for every node the cascade could touch."""

    def __init__(self):
        self.recall = []      # recall.load_index calls
        self.aggregate = []   # candidate_aggregation.aggregate_by_capability calls


_ONE = [Candidate("cap-a", 0.9, origin="recall", matched_example="create a folder")]
# Two RECALL candidates: K=2 — an ordinary candidate set handed to the selector
# (2 cards + the frozen REJECT card = 3 options).
_TWO = [Candidate("cap-a", 0.90, origin="recall", matched_example="create a folder"),
        Candidate("cap-b", 0.70, origin="recall", matched_example="does cap-b")]
# Three RECALL candidates with a clear margin: K=3 (3 cards + REJECT = 4 options).
_THREE = [Candidate("cap-a", 0.90, origin="recall", matched_example="create a folder"),
          Candidate("cap-b", 0.70, origin="recall", matched_example="does cap-b"),
          Candidate("cap-c", 0.60, origin="recall", matched_example="does cap-c")]
# Three RECALL candidates in a too-close race (both trusted, margin < 0.06):
# the selector IS consulted, and the stub never resolves the race.
_THREE_CLOSE = [Candidate("cap-a", 0.90, origin="recall"),
                Candidate("cap-b", 0.89, origin="recall"),
                Candidate("cap-c", 0.88, origin="recall")]


def _wire(monkeypatch, *, state, recall_candidates=_ONE,
          cids=("cap-a", "cap-b")) -> _Rec:
    entries = tuple(_entry(c) for c in cids)
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
            # three table patterns claim the turn (K=3 so the selector is consulted)
            return MatchResult(state=MATCH_AMBIGUOUS,
                               candidates=("cap-a", "cap-b", "cap-c"))
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
    return rec


async def _run(monkeypatch, caplog, *, state, selector=None,
               recall_candidates=_ONE, cids=("cap-a", "cap-b")):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    rec = _wire(monkeypatch, state=state, recall_candidates=recall_candidates,
                cids=cids)
    req = TurnRequirements()
    out = await funnel.route(_ctx(), deps=_deps(selector), requirements=req)
    return out, req, rec, _fallback(caplog)


# ── MISS/AMBIGUOUS -> Recall -> Aggregation -> cap_router ─────────────────────────


async def test_miss_routes_through_cap_router(monkeypatch, caplog):
    # K=3: the selector is consulted and picks cap-a (clear margin).
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=StubSelector(),
                                   recall_candidates=_THREE,
                                   cids=("cap-a", "cap-b", "cap-c"))
    assert rec.recall and rec.aggregate
    # cap_router selected cap-a -> the ARP: the production acquisition provider is
    # built per turn (deps carries none) so the capability IS declared; its
    # required MODEL slot needs the extractor, which deps does not inject here.
    assert out is req and fb == REASON_ACQUISITION_MODEL_PENDING


async def test_ambiguous_routes_through_cap_router(monkeypatch, caplog):
    # three ambiguous table patterns (K=3) + one recall hit: the mixed provenance
    # makes the stub answer NONE (the ARP is never reached).
    out, req, rec, fb = await _run(
        monkeypatch, caplog, state=MATCH_AMBIGUOUS, selector=StubSelector(),
        recall_candidates=_ONE, cids=("cap-a", "cap-b", "cap-c"))
    assert rec.recall and rec.aggregate
    assert out is req and fb == REASON_CAP_ROUTER_NONE


# ── HIT: capability decided -> ARP directly, no re-selection ─────────────────────


async def test_hit_goes_to_arp_without_reselection(monkeypatch, caplog):
    # The Matcher pinned the capability; the cascade must NOT re-select it and
    # must short-circuit BEFORE Recall / Aggregation. It goes STRAIGHT to the
    # Argument Path Router: the production provider declares the capability,
    # whose required MODEL slot needs the extractor (not injected here) -> Agent.
    sel = _Recording()
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_HIT,
                                   selector=sel)
    assert sel.calls == 0            # no cap_router selection
    assert not rec.recall            # no Recall
    assert not rec.aggregate         # no Candidate Aggregation
    assert out is req and fb == REASON_ACQUISITION_MODEL_PENDING


# ── NONE / unavailable fallbacks ─────────────────────────────────────────────────


async def test_none_exits_cap_router_none(monkeypatch, caplog):
    # a too-close three-candidate race (all trusted): the stub never resolves it
    out, req, rec, fb = await _run(
        monkeypatch, caplog, state=MATCH_MISS, selector=StubSelector(),
        recall_candidates=_THREE_CLOSE, cids=("cap-a", "cap-b", "cap-c"))
    assert out is req and fb == REASON_CAP_ROUTER_NONE


async def test_reject_route_exits_cap_router_reject(monkeypatch, caplog):
    # The V2 4th decision: the model answers REJECT -> ROUTE_REJECT -> the real
    # no-capability path (Agent). It never enters the argument chain.
    sel = _Recording(route=cap_router_mod.CapabilityRoute(
        cap_router_mod.ROUTE_REJECT, None))
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=_THREE,
                                   cids=("cap-a", "cap-b", "cap-c"))
    assert sel.calls == 1
    assert out is req and fb == REASON_CAP_ROUTER_REJECT


async def test_undeployed_service_exits_unavailable(monkeypatch, caplog):
    # No selector injected -> the PRODUCTION default (the deployed service). Its
    # endpoint is not configured in tests -> select() raises -> never falls
    # through to another model.
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   recall_candidates=_THREE,
                                   cids=("cap-a", "cap-b", "cap-c"))
    assert out is req and fb == REASON_CAP_ROUTER_UNAVAILABLE


async def test_selector_that_raises_exits_unavailable(monkeypatch, caplog):
    class _Boom:
        async def select(self, query, candidates, *, entries_by_id, facts=None):
            raise cap_router_mod.CapabilityRouterUnavailable("transport down")

    # K=3 so the selector is consulted (every K >= 1 reaches the selector now).
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=_Boom(), recall_candidates=_THREE,
                                   cids=("cap-a", "cap-b", "cap-c"))
    assert out is req and fb == REASON_CAP_ROUTER_UNAVAILABLE


# ── K normalization (§26.2): business layer shapes what the selector sees ─────────


class _Recording:
    """A selector double that records the candidate list it was handed."""

    def __init__(self, route=None):
        self.seen = None
        self.calls = 0
        self.route = route

    async def select(self, query, candidates, *, entries_by_id, facts=None):
        self.seen = list(candidates)
        self.calls += 1
        if self.route is not None:
            return self.route
        return cap_router_mod.CapabilityRoute(
            cap_router_mod.ROUTE_SELECTED, candidates[0].capability_id)


async def test_k1_consults_the_selector_with_the_single_candidate(
        monkeypatch, caplog):
    # K=1 is an ordinary candidate set: the single card is handed to the selector,
    # which selects it (or may legitimately REJECT). The business layer never
    # selects a capability directly and never fabricates a confidence.
    sel = _Recording()
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=_ONE)
    assert [c.capability_id for c in sel.seen] == ["cap-a"]
    assert out is req and fb == REASON_ACQUISITION_MODEL_PENDING


async def test_k1_selector_may_reject(monkeypatch, caplog):
    # K=1 is not a forced pick: the selector may answer REJECT (the real
    # no-capability path) even with a single card on the table.
    sel = _Recording(route=cap_router_mod.CapabilityRoute(
        cap_router_mod.ROUTE_REJECT, None))
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=_ONE)
    assert [c.capability_id for c in sel.seen] == ["cap-a"]
    assert out is req and fb == REASON_CAP_ROUTER_REJECT


async def test_k2_consults_the_selector_with_both_candidates(monkeypatch, caplog):
    # K=2 is NOT an ineligible/degradation case: both cards reach the selector.
    sel = _Recording()
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=_TWO)
    assert [c.capability_id for c in sel.seen] == ["cap-a", "cap-b"]
    assert out is req and fb == REASON_ACQUISITION_MODEL_PENDING


async def test_k2_selector_may_reject(monkeypatch, caplog):
    sel = _Recording(route=cap_router_mod.CapabilityRoute(
        cap_router_mod.ROUTE_REJECT, None))
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=_TWO)
    assert [c.capability_id for c in sel.seen] == ["cap-a", "cap-b"]
    assert out is req and fb == REASON_CAP_ROUTER_REJECT


async def test_k0_no_candidate_never_consults_the_selector(monkeypatch, caplog):
    # K=0 is the ONLY case that skips the selector: NO_CANDIDATE exits before the
    # cap_router stage is even entered.
    sel = _Recording()
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=[])
    assert sel.calls == 0
    assert out is req and fb == REASON_NO_CANDIDATE


async def test_k_at_least_4_truncates_to_top3_before_the_selector(
        monkeypatch, caplog):
    # 4 recall candidates -> the selector only ever sees the top-3 by score.
    four = [Candidate("cap-a", 0.95, origin="recall"),
            Candidate("cap-b", 0.90, origin="recall"),
            Candidate("cap-c", 0.85, origin="recall"),
            Candidate("cap-d", 0.80, origin="recall")]
    sel = _Recording()
    out, req, rec, fb = await _run(monkeypatch, caplog, state=MATCH_MISS,
                                   selector=sel, recall_candidates=four,
                                   cids=("cap-a", "cap-b", "cap-c", "cap-d"))
    assert [c.capability_id for c in sel.seen] == ["cap-a", "cap-b", "cap-c"]
    assert out is req and fb == REASON_ACQUISITION_MODEL_PENDING


async def test_trace_and_capture_expose_k_and_option_slots(monkeypatch, caplog):
    # trace/capture must distinguish raw_k / post_gate_k / effective_k /
    # option_slots, with option_slots == effective_k + 1 (cards + frozen REJECT).
    import core.application.chat.intent_funnel.orchestrator as orch

    _wire(monkeypatch, state=MATCH_MISS, recall_candidates=_TWO)
    sel = _Recording(route=cap_router_mod.CapabilityRoute(
        cap_router_mod.ROUTE_REJECT, None))
    trace, capture = orch.observability.new_trace(), {}
    req = TurnRequirements()
    await orch.run_nodes(_ctx(), _deps(sel), req, trace, capture=capture)
    assert trace["raw_k"] == 2 and trace["post_gate_k"] == 2
    assert trace["effective_k"] == 2 and trace["option_slots"] == 3
    assert capture["cap_router"]["effective_k"] == 2
    assert capture["cap_router"]["option_slots"] == 3


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

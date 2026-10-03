"""Phase 3 — cap_router stub backend + backend factory.

Pins the deterministic selector contract only: ONE capability or NONE, with the
leader-vs-runner-up margin discipline the legacy ToolIntentModel stub proved.
The stub authors NOTHING else — no arguments field exists on the output on
purpose (Phase 3 does not extract arguments).
"""
from __future__ import annotations

import pytest
from core.application.chat.intent_funnel.cap_router import (
    BACKEND_CAP_ROUTER,
    BACKEND_OFF,
    BACKEND_STUB,
    ROUTE_NONE,
    ROUTE_SELECTED,
)
from core.application.chat.intent_funnel.cap_router.backends import (
    CapabilitySelector,
    selector_for,
)
from core.application.chat.intent_funnel.cap_router.backends.service import CapRouterSelector
from core.application.chat.intent_funnel.cap_router.backends.stub import StubSelector
from core.application.chat.intent_funnel.contract import Candidate


def _c(cid, score, origin="recall", **kw):
    return Candidate(cid, score, origin=origin, **kw)


# ── the factory resolves only the shipped backend ────────────────────────────────


def test_factory_returns_stub_and_cap_router_else_none():
    s = selector_for(BACKEND_STUB)
    assert isinstance(s, StubSelector)
    assert isinstance(s, CapabilitySelector)  # honors the protocol
    # "cap_router" resolves UNCONDITIONALLY (endpoint resolution and availability
    # are separate concerns, ruling): a missing endpoint surfaces as
    # CapabilityRouterUnavailable from select(), NOT as a None selector here.
    cr = selector_for(BACKEND_CAP_ROUTER)
    assert isinstance(cr, CapRouterSelector)
    assert isinstance(cr, CapabilitySelector)  # honors the protocol
    # off / unknown -> None (the caller maps Agent)
    assert selector_for(BACKEND_OFF) is None
    assert selector_for("wat") is None


# ── the selection contract: ONE capability or NONE ───────────────────────────────


async def test_empty_candidate_set_is_none():
    route = await StubSelector().select("q", [], entries_by_id={})
    assert route.decision == ROUTE_NONE
    assert route.selected is False


async def test_single_trusted_candidate_selects():
    route = await StubSelector().select("q", [_c("cap-a", 0.9)], entries_by_id={})
    assert route.decision == ROUTE_SELECTED
    assert route.selected and route.capability_id == "cap-a"
    assert route.confidence == 0.9


async def test_single_matcher_hit_selects():
    route = await StubSelector().select(
        "q", [_c("cap-a", 1.0, origin="matcher_hit")], entries_by_id={})
    assert route.selected and route.capability_id == "cap-a"


async def test_single_matcher_ambiguous_escalates():
    # a table-ambiguous seed carries no score to trust -> NONE (never resolves)
    route = await StubSelector().select(
        "q", [_c("cap-a", 0.0, origin="matcher_ambiguous")], entries_by_id={})
    assert route.decision == ROUTE_NONE


async def test_clear_margin_selects_the_leader(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "chat_funnel_margin", 0.06)
    route = await StubSelector().select(
        "q", [_c("cap-a", 0.90), _c("cap-b", 0.70)], entries_by_id={})
    assert route.selected and route.capability_id == "cap-a"


async def test_close_race_is_none(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "chat_funnel_margin", 0.06)
    route = await StubSelector().select(
        "q", [_c("cap-a", 0.90), _c("cap-b", 0.89)], entries_by_id={})
    assert route.decision == ROUTE_NONE


async def test_mixed_provenance_is_none(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "chat_funnel_margin", 0.06)
    route = await StubSelector().select(
        "q", [_c("cap-a", 0.90),
              _c("cap-b", 0.10, origin="matcher_ambiguous")],
        entries_by_id={})
    assert route.decision == ROUTE_NONE


async def test_stub_never_authors_arguments():
    # the deliberate Phase 3 limit: selection only. The output carries no
    # arguments field at all -> nothing to extract, nothing to bind.
    route = await StubSelector().select("q", [_c("cap-a", 0.9)], entries_by_id={})
    assert not hasattr(route, "arguments")
    assert not hasattr(route, "args")

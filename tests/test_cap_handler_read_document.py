"""cap_handler — ReadDocumentHandler + its wiring into ``_acquisition_hop``.

Two layers are pinned here:

* the HANDLER contract itself (unit): ``cap-read-document`` resolves ``asset_id``
  ONLY from :class:`TurnFacts` (precedence ``attachment_asset_id`` ->
  ``path_asset_id`` -> ``viewer_asset_id``) and FAILS CLOSED (``None``) when none
  is present; ``pages`` comes ONLY from the viewer's declared range
  (``viewer_page_from``/``viewer_page_to``) — a page number in the SENTENCE is
  ignored — and a half-open/reversed range fails closed too (never a silent
  whole-document widening); it never emits a ``query`` slot (the capability has
  none);
* the WIRING: the registered handler short-circuits the generic acquisition chain
  (provider / path_router / Qwen) and its draft goes through the SAME ``_certify``
  -> Binder -> ``tool_intent`` handoff.

The funnel test drives the REAL ``funnel.route`` -> orchestrator cascade with a
faked Matcher HIT (mirroring ``test_phase4_acquisition_step2``).
"""
from __future__ import annotations

import logging
import re
import types

import pytest

from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    ReadDocumentHandler,
    handler_for,
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    MatchResult,
    TurnFacts,
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

# the shared strict parser — the handler's output must be legal for it
from core.infrastructure.pdf_pages import parse_pages_spec

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

ASSET = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"
MESSAGE = "帮我总结一下这份报告"


# ── Layer 1: asset_id sourcing + fail-closed ─────────────────────────────────────


async def test_read_doc_asset_id_prefers_attachment_then_path_then_viewer():
    handler = ReadDocumentHandler()
    all_three = TurnFacts(attachment_asset_id="A", path_asset_id="P", viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=all_three))["asset_id"] == "A"
    path_viewer = TurnFacts(path_asset_id="P", viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=path_viewer))["asset_id"] == "P"
    viewer_only = TurnFacts(viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=viewer_only))["asset_id"] == "V"


@pytest.mark.parametrize("facts", [None, TurnFacts(), TurnFacts(has_attachment=True)])
async def test_read_doc_without_an_asset_id_fails_closed(facts):
    handler = ReadDocumentHandler()
    assert await handler.acquire(query="总结第3页", facts=facts) is None


async def test_read_doc_never_emits_a_query_slot():
    handler = ReadDocumentHandler()
    draft = await handler.acquire(query=MESSAGE, facts=TurnFacts(viewer_asset_id=ASSET))
    assert set(draft) == {"asset_id"}            # no query/question on this capability


# ── Layer 1b: page scope comes from the VIEWER FACTS (never the sentence) ────────


@pytest.mark.parametrize("page_from,page_to,expected", [
    (3, 5, "3-5"),
    (3, 3, "3"),
    (12, 12, "12"),
    (7, 9, "7-9"),
])
async def test_read_doc_pages_come_from_the_viewer_range(page_from, page_to, expected):
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_page_from=page_from, viewer_page_to=page_to)
    draft = await handler.acquire(query="随便问一句", facts=facts)
    assert draft["pages"] == expected
    # the derived spec MUST be parseable by the shared strict parser
    assert parse_pages_spec(draft["pages"]) == parse_pages_spec(expected)


async def test_read_doc_omits_pages_for_the_whole_document():
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET)  # no page, no range -> whole document
    draft = await handler.acquire(query="总结全文", facts=facts)
    assert "pages" not in draft


@pytest.mark.parametrize("facts", [
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=3),          # half-open low
    TurnFacts(viewer_asset_id=ASSET, viewer_page_to=5),            # half-open high
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=5, viewer_page_to=3),  # reversed
])
async def test_read_doc_half_open_or_reversed_range_fails_closed(facts):
    # A page-scoped request is NEVER silently widened to the whole document.
    handler = ReadDocumentHandler()
    assert await handler.acquire(query="总结", facts=facts) is None


# ── Layer 1c: a page number in the SENTENCE is never the source of `pages` ───────


@pytest.mark.parametrize("message", [
    "总结第3页的内容",
    "读一下 3-5页",
    "read pages 3-5",
    "看下 p.3 的表格",
    "解读一下 2024 年财报",      # a year is not a page
    "ISO 9001 规范是什么",
])
async def test_read_doc_ignores_a_page_number_in_the_sentence(message):
    # Even when the sentence names pages, no fact carries a range → the slot is omitted;
    # page scope is settled upstream, not parsed from the user's text.
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET)  # no viewer range
    draft = await handler.acquire(query=message, facts=facts)
    assert "pages" not in draft


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_read_document():
    handler = handler_for("cap-read-document")
    assert isinstance(handler, ReadDocumentHandler)
    assert handler.capability_id == "cap-read-document"
    assert HANDLERS["cap-read-document"] is handler


def test_roster_leaves_unwired_capabilities_on_the_generic_chain():
    for cid in ("cap-open-pdf", "cap-translate", ""):
        assert handler_for(cid) is None


# ── Layer 3: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str, *, open_asset_id: str = "", page=None, page_from=None, page_to=None):
    viewer = (types.SimpleNamespace(asset_id=open_asset_id, page=page,
                                    page_from=page_from, page_to=page_to, selections=[])
              if open_asset_id else None)
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _doc_entry() -> CapabilityEntry:
    q = "读取文档内容"
    return CapabilityEntry(
        capability_id="cap-read-document", tool_binding="read_document",
        description="read an attached document",
        standard_queries=(QueryRecord(id="cap-read-document-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "asset_id": {"type": "string", "max_len": 64, "required": True,
                         "description": "the document asset id"},
            "pages": {"type": "string", "required": False, "description": "page spec"},
        },
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="读取文档内容")

    async def fake_select_and_extract(query, candidates, *, entries_by_id,
                                      llm=None, facts=None):
        rec.legacy.append(query)
        raise AssertionError("legacy select_and_extract must never run on the new lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(tool_intent_mod, "select_and_extract", fake_select_and_extract)
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")
    return rec, view


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_read_doc_turn_window_comes_from_the_viewer_facts(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _doc_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    # the viewer declares pages 3-5; the sentence says nothing about pages
    out = await _route(_ctx("帮我总结这一屏的内容", open_asset_id=ASSET,
                            page_from=3, page_to=5), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "read_document"
    assert act["capability_id"] == "cap-read-document"
    assert act["args"] == {"asset_id": ASSET, "pages": "3-5"}
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_read_doc_turn_ignores_a_page_number_in_the_sentence(monkeypatch, caplog):
    """A page named in the sentence is NOT an argument: the viewer fact owns scope."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _doc_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我总结第3页的内容", open_asset_id=ASSET), _deps(None), req)

    assert out is not req
    assert out.requested_action["args"] == {"asset_id": ASSET}    # no pages from the text
    assert not rec.legacy


async def test_read_doc_turn_without_a_document_exits_to_agent(monkeypatch, caplog):
    """No document in context -> the handler returns ``None`` -> the turn goes to
    the Agent with ``ACQUISITION_MISSING`` (fail-closed)."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _doc_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("读取一下文档内容"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_ACQUISITION_MISSING
    assert not rec.legacy                              # no Qwen, no legacy extraction


async def test_handler_bypasses_a_populated_generic_chain(monkeypatch, caplog):
    """Even with generic provider inputs that WOULD have produced a different
    draft, the registered handler owns the result — proof the branch precedes
    the generic chain."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _doc_entry()
    rec, _view = _wire(monkeypatch, entry)
    msg = "总结这份文档"
    poisoned = {"cap-read-document": AcquisitionInputs(
        model_values={"asset_id": "HALLUCINATED-UUID", "pages": "99"})}
    req = TurnRequirements()
    out = await _route(_ctx(msg, open_asset_id=ASSET),
                       _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    # facts win for asset_id; no page named -> pages omitted (never the "99")
    assert out.requested_action["args"] == {"asset_id": ASSET}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-read-document"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-read-document", _NoneHandler())
    entry = _doc_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query=MESSAGE, facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING

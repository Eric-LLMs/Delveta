"""cap_handler — ReadDocumentHandler + its wiring into ``_acquisition_hop``.

Two layers are pinned here:

* the HANDLER contract itself (unit): ``cap-read-document`` resolves ``asset_id``
  ONLY from :class:`TurnFacts` (precedence ``attachment_asset_id`` ->
  ``path_asset_id`` -> ``viewer_asset_id``) and FAILS CLOSED (``None``) when none
  is present; ``pages`` merges the handler's viewer-current-page value of the
  SAME document with the online model verdict. A valid model verdict wins; only
  a SEMANTIC ``ambiguous`` reply keeps the handler value. Technical failures
  (``invalid`` / ``unavailable`` / seam missing) are TERMINAL and never fall
  back. The viewer's open page/range is NEVER substituted for an unstated page:
  a whole-document request stays whole-document even when a viewer page exists.
  A deictic "this page" resolves to the viewer's current page only because the
  MODEL resolves it that way. The handler never emits a ``query`` slot (the
  capability has none);
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
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_SCOPE_UNSETTLED,
    HANDLERS,
    AcquisitionFailure,
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

# the shared strict parser — the handler's output must be legal for it
from core.infrastructure.pdf_pages import parse_pages_spec

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

ASSET = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"
MESSAGE = "帮我总结一下这份报告"


# ── the model stand-in ───────────────────────────────────────────────────────────
#
# No live model in the hermetic suite. These are RECORDED answers — the verdict
# a correct model gives for each sentence — keyed by the sentence itself. They
# are not a parser: a sentence missing from the table is answered as ambiguous,
# exactly like a model that cannot decide.

_RECORDED = {
    MESSAGE: (psr.SCOPE_ALL, None),
    "总结第3页的内容": (psr.SCOPE_SINGLE, "3"),
    "读一下 3-5页": (psr.SCOPE_RANGE, "3-5"),
    "read pages 3-5": (psr.SCOPE_RANGE, "3-5"),
    "打开第 2,4,6 页": (psr.SCOPE_RANGE, "2,4,6"),
    "看下 p.3 的表格": (psr.SCOPE_SINGLE, "3"),
    "总结全文": (psr.SCOPE_ALL, None),
    "总结": (psr.SCOPE_ALL, None),
    "帮我读第3到5页": (psr.SCOPE_RANGE, "3-5"),
    "帮我总结这一屏的内容": (psr.SCOPE_ALL, None),
    "总结这份文档": (psr.SCOPE_ALL, None),
    "读一下部分页": None,          # the sentence names no page
    "extract some pages": None,
}

_DEICTIC = ("把当前这一页读一下", "读一下这一页")


def _verdict(scope: str, pages: str | None) -> psr.PageScope:
    return psr.PageScope(target=psr.TARGET_DOCUMENT, page_scope=scope,
                         pages=pages)


def _ambiguous(reason: str) -> psr.PageScope:
    return psr.PageScope(target=psr.TARGET_DOCUMENT,
                         page_scope=psr.SCOPE_UNKNOWN,
                         status=psr.STATUS_AMBIGUOUS, reason=reason)


def _model_verdict(query: str, facts) -> psr.PageScope:
    if query in _DEICTIC:
        page = getattr(facts, "viewer_current_page", None)
        if page is None:
            return _ambiguous("no current page to resolve 'this page'")
        return _verdict(psr.SCOPE_SINGLE, str(page))
    recorded = _RECORDED.get(query, None)
    if recorded is None:
        return _ambiguous("the model could not settle the target or the scope")
    scope, pages = recorded
    return _verdict(scope, pages)


async def _resolver(*, query, facts, history=(), capability="", asset_id=""):
    return _model_verdict(query, facts)


async def _boom_resolver(**kwargs):
    raise AssertionError("the model must not be called without a target document")


# ── Layer 1: asset_id sourcing + fail-closed ─────────────────────────────────────


async def test_read_doc_asset_id_prefers_attachment_then_path_then_viewer():
    handler = ReadDocumentHandler()
    all_three = TurnFacts(attachment_asset_id="A", path_asset_id="P", viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=all_three,
                                  page_scope_resolver=_resolver))["asset_id"] == "A"
    path_viewer = TurnFacts(path_asset_id="P", viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=path_viewer,
                                  page_scope_resolver=_resolver))["asset_id"] == "P"
    viewer_only = TurnFacts(viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=viewer_only,
                                  page_scope_resolver=_resolver))["asset_id"] == "V"


@pytest.mark.parametrize("facts", [None, TurnFacts(), TurnFacts(has_attachment=True)])
async def test_read_doc_without_an_asset_id_fails_closed(facts):
    handler = ReadDocumentHandler()
    assert await handler.acquire(query="总结第3页", facts=facts,
                                 page_scope_resolver=_boom_resolver) is None


async def test_read_doc_never_emits_a_query_slot():
    handler = ReadDocumentHandler()
    draft = await handler.acquire(query=MESSAGE, facts=TurnFacts(viewer_asset_id=ASSET),
                                  page_scope_resolver=_resolver)
    assert set(draft) == {"asset_id"}            # no query/question on this capability


# ── Layer 1b: page scope — the sentence is the source, the viewer is not ────────


@pytest.mark.parametrize("message,expected", [
    ("总结第3页的内容", "3"),
    ("读一下 3-5页", "3-5"),
    ("read pages 3-5", "3-5"),
    ("打开第 2,4,6 页", "2,4,6"),
    ("看下 p.3 的表格", "3"),
])
async def test_read_doc_pages_come_from_the_sentence(message, expected):
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_current_page=12)
    draft = await handler.acquire(query=message, facts=facts,
                                  page_scope_resolver=_resolver)
    assert draft["pages"] == expected
    # the derived spec MUST be parseable by the shared strict parser
    assert parse_pages_spec(draft["pages"]) == parse_pages_spec(expected)


async def test_read_doc_deictic_page_uses_the_viewer_current_page():
    # "this page" is the ONE form whose value is the viewer's settled page — the
    # model is what resolves it that way.
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_current_page=12)
    draft = await handler.acquire(query="把当前这一页读一下", facts=facts,
                                  page_scope_resolver=_resolver)
    assert draft["pages"] == "12"


async def test_read_doc_omits_pages_for_the_whole_document():
    handler = ReadDocumentHandler()
    # A viewer page/range exists, yet the sentence states no restriction.
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_current_page=12,
                      viewer_page_from=3, viewer_page_to=5)
    draft = await handler.acquire(query="总结全文", facts=facts,
                                  page_scope_resolver=_resolver)
    assert draft == {"asset_id": ASSET}


@pytest.mark.parametrize("facts", [
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=3),          # half-open low
    TurnFacts(viewer_asset_id=ASSET, viewer_page_to=5),            # half-open high
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=5, viewer_page_to=3),  # reversed
])
async def test_read_doc_ignores_a_viewer_range_the_sentence_does_not_state(facts):
    # The viewer's range is not a page statement: the request stays whole-document.
    handler = ReadDocumentHandler()
    assert await handler.acquire(query="总结", facts=facts,
                                 page_scope_resolver=_resolver) == {"asset_id": ASSET}


# ── Layer 1c: semantic unknown vs technical failure ─────────────────────────────


@pytest.mark.parametrize("message", [
    "读一下部分页",
    "extract some pages",
])
async def test_read_doc_semantic_unknown_keeps_the_viewer_page(message):
    # The model replied that it did not extract a page; the viewer shows the
    # SAME document on page 12, so that is the handler's value.
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_current_page=12)
    out = await handler.acquire(query=message, facts=facts,
                                page_scope_resolver=_resolver)
    assert out == {"asset_id": ASSET, "pages": "12"}


async def test_read_doc_semantic_unknown_without_a_viewer_page_is_terminal():
    # No model extraction and no handler value: never a guess, never a silent
    # whole-document widening.
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET)
    out = await handler.acquire(query="读一下部分页", facts=facts,
                                page_scope_resolver=_resolver)
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED


@pytest.mark.parametrize("status", [
    psr.STATUS_INVALID,
    psr.STATUS_UNAVAILABLE,
])
async def test_read_doc_technical_failure_never_falls_back_to_the_viewer(status):
    # A timeout / non-JSON / contract violation is not a semantic "not
    # extracted": the viewer page may not be substituted and the turn fails.
    async def _bad(**kwargs):
        return psr.PageScope(target=psr.TARGET_DOCUMENT,
                             page_scope=psr.SCOPE_UNKNOWN,
                             status=status, reason="call failed")

    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_current_page=12)
    out = await handler.acquire(query=MESSAGE, facts=facts,
                                page_scope_resolver=_bad)
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED
    assert out.scope["pages"] is None


async def test_read_doc_without_the_model_seam_is_terminal():
    # A missing semantic judge is a technical failure -> the scope stays
    # unsettled. There is no deterministic fallback that could turn "some pages"
    # into the whole document, or into the viewer's page.
    handler = ReadDocumentHandler()
    facts = TurnFacts(viewer_asset_id=ASSET, viewer_current_page=12)
    out = await handler.acquire(query=MESSAGE, facts=facts)
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.scope["status"] == "not_wired"


async def test_read_doc_deictic_without_a_viewer_page_is_terminal():
    handler = ReadDocumentHandler()
    out = await handler.acquire(query="读一下这一页", facts=TurnFacts(viewer_asset_id=ASSET),
                                page_scope_resolver=_resolver)
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_read_document():
    handler = handler_for("cap-read-document")
    assert isinstance(handler, ReadDocumentHandler)
    assert handler.capability_id == "cap-read-document"
    assert HANDLERS["cap-read-document"] is handler


def test_roster_leaves_unwired_capabilities_on_the_generic_chain():
    for cid in ("cap-open-pdf", ""):
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


class _NoSelect:
    """Selector guard: a HIT-lane test fails loudly if selection is re-entered."""

    def __init__(self, rec: "_Rec") -> None:
        self.rec = rec

    async def select(self, query, candidates, *, entries_by_id, facts=None):
        self.rec.legacy.append(query)
        raise AssertionError("the selector must never run on a HIT lane")


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="读取文档内容")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    # A HIT is already decided: the ONE selector must never be consulted.
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: _NoSelect(rec))
    return rec, view


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider, page_scope_resolver=_resolver)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_read_doc_turn_reads_the_pages_named_in_the_sentence(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _doc_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    # the sentence names 3-5; the viewer's own range is a different one
    out = await _route(_ctx("帮我读第3到5页", open_asset_id=ASSET,
                            page_from=7, page_to=9), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "read_document"
    assert act["capability_id"] == "cap-read-document"
    assert act["args"] == {"asset_id": ASSET, "pages": "3-5"}
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_read_doc_turn_keeps_a_whole_document_request_whole(monkeypatch, caplog):
    """A viewer range is NOT a page statement: the whole document stays whole."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _doc_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我总结这一屏的内容", open_asset_id=ASSET,
                            page_from=3, page_to=5), _deps(None), req)

    assert out is not req
    assert out.requested_action["args"] == {"asset_id": ASSET}
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

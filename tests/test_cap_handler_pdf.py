"""cap_handler — the two PDF handlers (Phase 2-C) + the shared file-content scope.

The acceptance matrix, exercised over the THREE page-addressed handlers
(``read_document`` / ``pdf_extract_text`` / ``pdf_table_to_text``) because they share
ONE acquisition lane (``argument_acquisition.page_scope_settlement``):

1. a page named in the sentence (``第1页``/``pages 3-5``) -> the model's canonical
   spec (the ONLINE MODEL is the only page producer; no sentence parser runs);
2. a deictic "this page"                                -> the model resolves it to
   the viewer's current page;
3. no page restriction stated                           -> the slot is omitted
   (whole document), even when the viewer declares a page/range;
4. an unspelled page reference (``部分页``)              -> the model does not
   settle it: TERMINAL for the shared handlers, an Agent hand-off for the table
   capability — never a guess, never a silent whole-document widening;
5. no asset id in any fact                              -> ``acquire()`` -> ``None``;
6. a forged generic-chain provider / MODEL value is fully bypassed — the handler owns the
   draft and ``pages`` still comes from the trusted producers alone;
7. the tool layer still RAISES on an out-of-range/malformed spec, and a read-only tool
   failure escalates to the Agent (never swallowed into a silent whole-document read).
"""
from __future__ import annotations

import logging
import types

import pytest
from api.schemas import ViewerPayload
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.argument_acquisition.page_scope_resolver import (
    SCOPE_ALL,
    SCOPE_RANGE,
    SCOPE_SINGLE,
    SCOPE_UNKNOWN,
    STATUS_AMBIGUOUS,
    TARGET_DOCUMENT,
    PageScope,
)
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_SCOPE_UNSETTLED,
    HANDLERS,
    AcquisitionFailure,
    PdfExtractTextHandler,
    PdfTableToTextHandler,
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

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"
ASSET = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"

# capability_id -> (handler instance, its tool binding)
ALL_THREE = {
    "cap-read-document": (ReadDocumentHandler(), "read_document"),
    "cap-pdf-extract-text": (PdfExtractTextHandler(), "pdf_extract_text"),
    "cap-pdf-table-to-text": (PdfTableToTextHandler(), "pdf_table_to_text"),
}

TABLE = "cap-pdf-table-to-text"


# ── the model stand-in ───────────────────────────────────────────────────────────
#
# No live model in the hermetic suite. These are RECORDED answers — the verdict
# a correct model gives for each sentence — keyed by the sentence itself, plus
# the one deictic form whose answer depends on the viewer. Not a parser: an
# unrecorded sentence is answered as ambiguous.

_RECORDED = {
    "总结第12页": (SCOPE_SINGLE, "12"),
    "读一下第3到第5页": (SCOPE_RANGE, "3-5"),
    "summarise the whole document": (SCOPE_ALL, None),
    "summarise it": (SCOPE_ALL, None),
    "提取第2页的表格": (SCOPE_SINGLE, "2"),
    "第2页": (SCOPE_SINGLE, "2"),
    "提取第2到第4页 PDF 的表格": (SCOPE_RANGE, "2-4"),
    "q": (SCOPE_ALL, None),
    "读一下部分页": None,          # the sentence names no page
}

_DEICTIC = ("读一下当前这一页",)


def _model_verdict(query: str, facts) -> PageScope:
    """The verdict the online model returns for the sentence."""
    if query in _DEICTIC:
        page = getattr(facts, "viewer_current_page", None)
        if page is None:
            return PageScope(target=TARGET_DOCUMENT, page_scope=SCOPE_UNKNOWN,
                             status=STATUS_AMBIGUOUS,
                             reason="no current page to resolve 'this page'")
        return PageScope(target=TARGET_DOCUMENT, page_scope=SCOPE_SINGLE,
                         pages=str(page))
    recorded = _RECORDED.get(query, None)
    if recorded is None:
        return PageScope(target=TARGET_DOCUMENT, page_scope=SCOPE_UNKNOWN,
                         status=STATUS_AMBIGUOUS,
                         reason="the model could not settle the target or the scope")
    scope, pages = recorded
    return PageScope(target=TARGET_DOCUMENT, page_scope=scope, pages=pages)


def _model_resolver(query: str, facts):
    async def _resolve(**kwargs):
        return _model_verdict(query, facts)
    return _resolve


async def _resolver(*, query, facts, history=(), capability="", asset_id=""):
    """The seam shape the orchestrator calls (``ChatDeps.page_scope_resolver``)."""
    return _model_verdict(query, facts)


async def _boom_resolver(**kwargs):
    raise AssertionError("the model must not be called without a target document")


async def _acquire(handler, cid, query, facts, *, resolver=None):
    return await handler.acquire(
        query=query, facts=facts,
        page_scope_resolver=resolver or _model_resolver(query, facts))


def _facts(viewer) -> TurnFacts:
    """The settled facts for a viewer payload — the SAME path the funnel uses."""
    ctx = types.SimpleNamespace(
        body=types.SimpleNamespace(message="x", attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id="", session_id="s1",
    )
    return TurnFacts.of(ctx)


def _pdf_viewer(**kw) -> ViewerPayload:
    base = {"name": "p.pdf", "kind": "pdf", "asset_id": ASSET}
    base.update(kw)
    return ViewerPayload(**base)


# ── scenarios 1-3: single page, explicit range, whole document ────────────────────


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_a_sentence_named_page_is_a_single_page_spec(cid):
    handler, _ = ALL_THREE[cid]
    draft = await _acquire(handler, cid, "总结第12页", _facts(_pdf_viewer()))
    assert draft == {"asset_id": ASSET, "pages": "12"}


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_a_sentence_named_range_becomes_a_from_to_spec(cid):
    handler, _ = ALL_THREE[cid]
    draft = await _acquire(handler, cid, "读一下第3到第5页",
                           _facts(_pdf_viewer()))
    assert draft == {"asset_id": ASSET, "pages": "3-5"}


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_a_deictic_page_uses_the_viewer_current_page(cid):
    handler, _ = ALL_THREE[cid]
    draft = await _acquire(handler, cid, "读一下当前这一页",
                           _facts(_pdf_viewer(page=12)))
    assert draft == {"asset_id": ASSET, "pages": "12"}


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_no_page_restriction_follows_the_capability_policy(cid):
    # The viewer declares a page AND a range; the sentence states no restriction.
    # The shared handlers read the whole document; the table capability reads the
    # viewer's current page (it has no whole-document outcome).
    handler, _ = ALL_THREE[cid]
    draft = await _acquire(
        handler, cid, "summarise the whole document",
        _facts(_pdf_viewer(page=12, page_from=3, page_to=5)))
    if cid == "cap-pdf-table-to-text":
        assert draft == {"asset_id": ASSET, "pages": "12"}
    else:
        assert draft == {"asset_id": ASSET}


# ── scenario 4: a viewer range the sentence does not state is not a page scope ─────


@pytest.mark.parametrize("cid", ALL_THREE)
@pytest.mark.parametrize("facts", [
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=3),                     # half-open low
    TurnFacts(viewer_asset_id=ASSET, viewer_page_to=5),                       # half-open high
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=5, viewer_page_to=3),   # reversed
])
async def test_a_viewer_range_the_sentence_does_not_state_is_ignored(cid, facts):
    handler, _ = ALL_THREE[cid]
    out = await _acquire(handler, cid, "summarise it", facts)
    if cid == "cap-pdf-table-to-text":
        # No current page -> the table capability asks the Agent, never widens
        # the request to the whole document.
        assert isinstance(out, AcquisitionFailure)
        assert out.terminal is False
        assert out.category == CATEGORY_SCOPE_UNSETTLED
    else:
        assert out == {"asset_id": ASSET}


@pytest.mark.asyncio
@pytest.mark.parametrize("cid", ["cap-read-document", "cap-pdf-extract-text"])
async def test_an_unspelled_page_reference_is_a_terminal_failure(cid):
    # The model cannot name the pages, so the scope is unsettled: these two
    # capabilities end the turn rather than guess.
    handler, _ = ALL_THREE[cid]
    out = await _acquire(handler, cid, "读一下部分页", _facts(_pdf_viewer(page=12)))
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED


@pytest.mark.asyncio
@pytest.mark.parametrize("cid", ["cap-read-document", "cap-pdf-extract-text"])
async def test_an_unwired_model_seam_is_also_terminal(cid):
    # No semantic judge at all — still never a guess and never a whole-document
    # widening for these two capabilities.
    handler, _ = ALL_THREE[cid]
    out = await handler.acquire(query="读一下部分页", facts=_facts(_pdf_viewer(page=12)))
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.scope["status"] == "not_wired"


@pytest.mark.asyncio
async def test_an_unspelled_page_reference_hands_off_to_the_agent_for_tables():
    # The table capability has no whole-document outcome and never terminates: an
    # unsettled scope goes to the Agent (non-terminal), even with a viewer page.
    handler, _ = ALL_THREE["cap-pdf-table-to-text"]
    out = await _acquire(handler, "cap-pdf-table-to-text", "读一下部分页",
                         _facts(_pdf_viewer(page=12)))
    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED


# ── scenario 5: no asset id fails closed ───────────────────────────────────────────


@pytest.mark.parametrize("cid", ALL_THREE)
@pytest.mark.parametrize("facts", [None, TurnFacts(), TurnFacts(has_attachment=True)])
async def test_without_an_asset_id_fails_closed(cid, facts):
    handler, _ = ALL_THREE[cid]
    assert await _acquire(handler, cid, "q", facts,
                          resolver=_boom_resolver) is None


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_asset_id_precedence_attachment_then_path_then_viewer(cid):
    handler, _ = ALL_THREE[cid]
    # The table capability needs a settled page too; an explicit page keeps this
    # test about identity precedence only.
    query = "提取第2页的表格" if cid == "cap-pdf-table-to-text" else "q"
    assert (await _acquire(handler, cid, query, TurnFacts(
        attachment_asset_id="A", path_asset_id="P", viewer_asset_id="V")))["asset_id"] == "A"
    assert (await _acquire(handler, cid, query, TurnFacts(
        path_asset_id="P", viewer_asset_id="V")))["asset_id"] == "P"
    assert (await _acquire(handler, cid,
                           "第2页" if cid == "cap-pdf-table-to-text" else "q",
                           TurnFacts(viewer_asset_id="V")))["asset_id"] == "V"


# ── roster wiring ──────────────────────────────────────────────────────────────────


def test_roster_wires_the_two_pdf_handlers():
    for cid, (handler, _) in (
        ("cap-pdf-extract-text", (PdfExtractTextHandler(), "pdf_extract_text")),
        ("cap-pdf-table-to-text", (PdfTableToTextHandler(), "pdf_table_to_text")),
    ):
        wired = handler_for(cid)
        assert isinstance(wired, type(handler))
        assert wired.capability_id == cid
        assert HANDLERS[cid] is wired


# ── scenario 6: the funnel handler bypasses a forged generic chain ─────────────────


def _ctx(message: str, *, open_asset_id: str = "", page_from=None, page_to=None):
    viewer = (types.SimpleNamespace(asset_id=open_asset_id, page=None,
                                    page_from=page_from, page_to=page_to, selections=[])
              if open_asset_id else None)
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _pdf_entry(cid: str, tool: str) -> CapabilityEntry:
    q = "读取 PDF"
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description="read a pdf",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "asset_id": {"type": "string", "max_len": 64, "required": True,
                         "description": "the pdf asset id"},
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
                           matched_literal="读取 PDF")

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


async def test_pdf_handler_bypasses_a_forged_generic_chain(monkeypatch):
    entry = _pdf_entry("cap-pdf-extract-text", "pdf_extract_text")
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-pdf-extract-text": AcquisitionInputs(
        model_values={"asset_id": "HALLUCINATED-UUID", "pages": "99"})}
    req = TurnRequirements()
    out = await _route(_ctx("提取第2到第4页 PDF 的表格", open_asset_id=ASSET),
                       _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    # the trusted producers win for asset_id AND for pages — the forged "99" and
    # the hallucinated id are ignored
    assert out.requested_action["args"] == {"asset_id": ASSET, "pages": "2-4"}
    assert not rec.legacy


async def test_pdf_turn_without_a_document_exits_to_agent(monkeypatch, caplog):
    import re
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _pdf_entry("cap-pdf-table-to-text", "pdf_table_to_text")
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("读取一下 PDF"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_ACQUISITION_MISSING
    assert not rec.legacy


# ── scenario 7: the tool layer fails closed and escalates to the Agent ─────────────


def _two_page_pdf() -> bytes:
    import pymupdf
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


class _FakeRuntime:
    def __init__(self):
        self.defs: dict[str, object] = {}

    def register(self, definition):
        self.defs[definition.name] = definition


class _FakeLLM:
    async def chat(self, messages):
        return {"content": "TABLE"}


async def test_tool_layer_out_of_range_spec_raises_never_widens(monkeypatch):
    from apps.api.tools import pdf_tools

    async def _fake_load(asset_id, ctx):
        return _two_page_pdf()

    monkeypatch.setattr(pdf_tools, "_load_asset_bytes", _fake_load)
    rt = _FakeRuntime()
    pdf_tools.register(rt, ctx=None, llm=_FakeLLM())

    # a spec the file cannot satisfy bubbles as a hard error — never a whole-document read
    for tool in ("pdf_extract_text", "pdf_table_to_text"):
        with pytest.raises(ValueError, match="out of range"):
            await rt.defs[tool].execute({"asset_id": "a", "pages": "99"}, None)


def test_pdf_read_tools_are_read_only_so_any_failure_escalates_to_the_agent():
    # read-only tools are NOT in the mutating set, so ANY tool failure classifies as
    # ActionPreflightFailure → the executor escalates (Agent may clarify) — never a
    # silent success and never a whole-document substitute.
    from apps.api.routers.chat import _MUTATING_DIRECT_TOOLS
    for tool in ("pdf_extract_text", "pdf_table_to_text", "read_document"):
        assert tool not in _MUTATING_DIRECT_TOOLS

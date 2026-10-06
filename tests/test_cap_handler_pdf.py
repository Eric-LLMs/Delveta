"""cap_handler — the two PDF handlers (Phase 2-C) + the shared file-content scope.

The Phase-2-C acceptance matrix, exercised over the THREE page-addressed handlers
(``read_document`` / ``pdf_extract_text`` / ``pdf_table_to_text``) because they share
ONE scope rule (:mod:`core.application.chat.intent_funnel.cap_handler.scope`):

1. a lone current page (``page=12``)                 -> ``pages="12"``;
2. an explicit range (``from=3``, ``to=5``)          -> ``pages="3-5"``;
3. no page at all                                    -> the slot is omitted (whole document);
4. a half-open (``3,None``)/(``None,5``) or reversed -> fail-closed: ``acquire()`` -> ``None``;
5. no asset id in any fact                           -> fail-closed: ``acquire()`` -> ``None``;
6. a forged generic-chain provider / MODEL value is fully bypassed — the handler owns the
   draft and ``pages`` still comes from the facts alone;
7. the tool layer still RAISES on an out-of-range/malformed spec, and a read-only tool
   failure escalates to the Agent (never swallowed into a silent whole-document read).
"""
from __future__ import annotations

import logging
import types

import pytest
from api.schemas import ViewerPayload
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
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
from core.config import settings

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"
ASSET = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"

# capability_id -> (handler instance, its tool binding)
ALL_THREE = {
    "cap-read-document": (ReadDocumentHandler(), "read_document"),
    "cap-pdf-extract-text": (PdfExtractTextHandler(), "pdf_extract_text"),
    "cap-pdf-table-to-text": (PdfTableToTextHandler(), "pdf_table_to_text"),
}


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
async def test_current_page_is_a_single_page_spec(cid):
    handler, _ = ALL_THREE[cid]
    draft = await handler.acquire(query="q", facts=_facts(_pdf_viewer(page=12)))
    assert draft == {"asset_id": ASSET, "pages": "12"}


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_explicit_range_becomes_a_from_to_spec(cid):
    handler, _ = ALL_THREE[cid]
    draft = await handler.acquire(
        query="q", facts=_facts(_pdf_viewer(page_from=3, page_to=5)))
    assert draft == {"asset_id": ASSET, "pages": "3-5"}


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_no_page_omits_the_slot(cid):
    handler, _ = ALL_THREE[cid]
    draft = await handler.acquire(query="q", facts=_facts(_pdf_viewer()))
    assert draft == {"asset_id": ASSET}


# ── scenario 4: half-open / reversed range fails closed ────────────────────────────


@pytest.mark.parametrize("cid", ALL_THREE)
@pytest.mark.parametrize("facts", [
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=3),                     # half-open low
    TurnFacts(viewer_asset_id=ASSET, viewer_page_to=5),                       # half-open high
    TurnFacts(viewer_asset_id=ASSET, viewer_page_from=5, viewer_page_to=3),   # reversed
])
async def test_partial_or_reversed_range_fails_closed(cid, facts):
    handler, _ = ALL_THREE[cid]
    assert await handler.acquire(query="q", facts=facts) is None


# ── scenario 5: no asset id fails closed ───────────────────────────────────────────


@pytest.mark.parametrize("cid", ALL_THREE)
@pytest.mark.parametrize("facts", [None, TurnFacts(), TurnFacts(has_attachment=True)])
async def test_without_an_asset_id_fails_closed(cid, facts):
    handler, _ = ALL_THREE[cid]
    assert await handler.acquire(query="q", facts=facts) is None


@pytest.mark.parametrize("cid", ALL_THREE)
async def test_asset_id_precedence_attachment_then_path_then_viewer(cid):
    handler, _ = ALL_THREE[cid]
    assert (await handler.acquire(query="q", facts=TurnFacts(
        attachment_asset_id="A", path_asset_id="P", viewer_asset_id="V")))["asset_id"] == "A"
    assert (await handler.acquire(query="q", facts=TurnFacts(
        path_asset_id="P", viewer_asset_id="V")))["asset_id"] == "P"
    assert (await handler.acquire(query="q", facts=TurnFacts(
        viewer_asset_id="V")))["asset_id"] == "V"


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


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="读取 PDF")

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


async def test_pdf_handler_bypasses_a_forged_generic_chain(monkeypatch):
    entry = _pdf_entry("cap-pdf-extract-text", "pdf_extract_text")
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-pdf-extract-text": AcquisitionInputs(
        model_values={"asset_id": "HALLUCINATED-UUID", "pages": "99"})}
    req = TurnRequirements()
    out = await _route(_ctx("提取这一屏 PDF 的表格", open_asset_id=ASSET,
                            page_from=2, page_to=4),
                       _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    # facts win for asset_id AND for pages — the forged "99"/hallucinated id are ignored
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

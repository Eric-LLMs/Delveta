"""``cap-pdf-table-to-text`` page-scope policy — the capability-local merge.

The business rules under test:

1. no target document at all -> no draft (fail-closed, never a guessed document);
2. the ONLINE MODEL is the only semantic judge of the sentence; a valid
   whole-document / page / range verdict wins;
3. an explicit page / range from the model is used as-is and the Viewer's
   current page never overrides it; an explicit whole-document verdict omits
   ``pages``;
4. only a SEMANTIC ``ambiguous`` verdict (the model replied that it did not
   extract a page) keeps the Viewer's current page of the SAME document;
   without one the turn goes to the Agent;
5. a failed / unavailable call or a contract violation -> the Agent, with NO
   fallback to the handler value.

The last group drives the REAL acquisition hop (handler -> Binder -> certified
ACTION / Agent exit), so the assertions cover the final tool arguments and the
Agent hand-off, not just the policy function.
"""
from __future__ import annotations

import types

import pytest
from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_EXTRACTION_FAILED,
    CATEGORY_SCOPE_UNSETTLED,
    PdfTableToTextHandler,
)
from core.application.chat.intent_funnel.cap_handler.acquisition_failure import (
    CATEGORY_SCOPE_CONTRACT,
)
from core.application.chat.intent_funnel.cap_handler.table_page_scope import (
    PAGE_UNDETERMINED,
    SETTLED,
    SOURCE_LLM,
    SOURCE_VIEWER,
    settle_table_page_scope,
)
from core.application.chat.intent_funnel.contract import (
    REASON_ACQUISITION_MISSING,
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

ASSET = "asset-table"
VIEWER = "asset-viewer"
CAP = "cap-pdf-table-to-text"
TOOL = "pdf_table_to_text"


def _facts(**over) -> TurnFacts:
    base = {
        "has_viewer": True,
        "viewer_asset_id": ASSET,
        "viewer_current_page": 3,
        "viewer_kind": "pdf",
        "viewer_name": "survey.pdf",
    }
    base.update(over)
    return TurnFacts(**base)


def _scope(**over) -> psr.PageScope:
    base = {
        "target": psr.TARGET_DOCUMENT,
        "page_scope": psr.SCOPE_ALL,
        "pages": None,
        "status": psr.STATUS_OK,
        "reason": "",
    }
    base.update(over)
    return psr.PageScope(**base)


def _ambiguous(**over) -> psr.PageScope:
    """A semantically legal "page not extracted" reply."""
    return _scope(status=psr.STATUS_AMBIGUOUS, page_scope=psr.SCOPE_UNKNOWN,
                  **over)


def _resolver(scope=None, *, exc: Exception | None = None, seen=None):
    async def _resolve(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        if exc is not None:
            raise exc
        return scope
    return _resolve


def _boom(**kwargs):
    raise AssertionError("the model must not be called for this turn")


async def _acquire(query, facts, resolver):
    return await PdfTableToTextHandler().acquire(
        query=query, facts=facts, page_scope_resolver=resolver, history=())


# ── the model's valid verdict wins ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_model_page_is_used_for_an_unusual_page_form():
    # "第 3 张" is a page to a reader; the model reads it, so the page is honoured.
    out = await _acquire(
        "翻到第 3 张", _facts(viewer_current_page=1),
        _resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="3")))

    assert out == {"asset_id": ASSET, "pages": "3"}


@pytest.mark.asyncio
async def test_the_model_range_is_not_overridden_by_the_viewer_page():
    out = await _acquire(
        "读第1到2页", _facts(viewer_current_page=2),
        _resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="1-2")))

    assert out == {"asset_id": ASSET, "pages": "1-2"}


@pytest.mark.asyncio
async def test_the_model_page_wins_over_the_sentence_form():
    seen: dict = {}
    out = await _acquire(
        "提取第3页的表格", _facts(viewer_current_page=1),
        _resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="4"), seen=seen))

    assert out == {"asset_id": ASSET, "pages": "4"}
    assert seen["capability"] == CAP
    assert seen["asset_id"] == ASSET


@pytest.mark.asyncio
async def test_a_deictic_page_is_resolved_by_the_model_from_the_viewer_page():
    # The model is given the current page and returns it as an explicit page.
    out = await _acquire(
        "读一下当前这一页", _facts(viewer_current_page=7),
        _resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="7")))

    assert out == {"asset_id": ASSET, "pages": "7"}


# ── the model identified the whole document ──────────────────────────────────


@pytest.mark.asyncio
async def test_a_whole_document_verdict_omits_the_pages_slot():
    out = await _acquire(
        "把整个文档的表格都转成文本", _facts(viewer_current_page=3),
        _resolver(_scope(page_scope=psr.SCOPE_ALL)))

    assert out == {"asset_id": ASSET}


@pytest.mark.asyncio
async def test_a_whole_document_verdict_is_not_narrowed_to_the_viewer_page():
    # The viewer sits on page 1; the request names no page, so the model's
    # explicit whole-document verdict must not shrink to that page.
    out = await _acquire(
        "提取第3页的表格", _facts(viewer_current_page=1),
        _resolver(_scope(page_scope=psr.SCOPE_ALL)))

    assert out == {"asset_id": ASSET}


@pytest.mark.asyncio
async def test_a_missing_current_page_hands_the_turn_to_the_agent():
    evidence: dict = {}
    out = await PdfTableToTextHandler().acquire(
        query="把这些页的表格转成文本",
        facts=_facts(viewer_current_page=None),
        page_scope_resolver=_resolver(_ambiguous()),
        history=(), scope_evidence=evidence)

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED
    assert evidence["ok"] is False
    assert evidence["merged"]["pages"] is None
    assert evidence["handler_pages"] is None
    assert evidence["fallback_applied"] is False
    assert out.context["viewer_asset_id"] == ASSET


@pytest.mark.asyncio
async def test_an_invalid_current_page_hands_the_turn_to_the_agent():
    for bad in (0, -3, "none", ""):
        out = await _acquire(
            "把这些页的表格转成文本", _facts(viewer_current_page=bad),
            _resolver(_ambiguous()))
        assert out is not None and out.terminal is False, bad
        assert out.category == CATEGORY_SCOPE_UNSETTLED, bad


@pytest.mark.asyncio
async def test_a_viewer_page_from_another_document_is_not_reused():
    # The target is the attachment; the viewer's page belongs to a different
    # document, so it is not a valid page for this request -> Agent.
    facts = TurnFacts(
        has_viewer=True, has_attachment=True,
        attachment_asset_id="asset-attachment",
        viewer_asset_id=VIEWER, viewer_current_page=8,
        viewer_kind="pdf", viewer_name="other.pdf")
    out = await _acquire("把这些页的表格转成文本", facts,
                         _resolver(_ambiguous()))

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED


# ── no target document -> no draft, no model call ────────────────────────────


@pytest.mark.asyncio
async def test_no_target_document_produces_no_draft():
    assert await _acquire("把表格转成文本", TurnFacts(), _boom) is None
    assert await _acquire(
        "把表格转成文本", TurnFacts(has_attachment=True), _boom) is None


# ── technical failures -> Agent, never the handler value ─────────────────────


@pytest.mark.asyncio
async def test_a_model_exception_hands_the_turn_to_the_agent():
    # An explicit page is visible in the sentence; it still may not settle the
    # turn, because the semantic judge failed.
    out = await _acquire(
        "提取第3页的表格", _facts(),
        _resolver(exc=TimeoutError("scope model timeout")))

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_EXTRACTION_FAILED


@pytest.mark.asyncio
async def test_an_unavailable_model_hands_the_turn_to_the_agent():
    out = await _acquire(
        "提取第3页的表格", _facts(),
        _resolver(_scope(status=psr.STATUS_UNAVAILABLE, reason="timeout")))

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_EXTRACTION_FAILED


@pytest.mark.asyncio
async def test_a_contract_violation_hands_the_turn_to_the_agent():
    out = await _acquire(
        "提取第3页的表格", _facts(),
        _resolver(_scope(status=psr.STATUS_INVALID,
                         reason="unexpected field asset_id")))

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_CONTRACT


@pytest.mark.asyncio
async def test_a_semantic_unknown_keeps_the_viewer_current_page():
    # The model replied that it did not extract a page; the handler value (the
    # viewer's current page of the same document) is the approved fallback.
    evidence: dict = {}
    out = await PdfTableToTextHandler().acquire(
        query="读一下部分页", facts=_facts(viewer_current_page=3),
        page_scope_resolver=_resolver(_ambiguous()),
        history=(), scope_evidence=evidence)

    assert out == {"asset_id": ASSET, "pages": "3"}
    assert evidence["source"] == SOURCE_VIEWER
    assert evidence["handler_pages"] == "3"
    assert evidence["fallback_applied"] is True


@pytest.mark.asyncio
async def test_a_conversation_target_hands_the_turn_to_the_agent():
    out = await _acquire(
        "把刚才说的表格再发我一遍", _facts(viewer_current_page=3),
        _resolver(_scope(target=psr.TARGET_CONVERSATION)))

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED


@pytest.mark.asyncio
async def test_an_unwired_model_seam_hands_the_turn_to_the_agent():
    # No semantic judge -> no settlement, even when the sentence names a page.
    out = await PdfTableToTextHandler().acquire(
        query="提取第3页的表格", facts=_facts(), history=())

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_EXTRACTION_FAILED


# ── the policy function's own evidence ────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_policy_records_which_producer_supplied_the_page():
    from_model = await settle_table_page_scope(
        query="提取第2页", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="2")),
        asset_id=ASSET)
    assert from_model.status == SETTLED and from_model.source == SOURCE_LLM
    assert from_model.fallback_applied is False

    whole = await settle_table_page_scope(
        query="把整个文档的表格转成文本", facts=_facts(),
        resolver=_resolver(_scope()), asset_id=ASSET)
    assert whole.status == SETTLED and whole.source == SOURCE_LLM
    assert whole.pages is None
    assert whole.handler_pages == "3"

    from_viewer = await settle_table_page_scope(
        query="把表格转成文本", facts=_facts(),
        resolver=_resolver(_ambiguous()), asset_id=ASSET)
    assert from_viewer.source == SOURCE_VIEWER and from_viewer.pages == "3"
    assert from_viewer.fallback_applied is True

    undetermined = await settle_table_page_scope(
        query="把表格转成文本", facts=_facts(viewer_current_page=None),
        resolver=_resolver(_ambiguous()), asset_id=ASSET)
    assert undetermined.status == PAGE_UNDETERMINED
    assert undetermined.fallback_applied is False


# ── the real acquisition hop: Binder args and the Agent exit ──────────────────


def _entry() -> CapabilityEntry:
    q = "把表格转成文本"
    return CapabilityEntry(
        capability_id=CAP, tool_binding=TOOL,
        description="convert the tables in a pdf to text",
        standard_queries=(QueryRecord(id=f"{CAP}-q", query=q,
                                      language=derive_language(q)),),
        parameters={
            "asset_id": {"type": "string", "required": True,
                         "description": "the pdf asset id"},
            "pages": {"type": "string", "required": False,
                      "description": "page spec"},
        },
    )


def _view(entry: CapabilityEntry) -> RegistryLiveView:
    return RegistryLiveView(fingerprint=content_fingerprint([entry]),
                            entries=(entry,))


async def _hop(query, facts, resolver=None, *, failure=None):
    entry = _entry()
    trace: dict = {}
    capture: dict = {}
    deps = types.SimpleNamespace(argument_extractor=None,
                                 page_scope_resolver=resolver)
    out = await orch_mod._acquisition_hop(
        TurnRequirements(), deps, entry, query=query, facts=facts,
        view=_view(entry), trace=trace, capture=capture, history=(),
        failure=failure)
    return out, trace, capture


@pytest.mark.asyncio
async def test_the_hop_certifies_the_model_page_into_the_binder_arguments():
    out, trace, capture = await _hop(
        "读第1到2页", _facts(viewer_current_page=5),
        _resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="1-2")))

    # The Binder's final arguments carry the model's explicit page, not the
    # viewer's page.
    assert out.requested_action["tool"] == TOOL
    assert out.requested_action["args"] == {"asset_id": ASSET, "pages": "1-2"}
    assert trace["acquisition"] == "CAP_HANDLER"
    assert capture["page_scope_evidence"]["source"] == SOURCE_LLM


@pytest.mark.asyncio
async def test_the_hop_hands_a_model_failure_to_the_agent_with_context():
    failure: dict = {}
    out, trace, capture = await _hop(
        "提取第3页的表格", _facts(),
        _resolver(exc=RuntimeError("model down")), failure=failure)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    record = capture["acquisition"]["failure"]
    assert record["category"] == CATEGORY_EXTRACTION_FAILED
    assert record["stage"] == "argument_extraction"
    assert "terminal" not in record
    assert record["context"]["viewer_asset_id"] == ASSET
    assert failure.get("acquisition", {}).get("category") == CATEGORY_EXTRACTION_FAILED


@pytest.mark.asyncio
async def test_the_hop_certifies_a_whole_document_verdict_without_pages():
    out, trace, capture = await _hop(
        "把整个文档的表格转成文本", _facts(viewer_current_page=5),
        _resolver(_scope(page_scope=psr.SCOPE_ALL)))

    assert out.requested_action["tool"] == TOOL
    assert out.requested_action["args"] == {"asset_id": ASSET}
    assert capture["page_scope_evidence"]["source"] == SOURCE_LLM
    assert capture["page_scope_evidence"]["merged"]["pages"] is None


@pytest.mark.asyncio
async def test_the_hop_certifies_the_handler_page_after_a_semantic_unknown():
    out, trace, capture = await _hop(
        "读一下这些页的表格", _facts(viewer_current_page=3),
        _resolver(_ambiguous()))

    assert out.requested_action["args"] == {"asset_id": ASSET, "pages": "3"}
    assert capture["page_scope_evidence"]["source"] == SOURCE_VIEWER
    assert capture["page_scope_evidence"]["fallback_applied"] is True


@pytest.mark.asyncio
async def test_the_hop_hands_a_missing_viewer_page_to_the_agent():
    out, trace, capture = await _hop(
        "把这些页的表格转成文本", _facts(viewer_current_page=None),
        _resolver(_ambiguous()))

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    record = capture["acquisition"]["failure"]
    assert record["category"] == CATEGORY_SCOPE_UNSETTLED
    assert "terminal" not in record
    assert record["context"]["viewer_current_page"] is None


@pytest.mark.asyncio
async def test_the_hop_does_not_guess_a_document_when_none_is_present():
    out, trace, capture = await _hop("把表格转成文本", TurnFacts(), _boom)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    assert "acquisition" not in capture

"""``cap-pdf-table-to-text`` page-scope policy — the capability-local rule.

The business priority under test:

1. an explicit page/range in the sentence is the scope;
2. with no page named, a valid target document that is the open viewer document
   resolves to the viewer's current page — model inference never overrides it;
3. no target document at all -> no draft (fail-closed, never a guessed document);
4. a target document with no valid current page -> Agent clarification, never a
   whole-document widening;
5. a failed model call or a contract-violating reply -> Agent, no fallback.

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
    SOURCE_RULE,
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


def _resolver(scope=None, *, exc: Exception | None = None, seen=None):
    async def _resolve(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        if exc is not None:
            raise exc
        return scope
    return _resolve


async def _acquire(query, facts, resolver=None):
    return await PdfTableToTextHandler().acquire(
        query=query, facts=facts, page_scope_resolver=resolver, history=())


# ── priority 1: an explicit page in the sentence ──────────────────────────────


@pytest.mark.asyncio
async def test_an_explicit_page_is_taken_from_the_model_verdict():
    out = await _acquire(
        "extract the tables on pages 3 to 5", _facts(viewer_current_page=1),
        _resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="3-5")))

    assert out == {"asset_id": ASSET, "pages": "3-5"}


@pytest.mark.asyncio
async def test_the_model_may_correct_the_rule_parse():
    seen: dict = {}
    out = await _acquire(
        "提取第3页的表格", _facts(viewer_current_page=1),
        _resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="4"), seen=seen))

    assert out == {"asset_id": ASSET, "pages": "4"}
    assert seen["capability"] == CAP
    assert seen["asset_id"] == ASSET


@pytest.mark.asyncio
async def test_the_model_cannot_erase_an_explicit_page_with_a_whole_document_verdict():
    # "all" is not an outcome of this capability. The page named in the sentence
    # stands even when the model reports the whole document.
    out = await _acquire(
        "提取第3页的表格", _facts(viewer_current_page=1),
        _resolver(_scope(page_scope=psr.SCOPE_ALL)))

    assert out == {"asset_id": ASSET, "pages": "3"}


@pytest.mark.asyncio
async def test_an_ambiguous_model_verdict_keeps_the_explicit_page():
    out = await _acquire(
        "读一下第2到4页的表格", _facts(viewer_current_page=1),
        _resolver(_scope(status=psr.STATUS_AMBIGUOUS)))

    assert out == {"asset_id": ASSET, "pages": "2-4"}


# ── priority 2: no page named, viewer page is a turn fact ─────────────────────


@pytest.mark.asyncio
async def test_no_page_uses_the_viewer_current_page_without_a_model_call():
    async def _boom(**kwargs):
        raise AssertionError("no-page requests must not consult the model")

    out = await _acquire("把表格转成文本", _facts(viewer_current_page=3), _boom)

    assert out == {"asset_id": ASSET, "pages": "3"}


@pytest.mark.asyncio
async def test_model_inference_never_overrides_the_viewer_page():
    seen: dict = {}
    out = await _acquire(
        "把表格转成文本", _facts(viewer_current_page=3),
        _resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="9"), seen=seen))

    assert out == {"asset_id": ASSET, "pages": "3"}
    assert seen == {}


@pytest.mark.asyncio
async def test_a_deictic_page_is_resolved_from_the_viewer_fact():
    async def _boom(**kwargs):
        raise AssertionError("deictic pages must not depend on the model")

    out = await _acquire("把它当前这一页的表格转成文本",
                         _facts(viewer_current_page=7), _boom)

    assert out == {"asset_id": ASSET, "pages": "7"}


# ── priority 3: no target document -> no draft ────────────────────────────────


@pytest.mark.asyncio
async def test_no_target_document_produces_no_draft():
    assert await _acquire("把表格转成文本", TurnFacts()) is None
    assert await _acquire("把表格转成文本", TurnFacts(has_attachment=True)) is None


# ── priority 4: target document without a valid current page -> Agent ─────────


@pytest.mark.asyncio
async def test_a_missing_current_page_hands_the_turn_to_the_agent():
    evidence: dict = {}
    out = await PdfTableToTextHandler().acquire(
        query="把表格转成文本",
        facts=_facts(viewer_current_page=None),
        history=(), scope_evidence=evidence)

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED
    assert evidence["ok"] is False
    assert evidence["merged"]["pages"] is None
    assert out.context["viewer_asset_id"] == ASSET


@pytest.mark.asyncio
async def test_an_invalid_current_page_hands_the_turn_to_the_agent():
    for bad in (0, -3, "none", ""):
        out = await _acquire("把表格转成文本", _facts(viewer_current_page=bad))
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
    out = await _acquire("把表格转成文本", facts)

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED


# ── priority 5: model failure or contract violation -> Agent ──────────────────


@pytest.mark.asyncio
async def test_a_model_exception_hands_the_turn_to_the_agent():
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
        _resolver(_scope(status=psr.STATUS_INVALID, reason="unexpected field asset_id")))

    assert out is not None and out.terminal is False
    assert out.category == CATEGORY_SCOPE_CONTRACT


# ── the policy function's own evidence ────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_policy_records_which_producer_supplied_the_page():
    explicit = await settle_table_page_scope(
        query="提取第2页", facts=_facts(), resolver=None, asset_id=ASSET)
    assert explicit.status == SETTLED and explicit.source == SOURCE_RULE

    from_model = await settle_table_page_scope(
        query="提取第2页", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="2")),
        asset_id=ASSET)
    assert from_model.source == SOURCE_LLM

    from_viewer = await settle_table_page_scope(
        query="把表格转成文本", facts=_facts(), resolver=None, asset_id=ASSET)
    assert from_viewer.source == SOURCE_VIEWER and from_viewer.pages == "3"

    undetermined = await settle_table_page_scope(
        query="把表格转成文本", facts=_facts(viewer_current_page=None),
        resolver=None, asset_id=ASSET)
    assert undetermined.status == PAGE_UNDETERMINED


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
async def test_the_hop_certifies_the_viewer_page_into_the_binder_arguments():
    out, trace, capture = await _hop(
        "把表格转成文本", _facts(viewer_current_page=5),
        _resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="9")))

    # The Binder's final arguments carry the viewer page, never the model's "9".
    assert out.requested_action["tool"] == TOOL
    assert out.requested_action["args"] == {"asset_id": ASSET, "pages": "5"}
    assert trace["acquisition"] == "CAP_HANDLER"
    assert capture["page_scope_evidence"]["source"] == SOURCE_VIEWER


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
async def test_the_hop_hands_a_missing_viewer_page_to_the_agent():
    out, trace, capture = await _hop(
        "把表格转成文本", _facts(viewer_current_page=None))

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    record = capture["acquisition"]["failure"]
    assert record["category"] == CATEGORY_SCOPE_UNSETTLED
    assert "terminal" not in record
    assert record["context"]["viewer_current_page"] is None


@pytest.mark.asyncio
async def test_the_hop_does_not_guess_a_document_when_none_is_present():
    out, trace, capture = await _hop("把表格转成文本", TurnFacts())

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    assert "acquisition" not in capture

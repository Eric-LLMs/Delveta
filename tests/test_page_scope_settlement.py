"""Page-scope acquisition — the ONLINE MODEL is the only page producer.

Pins the approved business rules:

* the model reads the user's sentence and settles whole document / single page /
  range; no deterministic sentence parser participates;
* a missing seam, a transport fault, a contract violation and an ambiguous
  verdict are all unsettled — never ``all`` and never a guessed page;
* a conversation target is not a document page scope;
* the resolver receives the raw query, the turn facts, the history, the
  capability and the already-resolved ``asset_id`` — and owns none of them.
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.argument_acquisition.page_scope_settlement import (
    SOURCE_LLM,
    SOURCE_NONE,
    settle_page_scope,
    unsettled_scope,
)


def _facts(**over):
    base = {"viewer_current_page": 7, "viewer_name": "survey.pdf",
            "viewer_kind": "pdf", "has_viewer_selection": False}
    base.update(over)
    return types.SimpleNamespace(**base)


def _scope(**over) -> psr.PageScope:
    base = {"target": psr.TARGET_DOCUMENT, "page_scope": psr.SCOPE_ALL,
            "pages": None, "status": psr.STATUS_OK, "reason": ""}
    base.update(over)
    return psr.PageScope(**base)


def _resolver(scope=None, *, error: Exception | None = None, seen=None):
    async def _resolve(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        if error is not None:
            raise error
        return scope
    return _resolve


# ── the model's verdict is the settlement ─────────────────────────────────────


@pytest.mark.parametrize("page_scope,pages,expected", [
    (psr.SCOPE_SINGLE, "5", "5"),
    (psr.SCOPE_RANGE, "3-5", "3-5"),
    (psr.SCOPE_RANGE, "2,4,6", "2,4,6"),
])
async def test_a_stated_page_scope_is_the_settlement(page_scope, pages, expected):
    settlement = await settle_page_scope(
        query="whatever the user said", facts=_facts(),
        resolver=_resolver(_scope(page_scope=page_scope, pages=pages)))
    assert settlement.ok is True
    assert settlement.pages == expected
    assert settlement.source == SOURCE_LLM


async def test_a_whole_document_verdict_omits_the_pages_slot():
    settlement = await settle_page_scope(
        query="whatever the user said", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_ALL)))
    assert settlement.ok is True
    assert settlement.pages is None
    assert settlement.source == SOURCE_LLM


@pytest.mark.parametrize("status", [
    psr.STATUS_AMBIGUOUS,
    psr.STATUS_INVALID,
    psr.STATUS_UNAVAILABLE,
])
async def test_a_verdict_that_does_not_settle_is_not_ok(status):
    settlement = await settle_page_scope(
        query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(status=status, reason="not settled")))
    assert settlement.ok is False
    assert settlement.pages is None
    assert settlement.source == SOURCE_NONE
    assert settlement.llm["status"] == status


async def test_a_conversation_target_is_not_a_document_page_scope():
    settlement = await settle_page_scope(
        query="summarise what we discussed", facts=_facts(),
        resolver=_resolver(_scope(target=psr.TARGET_CONVERSATION)))
    assert settlement.ok is False
    assert settlement.source == SOURCE_NONE


async def test_a_document_verdict_without_a_page_scope_is_not_ok():
    settlement = await settle_page_scope(
        query="read it", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages=None)))
    assert settlement.ok is False


# ── no fallback of any kind ───────────────────────────────────────────────────


async def test_without_the_seam_the_scope_stays_unsettled():
    settlement = await settle_page_scope(
        query="读第3页", facts=_facts(), resolver=None)
    assert settlement.ok is False
    assert settlement.pages is None
    assert settlement.llm["status"] == "not_wired"


async def test_a_resolver_fault_never_falls_back_to_a_rule():
    settlement = await settle_page_scope(
        query="读第3页", facts=_facts(),
        resolver=_resolver(error=RuntimeError("gateway down")))
    assert settlement.ok is False
    assert settlement.pages is None
    assert settlement.llm["status"] == psr.STATUS_UNAVAILABLE


# ── the resolver's inputs ─────────────────────────────────────────────────────


async def test_the_model_receives_the_raw_query_and_trusted_context():
    seen: dict = {}
    await settle_page_scope(
        query="把第3到5页读一下", facts=_facts(), history=("earlier",),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="3-5"),
                           seen=seen),
        capability="cap-read-document", asset_id="asset-123")

    assert seen["query"] == "把第3到5页读一下"
    assert seen["history"] == ("earlier",)
    assert seen["capability"] == "cap-read-document"
    assert seen["asset_id"] == "asset-123"
    assert seen["facts"].viewer_current_page == 7


# ── evidence ──────────────────────────────────────────────────────────────────


async def test_the_record_names_the_producer_and_the_merged_pages():
    settlement = await settle_page_scope(
        query="read page 5", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="5")))
    record = settlement.record()
    assert record["ok"] is True
    assert record["source"] == SOURCE_LLM
    assert record["merged"] == {"pages": "5"}
    assert "rule" not in record


async def test_an_unsettled_failure_block_mirrors_the_model_verdict():
    settlement = await settle_page_scope(
        query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_AMBIGUOUS,
                                  page_scope=psr.SCOPE_UNKNOWN,
                                  reason="which pages?")))
    block = unsettled_scope(settlement)
    assert block["status"] == psr.STATUS_AMBIGUOUS
    assert block["page_scope"] == psr.SCOPE_UNKNOWN
    assert block["pages"] is None

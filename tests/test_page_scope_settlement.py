"""Combined page-scope acquisition — rule first, online LLM second, one merge.

Pins the approved business rules:

* the deterministic rule reproduces the sentence forms the file capabilities
  receive (single page / range / list / deictic / whole document / undecided);
* the online LLM is consulted AFTER the rule and may supplement or correct it;
* merge priority: rule-only -> rule, LLM-only -> LLM, conflict -> LLM wins,
  neither -> unsettled (never ``all``, never a guessed page);
* a resolver fault or a conversation-target verdict contributes nothing.
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.argument_acquisition.page_scope_rule import (
    DECISION_ALL,
    DECISION_PAGE,
    DECISION_UNKNOWN,
    rule_scope,
)
from core.application.chat.intent_funnel.argument_acquisition.page_scope_settlement import (
    SOURCE_AGREED,
    SOURCE_LLM,
    SOURCE_RULE,
    settle_page_scope,
)


def _facts(**over):
    base = {"viewer_current_page": None, "viewer_page_from": None,
            "viewer_page_to": None}
    base.update(over)
    return types.SimpleNamespace(**base)


def _scope(**over) -> psr.PageScope:
    base = {"target": psr.TARGET_DOCUMENT, "page_scope": psr.SCOPE_ALL,
            "pages": None, "status": psr.STATUS_OK, "reason": ""}
    base.update(over)
    return psr.PageScope(**base)


def _resolver(scope=None, *, error: Exception | None = None):
    async def _resolve(**kwargs):
        if error is not None:
            raise error
        return scope
    return _resolve


# ── the deterministic rule ────────────────────────────────────────────────────
@pytest.mark.parametrize("query,expected", [
    ("读一下这份文档的第1页", "1"),
    ("打开第 2 页和第 3 页", "2,3"),
    ("读第1到4页", "1-4"),
    ("把当前这一页读一下", "4"),
    ("read pages 10 through 12", "10-12"),
    ("打开第 2,4,6 页", "2,4,6"),
    ("extract page 5", "5"),
])
def test_rule_reproduces_the_stated_page_forms(query, expected):
    result = rule_scope(query, _facts(viewer_current_page=4))
    assert result.decision == DECISION_PAGE
    assert result.pages == expected


@pytest.mark.parametrize("query", ["总结一下这份文档", "summarise it"])
def test_rule_states_the_whole_document_when_no_page_is_named(query):
    result = rule_scope(query, _facts(viewer_current_page=4))
    assert result.decision == DECISION_ALL
    assert result.pages is None


@pytest.mark.parametrize("query", ["extract some pages", "读一下部分页"])
def test_rule_is_undecided_for_unspelled_page_references(query):
    assert rule_scope(query, _facts()).decision == DECISION_UNKNOWN


async def test_rule_and_llm_agree():
    settlement = await settle_page_scope(
        query="read pages 3-5", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="3-5")))
    assert settlement.ok
    assert settlement.pages == "3-5"
    assert settlement.source == SOURCE_AGREED


# ── merge priority ────────────────────────────────────────────────────────────
async def test_rule_only_keeps_the_rule_value():
    settlement = await settle_page_scope(
        query="read pages 3-5", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_UNAVAILABLE, reason="timeout")))
    assert settlement.ok
    assert settlement.pages == "3-5"
    assert settlement.source == SOURCE_RULE


async def test_llm_only_uses_the_llm_value():
    settlement = await settle_page_scope(
        query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="2")))
    assert settlement.ok
    assert settlement.pages == "2"
    assert settlement.source == SOURCE_LLM


async def test_conflicting_page_values_let_the_llm_correction_win():
    settlement = await settle_page_scope(
        query="read page 3", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="5")))
    assert settlement.ok
    assert settlement.pages == "5"
    assert settlement.source == SOURCE_LLM


async def test_whole_document_rule_can_be_narrowed_by_the_llm():
    settlement = await settle_page_scope(
        query="summarise it", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="2-3")))
    assert settlement.ok
    assert settlement.pages == "2-3"
    assert settlement.source == SOURCE_LLM


async def test_stated_pages_can_be_widened_by_the_llm():
    settlement = await settle_page_scope(
        query="read page 3", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_ALL)))
    assert settlement.ok
    assert settlement.pages is None
    assert settlement.source == SOURCE_LLM


async def test_neither_producer_settles_the_scope():
    settlement = await settle_page_scope(
        query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_AMBIGUOUS)))
    assert settlement.ok is False
    assert settlement.pages is None


async def test_without_the_seam_an_undecided_rule_stays_unsettled():
    settlement = await settle_page_scope(
        query="extract some pages", facts=_facts(), resolver=None)
    assert settlement.ok is False
    assert settlement.llm["status"] == "not_wired"


async def test_a_resolver_fault_contributes_nothing():
    settlement = await settle_page_scope(
        query="read page 3", facts=_facts(),
        resolver=_resolver(error=RuntimeError("gateway down")))
    assert settlement.ok
    assert settlement.pages == "3"
    assert settlement.source == SOURCE_RULE


async def test_a_conversation_target_is_not_a_document_page_scope():
    settlement = await settle_page_scope(
        query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(target=psr.TARGET_CONVERSATION,
                                  page_scope=psr.SCOPE_SINGLE, pages="2")))
    assert settlement.ok is False


async def test_the_llm_is_consulted_even_when_the_rule_decided():
    calls: list[dict] = []

    async def _resolver_seen(**kwargs):
        calls.append(kwargs)
        return _scope(page_scope=psr.SCOPE_SINGLE, pages="1")

    await settle_page_scope(
        query="read page 1", facts=_facts(), resolver=_resolver_seen,
        capability="cap-read-document", asset_id="asset-123")
    assert len(calls) == 1
    assert calls[0]["capability"] == "cap-read-document"
    assert calls[0]["asset_id"] == "asset-123"

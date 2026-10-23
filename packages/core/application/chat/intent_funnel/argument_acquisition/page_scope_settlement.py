"""Combined page-scope acquisition — deterministic RULE first, then the ONLINE LLM.

One responsibility: settle the reading scope of a file request (which pages) by
running BOTH producers in a fixed order and merging them under the approved
priority. It never touches asset identity, authorization, the Binder or the tool:

1. the deterministic :func:`~.page_scope_rule.rule_scope` reads the sentence and
   the trusted turn facts (never I/O, never a guess);
2. the online :func:`~.page_scope_resolver.resolve` is then ALWAYS consulted
   when the host wired it — it supplements what the rule left undecided and may
   correct a value the rule produced;
3. the two verdicts merge into ONE settlement:

   ===================  ==========================================
   rule / LLM outcome   merged result
   ===================  ==========================================
   rule only            keep the rule value
   LLM only             use the LLM value
   both, different      the LLM correction wins
   neither              unsettled — no value is invented
   ===================  ==========================================

``unsettled`` is an honest terminal outcome, never ``all`` and never a guessed
page. The caller turns it into the capability's explicit acquisition failure.

The merged ``pages`` value is already a canonical tool spec (validated by the
shared strict parser), so it can only enter the draft in a form the tool and the
Binder accept.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .page_scope_rule import (
    DECISION_ALL,
    DECISION_PAGE,
    DECISION_UNKNOWN,
    RuleScope,
    rule_scope,
)
from .page_scope_resolver import (
    SCOPE_ALL,
    SCOPE_RANGE,
    SCOPE_SINGLE,
    TARGET_DOCUMENT,
)

logger = logging.getLogger(__name__)

SOURCE_RULE = "rule"
SOURCE_LLM = "llm"
SOURCE_AGREED = "agreed"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class PageScopeSettlement:
    """The merged verdict both producers contributed to.

    ``ok`` and ``pages is None`` together mean "the whole document". ``source``
    names which producer supplied the surviving value, so a report can separate
    rule-only / LLM-only / agreed / LLM-corrected turns.
    """

    ok: bool
    pages: str | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    rule: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """JSON-safe evidence for the trace capture (never read back)."""
        return {
            "ok": self.ok,
            "source": self.source,
            "detail": self.detail,
            "rule": dict(self.rule),
            "llm": dict(self.llm),
            "merged": {"pages": self.pages},
        }


def _llm_verdict(scope) -> tuple[str, str | None]:
    """Map a resolver verdict to ``(decision, pages)`` in rule vocabulary.

    Anything that is not a document-scoped, contract-valid verdict contributes
    nothing (``unknown``): a conversation target, an ambiguous/invalid reply and
    an unavailable resolver are all "the LLM did not supply this parameter".
    """
    if scope is None or not getattr(scope, "ok", False):
        return DECISION_UNKNOWN, None
    if getattr(scope, "target", "") != TARGET_DOCUMENT:
        return DECISION_UNKNOWN, None
    if scope.page_scope == SCOPE_ALL:
        return DECISION_ALL, None
    if scope.page_scope in (SCOPE_SINGLE, SCOPE_RANGE) and scope.pages:
        return DECISION_PAGE, scope.pages
    return DECISION_UNKNOWN, None


def _llm_evidence(scope) -> dict[str, Any]:
    if scope is None:
        return {"status": "not_wired"}
    return {
        "status": getattr(scope, "status", ""),
        "target": getattr(scope, "target", ""),
        "page_scope": getattr(scope, "page_scope", ""),
        "pages": getattr(scope, "pages", None),
        "reason": getattr(scope, "reason", ""),
        "usage": dict(getattr(scope, "usage", None) or {}),
    }


def _merge(rule: RuleScope, llm_decision: str,
           llm_pages: str | None) -> tuple[bool, str | None, str, str]:
    """Apply the approved merge priority. Returns ``(ok, pages, source, detail)``."""
    if rule.decision == DECISION_PAGE:
        if llm_decision == DECISION_UNKNOWN:
            return True, rule.pages, SOURCE_RULE, "rule value kept; LLM supplied none"
        if llm_decision == DECISION_PAGE:
            if llm_pages == rule.pages:
                return True, llm_pages, SOURCE_AGREED, "rule and LLM agree"
            return (True, llm_pages, SOURCE_LLM,
                    f"LLM correction wins (rule={rule.pages!r}, llm={llm_pages!r})")
        # rule named pages, the LLM says whole document: a conflict, LLM wins.
        return True, None, SOURCE_LLM, "LLM correction wins (rule=pages, llm=all)"

    if rule.decision == DECISION_ALL:
        if llm_decision == DECISION_UNKNOWN:
            return True, None, SOURCE_RULE, "whole document kept; LLM supplied none"
        if llm_decision == DECISION_ALL:
            return True, None, SOURCE_AGREED, "rule and LLM agree (whole document)"
        return (True, llm_pages, SOURCE_LLM,
                f"LLM correction wins (rule=all, llm={llm_pages!r})")

    # rule undecided
    if llm_decision == DECISION_PAGE:
        return True, llm_pages, SOURCE_LLM, "LLM supplied the pages the rule left unstated"
    if llm_decision == DECISION_ALL:
        return True, None, SOURCE_LLM, "LLM settled the whole document"
    return False, None, SOURCE_NONE, "neither the rule nor the LLM settled the page scope"


async def settle_page_scope(
    *,
    query: str,
    facts,
    history=(),
    resolver=None,
    capability: str = "",
    asset_id: str = "",
) -> PageScopeSettlement:
    """Run rule -> online LLM -> merge for ONE file request.

    ``resolver`` is the host's page-scope seam (``ChatDeps.page_scope_resolver``).
    When it is absent the rule runs ALONE: an undecided rule is then genuinely
    unsettled (no value is invented). A resolver exception is treated exactly
    like an unavailable verdict — it never escapes into the draft.
    """
    rule = rule_scope(query, facts)
    scope = None
    if resolver is not None:
        try:
            scope = await resolver(
                query=query, facts=facts, history=history,
                capability=capability, asset_id=asset_id)
        except Exception as exc:  # noqa: BLE001 - a resolver fault is "no LLM value"
            logger.info(
                "page_scope_settlement resolver fault capability=%s error=%s",
                capability, f"{type(exc).__name__}: {exc}")
            scope = None

    llm_decision, llm_pages = _llm_verdict(scope)
    ok, pages, source, detail = _merge(rule, llm_decision, llm_pages)
    settlement = PageScopeSettlement(
        ok=ok, pages=pages, source=source, detail=detail,
        rule={"decision": rule.decision, "pages": rule.pages, "reason": rule.reason},
        llm=_llm_evidence(scope),
    )
    logger.info(
        "page_scope_settlement capability=%s ok=%s source=%s pages=%s "
        "rule=%s llm=%s/%s",
        capability, ok, source, pages, rule.decision,
        settlement.llm.get("status"), settlement.llm.get("page_scope"))
    return settlement


def unsettled_scope(settlement: PageScopeSettlement) -> dict[str, Any]:
    """The acquisition-failure ``scope`` block for an unsettled settlement.

    Mirrors the resolver verdict shape the failure record already understands
    (status / target / page_scope / pages / reason) so downstream readers need
    no new vocabulary.
    """
    llm = settlement.llm
    return {
        "status": str(llm.get("status") or "unsettled"),
        "target": str(llm.get("target") or ""),
        "page_scope": str(llm.get("page_scope") or ""),
        "pages": settlement.pages,
        "reason": settlement.detail,
    }

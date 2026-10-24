"""Page-scope policy for ``cap-pdf-table-to-text`` ONLY.

The table capability does not read a whole document: every request must resolve
to an explicit page or to the page the Viewer is showing. The business priority
is fixed:

1. the sentence spells out a page / range / list -> that page. The online model
   identifies the explicit intent and may correct the deterministic parse; it
   may NOT erase an explicit page by answering ``all``.
2. the sentence names no page, the target document IS the viewer document, and
   the viewer reports a positive current page -> that page. Model inference is
   never consulted here: the current page is a turn fact, not a model value.
3. no target document at all -> the handler returns no draft (fail-closed); the
   funnel never guesses a document.
4. a target document exists but no valid current page can be established ->
   an acquisition failure with the turn context, handed to the Agent for
   clarification. The scope is NEVER widened to the whole document.
5. the model call raises, or its reply violates the output contract -> an
   acquisition failure handed to the Agent, with no automatic fallback.

This module owns ONLY that decision. Document identity stays in ``TurnFacts``;
the Binder, authorization and the tool runtime are untouched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..argument_acquisition.page_scope_resolver import (
    SCOPE_ALL,
    SCOPE_RANGE,
    SCOPE_SINGLE,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    TARGET_DOCUMENT,
)
from ..argument_acquisition.page_scope_rule import DECISION_PAGE, rule_scope

logger = logging.getLogger(__name__)

# ── outcome statuses ──────────────────────────────────────────────────────────
SETTLED = "settled"
PAGE_UNDETERMINED = "page_undetermined"
LLM_UNAVAILABLE = "llm_unavailable"
LLM_CONTRACT_INVALID = "llm_contract_invalid"

# ── evidence sources ──────────────────────────────────────────────────────────
SOURCE_RULE = "rule"
SOURCE_LLM = "llm"
SOURCE_VIEWER = "viewer"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class TablePageScope:
    """One page-scope decision for the table capability.

    ``settled`` always carries a page spec: the table capability has no
    whole-document outcome. Every other status is an Agent hand-off.
    """

    status: str
    pages: str | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    rule: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        return self.status == SETTLED

    def record(self) -> dict[str, Any]:
        """JSON-safe evidence for the funnel trace (never read back)."""
        return {
            "ok": self.settled,
            "source": self.source,
            "detail": self.detail,
            "rule": dict(self.rule),
            "llm": dict(self.llm),
            "merged": {"pages": self.pages},
        }


def _viewer_page(facts) -> int | None:
    raw = getattr(facts, "viewer_current_page", None) if facts is not None else None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _viewer_asset(facts) -> str:
    if facts is None:
        return ""
    return str(getattr(facts, "viewer_asset_id", "") or "")


def _is_model_owned(rule) -> bool:
    """A page the sentence SPELLS OUT (``第3页`` / ``pages 2-4``).

    Deictic forms (``this page`` / ``当前页``) are resolved from the viewer fact,
    so they must never depend on a model call: the model seam cannot make the
    viewer's own page unavailable.
    """
    return rule.decision == DECISION_PAGE and rule.reason.startswith("explicit")


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


def _llm_pages(scope) -> str | None:
    """The model's page verdict, only when it identifies a document scope."""
    if scope is None or getattr(scope, "status", "") != STATUS_OK:
        return None
    if getattr(scope, "target", "") != TARGET_DOCUMENT:
        return None
    if getattr(scope, "page_scope", "") not in (SCOPE_SINGLE, SCOPE_RANGE):
        return None
    pages = getattr(scope, "pages", None)
    return str(pages) if pages else None


async def settle_table_page_scope(
    *,
    query: str,
    facts,
    history=(),
    resolver=None,
    capability: str = "",
    asset_id: str = "",
) -> TablePageScope:
    """Resolve the table request's page under the fixed priority above.

    ``asset_id`` is the already-resolved target document (attachment -> path ->
    viewer). The caller has established that it exists.
    """
    rule = rule_scope(query, facts)
    rule_evidence = {
        "decision": rule.decision,
        "pages": rule.pages,
        "reason": rule.reason,
    }
    llm_evidence: dict[str, Any] = {"status": "not_called"}

    # ── priority 1: the sentence states a page ───────────────────────────────
    if _is_model_owned(rule):
        if resolver is None:
            # Degraded host (no model seam): the deterministic rule already read
            # the explicit page, so the parameter is still settled.
            return TablePageScope(
                status=SETTLED, pages=rule.pages, source=SOURCE_RULE,
                detail="explicit page taken from the deterministic rule; "
                       "the model seam is not wired",
                rule=rule_evidence, llm=llm_evidence)
        try:
            scope = await resolver(
                query=query, facts=facts, history=history,
                capability=capability, asset_id=asset_id)
        except Exception as exc:  # noqa: BLE001 - a model fault is a hand-off
            logger.info(
                "table_page_scope model fault capability=%s error=%s",
                capability, f"{type(exc).__name__}: {exc}")
            llm_evidence = {"status": STATUS_UNAVAILABLE,
                            "reason": f"{type(exc).__name__}: {exc}"}
            return TablePageScope(
                status=LLM_UNAVAILABLE, source=SOURCE_NONE,
                detail="the page-scope model call failed; the turn goes to the Agent",
                rule=rule_evidence, llm=llm_evidence)

        llm_evidence = _llm_evidence(scope)
        status = getattr(scope, "status", "")
        if status == STATUS_UNAVAILABLE:
            return TablePageScope(
                status=LLM_UNAVAILABLE, source=SOURCE_NONE,
                detail="the page-scope model is unavailable; the turn goes to the Agent",
                rule=rule_evidence, llm=llm_evidence)
        if status == STATUS_INVALID:
            return TablePageScope(
                status=LLM_CONTRACT_INVALID, source=SOURCE_NONE,
                detail="the page-scope model violated its output contract; "
                       "the turn goes to the Agent",
                rule=rule_evidence, llm=llm_evidence)

        pages = _llm_pages(scope)
        if pages:
            return TablePageScope(
                status=SETTLED, pages=pages, source=SOURCE_LLM,
                detail="the model identified the page(s) named in the sentence",
                rule=rule_evidence, llm=llm_evidence)

        # The model supplied no usable page (ambiguous / conversation target /
        # all). An explicit page in the sentence stands: the whole document is
        # not an outcome this capability may take.
        if status == STATUS_OK and getattr(scope, "page_scope", "") == SCOPE_ALL:
            detail = ("the model reported the whole document; the explicit page "
                      "in the sentence stands")
        else:
            detail = ("the model supplied no page; the explicit page in the "
                      "sentence stands")
        return TablePageScope(
            status=SETTLED, pages=rule.pages, source=SOURCE_RULE,
            detail=detail, rule=rule_evidence, llm=llm_evidence)

    # ── priorities 2 and 4: the sentence names no page ───────────────────────
    page = _viewer_page(facts)
    viewer_is_target = bool(_viewer_asset(facts)) and _viewer_asset(facts) == asset_id
    if page is not None and viewer_is_target:
        if rule.decision == DECISION_PAGE:
            detail = "the sentence refers to the current page; the viewer's page is used"
        else:
            detail = ("no page was named; the target document is open in the "
                      "viewer, so its current page is used")
        return TablePageScope(
            status=SETTLED, pages=str(page), source=SOURCE_VIEWER,
            detail=detail, rule=rule_evidence, llm=llm_evidence)

    if rule.decision == DECISION_PAGE:
        detail = ("the sentence refers to the current page, but the target "
                  "document is not the viewer document")
    elif page is None:
        detail = ("no page was named and the target document has no valid "
                  "viewer page")
    else:
        detail = ("no page was named and the viewer's page belongs to a "
                  "different document")
    return TablePageScope(
        status=PAGE_UNDETERMINED, source=SOURCE_NONE, detail=detail,
        rule=rule_evidence, llm=llm_evidence)

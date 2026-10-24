"""Page-scope policy for ``cap-pdf-table-to-text`` ONLY.

The table capability does not read a whole document: every request must resolve
to a page the user named or to the page the Viewer is showing. The online model
is the only judge of what the user's sentence means:

1. no target document at all -> the handler returns no draft (fail-closed); the
   funnel never guesses a document and never defaults to the whole document.
2. the model reads the raw sentence and decides whether the user explicitly
   named pages. When it did -> those pages are used; the Viewer's current page
   never overrides them.
3. the model decides the user named no page -> the current page of the SAME
   target document, when the Viewer reports a valid one.
4. no valid current page (or the model judged the request is not about this
   document, or could not decide) -> an acquisition failure with the turn
   context, handed to the Agent for clarification. The scope is NEVER widened
   to the whole document.
5. the model call raises, is unavailable, or its reply violates the output
   contract -> an acquisition failure handed to the Agent, with NO
   deterministic-rule fallback.

No deterministic page parser participates in this decision: the table path
reads document identity from the turn facts, the page intent from the model,
and the current page from the Viewer.

This module owns ONLY that decision. Document identity stays in ``TurnFacts``;
the Binder, authorization and the tool runtime are untouched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..argument_acquisition.page_scope_resolver import (
    SCOPE_RANGE,
    SCOPE_SINGLE,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    TARGET_DOCUMENT,
)

logger = logging.getLogger(__name__)

# ── outcome statuses ──────────────────────────────────────────────────────────
SETTLED = "settled"
PAGE_UNDETERMINED = "page_undetermined"
LLM_UNAVAILABLE = "llm_unavailable"
LLM_CONTRACT_INVALID = "llm_contract_invalid"

# ── evidence sources ──────────────────────────────────────────────────────────
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
    # ── priority 5: no model seam -> no semantic judge -> Agent ──────────────
    if resolver is None:
        return TablePageScope(
            status=LLM_UNAVAILABLE, source=SOURCE_NONE,
            detail="the page-scope model seam is not wired; the turn goes to the Agent",
            llm={"status": "not_wired"})

    # ── priority 2: the model reads the raw sentence ─────────────────────────
    try:
        scope = await resolver(
            query=query, facts=facts, history=history,
            capability=capability, asset_id=asset_id)
    except Exception as exc:  # noqa: BLE001 - a model fault is a hand-off
        logger.info(
            "table_page_scope model fault capability=%s error=%s",
            capability, f"{type(exc).__name__}: {exc}")
        return TablePageScope(
            status=LLM_UNAVAILABLE, source=SOURCE_NONE,
            detail="the page-scope model call failed; the turn goes to the Agent",
            llm={"status": STATUS_UNAVAILABLE,
                 "reason": f"{type(exc).__name__}: {exc}"})

    llm_evidence = _llm_evidence(scope)
    status = getattr(scope, "status", "")
    if status == STATUS_UNAVAILABLE:
        return TablePageScope(
            status=LLM_UNAVAILABLE, source=SOURCE_NONE,
            detail="the page-scope model is unavailable; the turn goes to the Agent",
            llm=llm_evidence)
    if status == STATUS_INVALID:
        return TablePageScope(
            status=LLM_CONTRACT_INVALID, source=SOURCE_NONE,
            detail="the page-scope model violated its output contract; "
                   "the turn goes to the Agent",
            llm=llm_evidence)
    if status != STATUS_OK:
        # ambiguous = the model could not decide the target or the scope. That is
        # not a judgment of "no page", so the viewer page may not substitute.
        return TablePageScope(
            status=PAGE_UNDETERMINED, source=SOURCE_NONE,
            detail="the model could not determine the page scope; "
                   "the turn goes to the Agent",
            llm=llm_evidence)
    if getattr(scope, "target", "") != TARGET_DOCUMENT:
        # The model judged the request is about something other than this
        # document; a page of this document would answer a different request.
        return TablePageScope(
            status=PAGE_UNDETERMINED, source=SOURCE_NONE,
            detail="the model judged the request is not about this document; "
                   "the turn goes to the Agent",
            llm=llm_evidence)

    # ── priority 3: the model identified an explicit page / range ────────────
    pages = _llm_pages(scope)
    if pages:
        return TablePageScope(
            status=SETTLED, pages=pages, source=SOURCE_LLM,
            detail="the model identified the page(s) named in the sentence",
            llm=llm_evidence)

    # ── priority 4: the model judged that no page was named ──────────────────
    page = _viewer_page(facts)
    viewer_is_target = bool(_viewer_asset(facts)) and _viewer_asset(facts) == asset_id
    if page is not None and viewer_is_target:
        return TablePageScope(
            status=SETTLED, pages=str(page), source=SOURCE_VIEWER,
            detail=("the model judged that no page was named; the target "
                    "document is open in the viewer, so its current page is used"),
            llm=llm_evidence)

    if page is None:
        detail = ("no page was named and the target document has no valid "
                  "viewer page")
    else:
        detail = ("no page was named and the viewer's page belongs to a "
                  "different document")
    return TablePageScope(
        status=PAGE_UNDETERMINED, source=SOURCE_NONE, detail=detail,
        llm=llm_evidence)

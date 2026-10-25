"""Page-scope policy for ``cap-pdf-table-to-text`` ONLY.

One responsibility: settle the page(s) a table request reads, merging the two
approved value sources:

* the UI HANDLER default — the viewer's current page of the SAME document,
  produced deterministically by
  :func:`..argument_acquisition.handler_page_default.handler_page_default`;
* the ONLINE MODEL verdict — the only semantic judge of the sentence's page
  intent ("this page", "pages 3-5", the whole document, a corrected range).

Merge order:

==========================  =====================================================
model verdict               outcome
==========================  =====================================================
``all``                     settled, ``pages=None`` (the whole document)
``single`` / ``range``      settled, the model's canonical page spec
semantic ``unknown``        settled on the viewer's current page **only** when
(``ambiguous``)             the handler has that value; otherwise a hand-off
technical failure           Agent hand-off — never the viewer page, never a
(``unavailable``/``invalid``) successful read
==========================  =====================================================

A semantic unknown is the model REPLYING that it did not extract a page; a
timeout, an empty/non-JSON reply or a contract violation is a technical failure
and must not be disguised as "read the current page".

Document identity is not this module's business: ``asset_id`` is system-owned
and is only used to decide whether the viewer is showing the same document. The
Binder, authorization and the tool runtime are untouched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..argument_acquisition.handler_page_default import handler_page_default
from ..argument_acquisition.page_scope_resolver import (
    SCOPE_ALL,
    SCOPE_RANGE,
    SCOPE_SINGLE,
    STATUS_AMBIGUOUS,
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

    A ``settled`` outcome carries either a page spec or ``pages=None`` for the
    whole document (the tool's own "no page window" form). Every other status is
    an Agent hand-off. ``handler_pages`` is the UI default that was available
    (even when the model overrode it); ``fallback_applied`` is True only when a
    semantic unknown kept that default.
    """

    status: str
    pages: str | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    llm: dict[str, Any] = field(default_factory=dict)
    handler_pages: str | None = None
    fallback_applied: bool = False

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
            "handler_pages": self.handler_pages,
            "fallback_applied": self.fallback_applied,
        }


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


def _finish(capability: str, result: TablePageScope) -> TablePageScope:
    logger.info(
        "table_page_scope capability=%s status=%s source=%s pages=%s "
        "handler_pages=%s fallback=%s llm=%s/%s",
        capability, result.status, result.source, result.pages,
        result.handler_pages, result.fallback_applied,
        result.llm.get("status"), result.llm.get("page_scope"))
    return result


async def settle_table_page_scope(
    *,
    query: str,
    facts,
    history=(),
    resolver=None,
    capability: str = "",
    asset_id: str = "",
) -> TablePageScope:
    """Settle the table request's page under the merge order above.

    ``asset_id`` is the already-resolved target document (attachment -> path ->
    viewer). The caller has established that it exists.
    """
    handler_pages = handler_page_default(facts, asset_id=asset_id)

    # ── no model seam -> no semantic judge -> Agent ──────────────────────────
    if resolver is None:
        return _finish(capability, TablePageScope(
            status=LLM_UNAVAILABLE, source=SOURCE_NONE,
            detail="the page-scope model seam is not wired; the turn goes to the Agent",
            llm={"status": "not_wired"}, handler_pages=handler_pages))

    # ── the model reads the raw sentence ─────────────────────────────────────
    try:
        scope = await resolver(
            query=query, facts=facts, history=history,
            capability=capability, asset_id=asset_id)
    except Exception as exc:  # noqa: BLE001 - a model fault is a hand-off
        return _finish(capability, TablePageScope(
            status=LLM_UNAVAILABLE, source=SOURCE_NONE,
            detail="the page-scope model call failed; the turn goes to the Agent",
            llm={"status": STATUS_UNAVAILABLE,
                 "reason": f"{type(exc).__name__}: {exc}"},
            handler_pages=handler_pages))

    llm_evidence = _llm_evidence(scope)
    status = getattr(scope, "status", "")

    if status == STATUS_AMBIGUOUS:
        # Semantic outcome: the model replied that it did not extract a page.
        # The handler's value is then the approved page source.
        if handler_pages is None:
            return _finish(capability, TablePageScope(
                status=PAGE_UNDETERMINED, source=SOURCE_NONE,
                detail="the model did not extract a page and the handler has "
                       "no value for this document; the turn goes to the Agent",
                llm=llm_evidence))
        return _finish(capability, TablePageScope(
            status=SETTLED, pages=handler_pages, source=SOURCE_VIEWER,
            detail="the model did not extract a page; the target document is "
                   "open in the viewer, so its current page is used",
            llm=llm_evidence, handler_pages=handler_pages,
            fallback_applied=True))

    if status == STATUS_UNAVAILABLE:
        return _finish(capability, TablePageScope(
            status=LLM_UNAVAILABLE, source=SOURCE_NONE,
            detail="the page-scope model is unavailable; the turn goes to the "
                   "Agent",
            llm=llm_evidence, handler_pages=handler_pages))
    if status == STATUS_INVALID:
        return _finish(capability, TablePageScope(
            status=LLM_CONTRACT_INVALID, source=SOURCE_NONE,
            detail="the page-scope model violated its output contract; "
                   "the turn goes to the Agent",
            llm=llm_evidence, handler_pages=handler_pages))
    if status != STATUS_OK:
        return _finish(capability, TablePageScope(
            status=PAGE_UNDETERMINED, source=SOURCE_NONE,
            detail="the model could not determine the page scope; "
                   "the turn goes to the Agent",
            llm=llm_evidence, handler_pages=handler_pages))
    if getattr(scope, "target", "") != TARGET_DOCUMENT:
        # The model judged the request is about something other than this
        # document; a page of this document, or its whole text, would answer a
        # different request.
        return _finish(capability, TablePageScope(
            status=PAGE_UNDETERMINED, source=SOURCE_NONE,
            detail="the model judged the request is not about this document; "
                   "the turn goes to the Agent",
            llm=llm_evidence, handler_pages=handler_pages))

    # ── the model identified the whole document ──────────────────────────────
    if getattr(scope, "page_scope", "") == SCOPE_ALL:
        return _finish(capability, TablePageScope(
            status=SETTLED, pages=None, source=SOURCE_LLM,
            detail="the model settled the whole document",
            llm=llm_evidence, handler_pages=handler_pages))

    # ── the model identified an explicit page / range ────────────────────────
    if (getattr(scope, "page_scope", "") in (SCOPE_SINGLE, SCOPE_RANGE)
            and getattr(scope, "pages", None)):
        return _finish(capability, TablePageScope(
            status=SETTLED, pages=str(scope.pages), source=SOURCE_LLM,
            detail="the model identified the page(s) named in the sentence",
            llm=llm_evidence, handler_pages=handler_pages))

    return _finish(capability, TablePageScope(
        status=PAGE_UNDETERMINED, source=SOURCE_NONE,
        detail="the model returned a document verdict without a page scope; "
               "the turn goes to the Agent",
        llm=llm_evidence, handler_pages=handler_pages))

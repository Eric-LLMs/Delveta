"""Page-scope acquisition — the handler default plus the ONLINE model verdict.

One responsibility: settle the reading scope (which pages) of a file request.
Two values are merged under the approved acquisition rule:

* the UI HANDLER default — the viewer's current page of the SAME document,
  produced deterministically by :func:`..handler_page_default.handler_page_default`;
* the ONLINE MODEL verdict — the ONLY semantic judge of the sentence's page
  intent ("this page", "pages 3-5", the whole document, a corrected range).

Merge order:

==========================  =====================================================
model verdict               settlement
==========================  =====================================================
``all``                     ok, ``pages=None`` (whole document) — the model's
                            explicit whole-document value wins over the default
``single`` / ``range``      ok, the model's canonical page spec
semantic ``unknown``        ok **only** when the handler default exists; the
(``ambiguous``)             viewer's current page is then kept
technical failure           NOT ok — never falls back to the handler default and
(``unavailable``/``invalid``) is never reported as a successful page read
==========================  =====================================================

A semantic unknown is the model REPLYING that it did not extract a page; a
timeout, an empty/non-JSON reply or a contract violation is a technical failure
and must not be disguised as "read the current page". A model that explicitly
identifies a conversation target is not a page-scope unknown either: the handler
default is not a substitute for a different request.

Document identity is NOT this module's business: ``asset_id`` comes from
:class:`~..contract.TurnFacts` and is never sent to the model. The merged
``pages`` value is already a canonical tool spec (validated by the shared strict
parser), so it can only enter the draft in a form the tool and the Binder accept.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .handler_page_default import handler_page_default
from .page_scope_resolver import (
    SCOPE_ALL,
    SCOPE_RANGE,
    SCOPE_SINGLE,
    STATUS_AMBIGUOUS,
    STATUS_INVALID,
    STATUS_UNAVAILABLE,
    TARGET_DOCUMENT,
)

logger = logging.getLogger(__name__)

SOURCE_LLM = "llm"
SOURCE_VIEWER = "viewer"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class PageScopeSettlement:
    """The merged reading-scope verdict.

    ``ok`` and ``pages is None`` together mean "the request restricts no
    pages": each calling capability then applies its own documented default
    (the whole document, or the page in view). ``source`` names the producer
    (``llm``, ``viewer`` or ``none``), so a report can tell a settled turn from
    a failure. ``handler_pages`` is the UI default that was available (even when
    the model overrode it); ``fallback_applied`` is True only when a semantic
    unknown kept that default.
    """

    ok: bool
    pages: str | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    llm: dict[str, Any] = field(default_factory=dict)
    handler_pages: str | None = None
    fallback_applied: bool = False

    def record(self) -> dict[str, Any]:
        """JSON-safe evidence for the trace capture (never read back)."""
        return {
            "ok": self.ok,
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


def _finish(capability: str,
            settlement: PageScopeSettlement) -> PageScopeSettlement:
    logger.info(
        "page_scope_settlement capability=%s ok=%s source=%s pages=%s "
        "handler_pages=%s fallback=%s llm=%s/%s",
        capability, settlement.ok, settlement.source, settlement.pages,
        settlement.handler_pages, settlement.fallback_applied,
        settlement.llm.get("status"), settlement.llm.get("page_scope"))
    return settlement


async def settle_page_scope(
    *,
    query: str,
    facts,
    history=(),
    resolver=None,
    capability: str = "",
    asset_id: str = "",
) -> PageScopeSettlement:
    """Merge the handler's default page with the online model's verdict.

    ``resolver`` is the host's page-scope seam (``ChatDeps.page_scope_resolver``).
    The handler default is computed first, independently of the model, so it is
    also visible as evidence when the model fails.

    Only the model's SEMANTIC ``ambiguous`` verdict keeps the handler default.
    A missing seam, a resolver exception, ``unavailable`` (timeout, transport,
    empty or non-JSON reply) and ``invalid`` (contract violation) are technical
    failures: they settle nothing, never fall back and never pretend the call
    succeeded. A verdict naming a conversation target is not a page-scope
    unknown either, so it does not reuse the viewer's page.

    ``asset_id`` is the system-owned target document; it is only used to decide
    whether the viewer is showing that same document. The model never sees it.
    """
    default_pages = handler_page_default(facts, asset_id=asset_id)

    if resolver is None:
        return _finish(capability, PageScopeSettlement(
            ok=False, source=SOURCE_NONE, handler_pages=default_pages,
            detail="the page-scope model seam is not wired; the turn goes to "
                   "the failure path",
            llm={"status": "not_wired"}))

    try:
        scope = await resolver(
            query=query, facts=facts, history=history,
            capability=capability, asset_id=asset_id)
    except Exception as exc:  # noqa: BLE001 - a resolver fault is a hand-off
        return _finish(capability, PageScopeSettlement(
            ok=False, source=SOURCE_NONE, handler_pages=default_pages,
            detail="the page-scope model call failed; no fallback was applied",
            llm={"status": STATUS_UNAVAILABLE,
                 "reason": f"{type(exc).__name__}: {exc}"}))

    evidence = _llm_evidence(scope)
    status = str(evidence.get("status") or "")

    if status == STATUS_AMBIGUOUS:
        # The model replied that it did not extract a page: a semantic outcome,
        # not a failure. The handler's value is the approved fallback.
        if default_pages is None:
            return _finish(capability, PageScopeSettlement(
                ok=False, source=SOURCE_NONE,
                detail="the model did not extract a page and the handler has "
                       "no value for this document",
                llm=evidence))
        return _finish(capability, PageScopeSettlement(
            ok=True, pages=default_pages, source=SOURCE_VIEWER,
            detail="the model did not extract a page; the viewer's current "
                   "page is kept",
            llm=evidence, handler_pages=default_pages, fallback_applied=True))

    if status in (STATUS_UNAVAILABLE, STATUS_INVALID):
        # Technical failure: no reliable model result, so the turn fails rather
        # than silently reading the page in view.
        return _finish(capability, PageScopeSettlement(
            ok=False, source=SOURCE_NONE, handler_pages=default_pages,
            detail=f"the page-scope model call failed ({status}); no fallback "
                   "was applied",
            llm=evidence))

    if not getattr(scope, "ok", False):
        return _finish(capability, PageScopeSettlement(
            ok=False, source=SOURCE_NONE, handler_pages=default_pages,
            detail=("the page-scope model did not settle the scope ("
                    f"{status or 'unknown'})"),
            llm=evidence))
    if getattr(scope, "target", "") != TARGET_DOCUMENT:
        return _finish(capability, PageScopeSettlement(
            ok=False, source=SOURCE_NONE, handler_pages=default_pages,
            detail="the model judged the request is not about this document",
            llm=evidence))

    if scope.page_scope == SCOPE_ALL:
        settlement = PageScopeSettlement(
            ok=True, pages=None, source=SOURCE_LLM,
            detail="the model settled the whole document",
            llm=evidence, handler_pages=default_pages)
    elif scope.page_scope in (SCOPE_SINGLE, SCOPE_RANGE) and scope.pages:
        settlement = PageScopeSettlement(
            ok=True, pages=str(scope.pages), source=SOURCE_LLM,
            detail="the model settled the page(s) named in the sentence",
            llm=evidence, handler_pages=default_pages)
    else:
        return _finish(capability, PageScopeSettlement(
            ok=False, source=SOURCE_NONE, handler_pages=default_pages,
            detail="the model returned a document verdict without a page scope",
            llm=evidence))

    return _finish(capability, settlement)


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

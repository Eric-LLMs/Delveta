"""Page-scope acquisition — the ONLINE LLM is the only page producer.

One responsibility: settle the reading scope (which pages) of a file request
from the user's sentence. The model seam is the ONLY producer of that
parameter:

* the sentence's page semantics ("this page", "pages 3-5", the whole document,
  a range the user corrected) cannot be expressed by keyword rules, so no
  deterministic sentence parser participates;
* a missing seam, a transport fault, a contract violation and an ambiguous
  verdict are all "the model did not settle the scope" — never ``all`` and
  never a guessed page.

The model's verdict maps to exactly one outcome:

===================  ==================================================
model verdict        settlement
===================  ==================================================
``all``              ok, ``pages=None`` (the request restricts no pages;
                     the capability applies its own documented default)
``single``/``range`` ok, canonical page spec
anything else        not ok — the caller's acquisition-failure path
===================  ==================================================

Document identity is NOT this module's business: ``asset_id`` comes from
:class:`~..contract.TurnFacts` and is never sent to the model.

The merged ``pages`` value is already a canonical tool spec (validated by the
shared strict parser), so it can only enter the draft in a form the tool and the
Binder accept.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .page_scope_resolver import (
    SCOPE_ALL,
    SCOPE_RANGE,
    SCOPE_SINGLE,
    STATUS_UNAVAILABLE,
    TARGET_DOCUMENT,
)

logger = logging.getLogger(__name__)

SOURCE_LLM = "llm"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class PageScopeSettlement:
    """The model's settled verdict.

    ``ok`` and ``pages is None`` together mean "the request restricts no
    pages": each calling capability then applies its own documented default
    (the whole document, or the page in view). ``source`` names the producer
    (``llm`` or ``none``), so a report can tell a settled turn from an Agent
    hand-off.
    """

    ok: bool
    pages: str | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    llm: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """JSON-safe evidence for the trace capture (never read back)."""
        return {
            "ok": self.ok,
            "source": self.source,
            "detail": self.detail,
            "llm": dict(self.llm),
            "merged": {"pages": self.pages},
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


async def settle_page_scope(
    *,
    query: str,
    facts,
    history=(),
    resolver=None,
    capability: str = "",
    asset_id: str = "",
) -> PageScopeSettlement:
    """Ask the online model for ONE file request's reading scope.

    ``resolver`` is the host's page-scope seam (``ChatDeps.page_scope_resolver``).
    Without it there is no semantic judge, so the scope stays unsettled: no
    deterministic fallback, no guessed page, no silent whole-document read. A
    resolver exception is treated exactly like an unavailable verdict — it never
    escapes into the draft.
    """
    if resolver is None:
        return PageScopeSettlement(
            ok=False, source=SOURCE_NONE,
            detail="the page-scope model seam is not wired; the turn goes to "
                   "the failure path",
            llm={"status": "not_wired"})

    try:
        scope = await resolver(
            query=query, facts=facts, history=history,
            capability=capability, asset_id=asset_id)
    except Exception as exc:  # noqa: BLE001 - a resolver fault is a hand-off
        logger.info(
            "page_scope_settlement resolver fault capability=%s error=%s",
            capability, f"{type(exc).__name__}: {exc}")
        return PageScopeSettlement(
            ok=False, source=SOURCE_NONE,
            detail="the page-scope model call failed; no fallback was applied",
            llm={"status": STATUS_UNAVAILABLE,
                 "reason": f"{type(exc).__name__}: {exc}"})

    evidence = _llm_evidence(scope)
    if not getattr(scope, "ok", False):
        return PageScopeSettlement(
            ok=False, source=SOURCE_NONE,
            detail=("the page-scope model did not settle the scope ("
                    f"{evidence['status'] or 'unknown'})"),
            llm=evidence)
    if getattr(scope, "target", "") != TARGET_DOCUMENT:
        return PageScopeSettlement(
            ok=False, source=SOURCE_NONE,
            detail="the model judged the request is not about this document",
            llm=evidence)

    if scope.page_scope == SCOPE_ALL:
        settlement = PageScopeSettlement(
            ok=True, pages=None, source=SOURCE_LLM,
            detail="the model settled the whole document", llm=evidence)
    elif scope.page_scope in (SCOPE_SINGLE, SCOPE_RANGE) and scope.pages:
        settlement = PageScopeSettlement(
            ok=True, pages=str(scope.pages), source=SOURCE_LLM,
            detail="the model settled the page(s) named in the sentence",
            llm=evidence)
    else:
        return PageScopeSettlement(
            ok=False, source=SOURCE_NONE,
            detail="the model returned a document verdict without a page scope",
            llm=evidence)

    logger.info(
        "page_scope_settlement capability=%s ok=%s source=%s pages=%s llm=%s/%s",
        capability, settlement.ok, settlement.source, settlement.pages,
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

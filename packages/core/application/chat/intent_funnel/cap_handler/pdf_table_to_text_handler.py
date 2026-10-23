"""PdfTableToTextHandler — argument acquisition for ``cap-pdf-table-to-text``.

Owns ONLY this capability's parameter preparation; the real table detection,
rendering and vision transcription stay entirely inside the shared
``pdf_table_to_text`` tool. Both schema slots are acquired the same way as the
other file capabilities:

* ``asset_id`` — precedence ``attachment_asset_id`` -> ``path_asset_id`` ->
  ``viewer_asset_id``; no fact carries one -> ``None`` (fail-closed -> Agent).
* ``pages`` — the COMBINED lane
  (:mod:`..argument_acquisition.page_scope_settlement`): deterministic rule
  first, online LLM supplement/correction second, approved merge priority last.
  ``all`` omits the slot. A whole-document request is NEVER turned into the
  viewer's current page/range merely because a viewer page exists.

An unsettled scope returns a TERMINAL :class:`AcquisitionFailure` — never a
guess and never an Agent hand-off.

The returned draft is passed through the SAME Binder / ActionExecutor /
ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import logging

from ..argument_acquisition.page_scope_settlement import (
    settle_page_scope,
    unsettled_scope,
)
from .acquisition_failure import (
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
    turn_context,
)
from .scope import resolve_asset_id

logger = logging.getLogger(__name__)


class PdfTableToTextHandler:
    """``cap-pdf-table-to-text`` parameter handler (stateless)."""

    capability_id = "cap-pdf-table-to-text"
    accepts_page_scope = True
    accepts_scope_evidence = True

    async def acquire(self, *, query: str, facts, page_scope_resolver=None,
                      history=(), scope_evidence: dict | None = None):
        """Return the ``pdf_table_to_text`` argument draft, or ``None`` when no
        PDF is in context, or a TERMINAL failure when the page scope cannot be
        settled by the rule and the online LLM together."""
        asset_id = resolve_asset_id(facts)
        if not asset_id:
            return None
        settlement = await settle_page_scope(
            query=query, facts=facts, history=history,
            resolver=page_scope_resolver, capability=self.capability_id,
            asset_id=asset_id)
        if scope_evidence is not None:
            scope_evidence.update(settlement.record())
        if not settlement.ok:
            logger.info(
                "pdf_table_to_text page scope unsettled rule=%s llm=%s detail=%s",
                settlement.rule.get("decision"), settlement.llm.get("status"),
                settlement.detail)
            return AcquisitionFailure(
                category=CATEGORY_SCOPE_UNSETTLED,
                stage=STAGE_SCOPE_RESOLUTION,
                detail=settlement.detail,
                scope=unsettled_scope(settlement),
                context=turn_context(facts, history),
                terminal=True,
            )
        draft: dict[str, object] = {"asset_id": asset_id}
        if settlement.pages is not None:
            draft["pages"] = settlement.pages
        return draft

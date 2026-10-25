"""PdfTableToTextHandler — argument acquisition for ``cap-pdf-table-to-text``.

Owns ONLY this capability's parameter preparation; the real table detection,
rendering and vision transcription stay entirely inside the shared
``pdf_table_to_text`` tool. Both schema slots are acquired here:

* ``asset_id`` — precedence ``attachment_asset_id`` -> ``path_asset_id`` ->
  ``viewer_asset_id``; no fact carries one -> ``None`` (fail-closed: no draft,
  the turn never guesses a document).
* ``pages`` — the capability-local merge in
  :mod:`..cap_handler.table_page_scope`: the UI handler value is the viewer's
  current page of the SAME document; the online model reads the raw sentence
  and is the only semantic judge of the page intent. A valid whole document
  (the slot is then omitted), single page or range from the model overrides the
  handler value. Only a SEMANTIC "not extracted" reply (``unknown``/``ambiguous``)
  keeps the handler value; a timeout, transport error, non-JSON reply or
  contract violation is a technical failure in which no result is trusted.

Failures never fall back. A missing target document returns ``None``; a
document without a handler page value and without a model extraction, a
failed/unavailable model call or a model contract violation returns a
NON-TERMINAL :class:`AcquisitionFailure`, which the funnel hands to the Agent
with the turn context for clarification. A missing model seam is treated the
same way: no semantic judge, no settlement.

The returned draft is passed through the SAME Binder / ActionExecutor /
ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import logging

from .acquisition_failure import (
    CATEGORY_EXTRACTION_FAILED,
    CATEGORY_SCOPE_CONTRACT,
    CATEGORY_SCOPE_UNSETTLED,
    STAGE_ARGUMENT_EXTRACTION,
    STAGE_SCOPE_RESOLUTION,
    AcquisitionFailure,
    turn_context,
)
from .scope import resolve_asset_id
from .table_page_scope import (
    LLM_CONTRACT_INVALID,
    LLM_UNAVAILABLE,
    settle_table_page_scope,
)

logger = logging.getLogger(__name__)


class PdfTableToTextHandler:
    """``cap-pdf-table-to-text`` parameter handler (stateless)."""

    capability_id = "cap-pdf-table-to-text"
    accepts_page_scope = True
    accepts_scope_evidence = True

    async def acquire(self, *, query: str, facts, page_scope_resolver=None,
                      history=(), scope_evidence: dict | None = None):
        """Return the ``pdf_table_to_text`` argument draft, ``None`` when no PDF
        is in context, or a non-terminal failure the Agent must resolve."""
        asset_id = resolve_asset_id(facts)
        if not asset_id:
            # Priority 3: no target document -> no draft, no guess, no
            # whole-document default.
            return None

        scope = await settle_table_page_scope(
            query=query, facts=facts, history=history,
            resolver=page_scope_resolver, capability=self.capability_id,
            asset_id=asset_id)
        if scope_evidence is not None:
            scope_evidence.update(scope.record())

        if scope.settled:
            draft: dict[str, object] = {"asset_id": asset_id}
            if scope.pages is not None:
                draft["pages"] = scope.pages
            return draft

        logger.info(
            "pdf_table_to_text page scope undetermined status=%s llm=%s detail=%s",
            scope.status, scope.llm.get("status"), scope.detail)
        if scope.status == LLM_UNAVAILABLE:
            category, stage = CATEGORY_EXTRACTION_FAILED, STAGE_ARGUMENT_EXTRACTION
        elif scope.status == LLM_CONTRACT_INVALID:
            category, stage = CATEGORY_SCOPE_CONTRACT, STAGE_SCOPE_RESOLUTION
        else:
            category, stage = CATEGORY_SCOPE_UNSETTLED, STAGE_SCOPE_RESOLUTION
        return AcquisitionFailure(
            category=category,
            stage=stage,
            detail=scope.detail,
            scope={
                "status": str(scope.llm.get("status") or scope.status),
                "target": str(scope.llm.get("target") or ""),
                "page_scope": str(scope.llm.get("page_scope") or ""),
                "pages": scope.pages,
                "reason": scope.detail,
            },
            context=turn_context(facts, history),
            terminal=False,
        )

"""PdfExtractTextHandler — argument acquisition for ``cap-pdf-extract-text``.

Owns ONLY this capability's parameter preparation; the real PDF read and text
extraction stay entirely inside the shared ``pdf_extract_text`` tool. Both schema
slots are sourced from :class:`~..contract.TurnFacts` via :mod:`.scope` and never
from the model or the sentence:

* ``asset_id`` — precedence ``attachment_asset_id`` -> ``path_asset_id`` ->
  ``viewer_asset_id``; no fact carries one -> ``None`` (fail-closed -> Agent).
* ``pages`` — when the orchestrator wires the page-scope resolver, the ONLINE
  LLM settles the sentence's reading scope (whole document / one page / a page
  range) and this handler maps it to the tool's ``pages`` spec; ``all`` omits
  the slot. Without a resolver the legacy viewer RANGE from TurnFacts via
  :mod:`.scope` is preserved. An unsettled scope fails closed — never a guess,
  never a silent whole-document widening.

The returned draft is passed through the SAME Binder / ActionExecutor /
ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import logging

from ..argument_acquisition.page_scope_resolver import TARGET_DOCUMENT
from .scope import acquire_file_scope, resolve_asset_id

logger = logging.getLogger(__name__)


class PdfExtractTextHandler:
    """``cap-pdf-extract-text`` parameter handler (stateless)."""

    capability_id = "cap-pdf-extract-text"
    # The orchestrator may pass the page-scope resolver into ``acquire`` (the
    # online-LLM sentence-scope lane). Absent -> the legacy viewer-window draft.
    accepts_page_scope = True

    async def acquire(self, *, query: str, facts,
                      page_scope_resolver=None, history=()) -> dict[str, object] | None:
        """Return the ``pdf_extract_text`` argument draft, or ``None`` when no
        PDF is in context, the resolver cannot settle the scope, or the viewer's
        page range is incomplete (fail-closed).

        With a resolver wired, ``pages`` comes from the ONLINE LLM's page-scope
        decision (``all`` omits the slot). Without one, the legacy viewer-window
        rule is preserved for tests and undecorated deps."""
        asset_id = resolve_asset_id(facts)
        if not asset_id:
            return None
        if page_scope_resolver is None:
            return acquire_file_scope(facts)
        scope = await page_scope_resolver(
            query=query, facts=facts, history=history,
            capability=self.capability_id, asset_id=asset_id)
        if not scope.ok or scope.target != TARGET_DOCUMENT:
            # Never fall back to the viewer page, the whole document or a guess:
            # the acquisition-failure path owns an unsettled scope.
            logger.info(
                "pdf_extract_text page scope rejected status=%s target=%s reason=%s",
                scope.status, scope.target, scope.reason)
            return None
        draft: dict[str, object] = {"asset_id": asset_id}
        if scope.pages is not None:
            draft["pages"] = scope.pages
        return draft

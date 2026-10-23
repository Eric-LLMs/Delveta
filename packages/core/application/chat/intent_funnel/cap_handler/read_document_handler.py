"""ReadDocumentHandler — argument acquisition for ``cap-read-document``.

Owns ONLY this capability's parameter preparation; it never loads document bytes
and never parses a PDF/slide — the real file read and extraction stay entirely
inside the shared ``read_document`` tool. The two schema slots of
``cap-read-document``:

* ``asset_id`` — the ONLY legal source is :class:`~..contract.TurnFacts`
  (precedence ``attachment_asset_id`` -> ``path_asset_id`` -> ``viewer_asset_id``).
  There is NO fail-open path: when no fact carries an asset id the handler returns
  ``None`` and the turn exits to the Agent's existing missing/clarify handling —
  never a guessed or fabricated id.
* ``pages`` — when the orchestrator wires the page-scope resolver, the ONLINE
  LLM settles the sentence's reading scope (whole document / one page / a page
  range) and this handler maps it to the tool's ``pages`` spec; ``all`` omits
  the slot. Without a resolver (tests, undecorated deps) the legacy viewer
  RANGE from TurnFacts via :mod:`.scope` is preserved. A sentence page number is
  never parsed by this handler itself, and an unsettled scope fails closed —
  it is never guessed, never widened to the whole document.

Note: there is NO ``query`` / ``question`` slot on this capability — neither the
Registry schema nor the ``read_document`` tool declares one, so none is emitted
(an unknown slot would fail the Binder as ``BIND_INVALID``). The user's question
is not an argument here: the tool reads the document, and the turn's own LLM
composes the answer from the extracted text.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

import logging

from ..argument_acquisition.page_scope_resolver import TARGET_DOCUMENT
from .scope import acquire_file_scope, resolve_asset_id

logger = logging.getLogger(__name__)


class ReadDocumentHandler:
    """``cap-read-document`` parameter handler (stateless)."""

    capability_id = "cap-read-document"
    # The orchestrator may pass the page-scope resolver into ``acquire`` (the
    # online-LLM sentence-scope lane). Absent -> the legacy viewer-window draft.
    accepts_page_scope = True

    async def acquire(self, *, query: str, facts,
                      page_scope_resolver=None, history=()) -> dict[str, object] | None:
        """Return the ``read_document`` argument draft, or ``None`` when no
        document is in context, the resolver cannot settle the scope, or the
        viewer's page range is incomplete (fail-closed — the Agent owns the
        clarification).

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
                "read_document page scope rejected status=%s target=%s reason=%s",
                scope.status, scope.target, scope.reason)
            return None
        draft: dict[str, object] = {"asset_id": asset_id}
        if scope.pages is not None:
            draft["pages"] = scope.pages
        return draft

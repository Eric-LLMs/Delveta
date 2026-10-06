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
* ``pages`` — the viewer's declared page RANGE, sourced from TurnFacts via
  :mod:`.scope` — NEVER a page number parsed from the user's sentence (the old
  sentence regex is deliberately gone: page scope is settled upstream). The whole
  document (both bounds unset) omits the slot; a complete range becomes ``"3"`` /
  ``"3-5"``; a half-open or reversed range returns ``None`` (fail-closed — the
  Agent owns the clarification) rather than silently widening to the whole file.

Note: there is NO ``query`` / ``question`` slot on this capability — neither the
Registry schema nor the ``read_document`` tool declares one, so none is emitted
(an unknown slot would fail the Binder as ``BIND_INVALID``). The user's question
is not an argument here: the tool reads the document, and the turn's own LLM
composes the answer from the extracted text.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .scope import acquire_file_scope


class ReadDocumentHandler:
    """``cap-read-document`` parameter handler (stateless)."""

    capability_id = "cap-read-document"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``read_document`` argument draft, or ``None`` when no document
        is in context or the viewer's page range is incomplete (fail-closed — the
        Agent owns the clarification)."""
        return acquire_file_scope(facts)

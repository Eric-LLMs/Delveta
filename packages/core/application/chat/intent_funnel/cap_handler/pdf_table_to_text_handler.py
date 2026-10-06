"""PdfTableToTextHandler — argument acquisition for ``cap-pdf-table-to-text``.

Owns ONLY this capability's parameter preparation; the real table detection,
rendering and vision transcription stay entirely inside the shared
``pdf_table_to_text`` tool. Both schema slots are sourced from
:class:`~..contract.TurnFacts` via :mod:`.scope` and never from the model or the
sentence:

* ``asset_id`` — precedence ``attachment_asset_id`` -> ``path_asset_id`` ->
  ``viewer_asset_id``; no fact carries one -> ``None`` (fail-closed -> Agent).
* ``pages`` — the viewer's declared range (``"3"`` / ``"3-5"``); the whole
  document omits it; a half-open/reversed range -> ``None`` (fail-closed — never
  a silent whole-document widening).

The returned draft is passed through the SAME Binder / ActionExecutor /
ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .scope import acquire_file_scope


class PdfTableToTextHandler:
    """``cap-pdf-table-to-text`` parameter handler (stateless)."""

    capability_id = "cap-pdf-table-to-text"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``pdf_table_to_text`` argument draft, or ``None`` when no PDF
        is in context or the viewer's page range is incomplete (fail-closed)."""
        return acquire_file_scope(facts)

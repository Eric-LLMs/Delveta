"""RagSearchHandler — argument acquisition for ``cap-rag-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the RAG pipeline. The three schema slots of
``cap-rag-search``:

* ``query``  — the user's sentence, taken VERBATIM. No model call: the turn's
  message already IS the search intent, and asking a model to re-extract it only
  adds a failure surface (the formal-500 audit measured the extractor's ``query``
  pick at 9.1% — worse than copying the sentence).
* ``top_k``  — NOT emitted: the ``rag_search`` tool's own default (5) applies, so
  the value is never guessed by a model.
* ``domain`` — NOT emitted: no turn fact carries a domain id, and a plain entity
  word in the sentence must never become one. Absent source -> absent slot.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations


class RagSearchHandler:
    """``cap-rag-search`` parameter handler (stateless)."""

    capability_id = "cap-rag-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``rag_search`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — RAG takes nothing from turn context.
        """
        message = str(query or "").strip()
        if not message:
            return None
        # top_k / domain deliberately omitted (tool default / no fact source).
        return {"query": message}

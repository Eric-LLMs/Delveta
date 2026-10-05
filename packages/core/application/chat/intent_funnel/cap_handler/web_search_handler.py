"""WebSearchHandler — argument acquisition for ``cap-web-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the web-search pipeline. The two schema slots of
``cap-web-search``:

* ``query`` — the user's sentence, taken VERBATIM. No model call: the turn's
  message already IS the search intent, and asking a model to re-extract it only
  adds a failure surface (the same ruling the RAG handler records).
* ``top_k`` — NOT emitted: the ``web_search`` tool's own ``_coerce_top_k`` falls
  back to its default (5) whenever the slot is absent, so the value is never
  guessed by a model.

No ``scope`` / ``domain`` / ``engine`` slot exists in the tool schema, so none is
invented here. The returned draft (``{slot: value}``) is passed through the SAME
Binder / ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations


class WebSearchHandler:
    """``cap-web-search`` parameter handler (stateless)."""

    capability_id = "cap-web-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``web_search`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — web search takes nothing from turn
        context.
        """
        message = str(query or "").strip()
        if not message:
            return None
        # top_k deliberately omitted: web_search_tool._coerce_top_k defaults it.
        return {"query": message}

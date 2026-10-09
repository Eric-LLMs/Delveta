"""RagSearchHandler — argument acquisition for ``cap-rag-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the RAG pipeline. The three schema slots of
``cap-rag-search``:

* ``query`` — the user's sentence, taken VERBATIM by ``acquire()`` as the
  deterministic FALLBACK only. ``slot_plan()`` authorizes the unified extractor
  to return a CLEANED search TOPIC (the acquisition contract: the model owns
  natural-language understanding); a valid, constraint-passing model value
  replaces the verbatim sentence, and an empty / invalid / unavailable one
  leaves the verbatim sentence in place.
* ``top_k`` — a MODEL-owned optional slot: the result count the sentence states
  is understood by the MODEL, never by a deterministic rule. When the model
  omits it, the ``rag_search`` tool applies its own default (5).
* ``domain`` — a MODEL-owned optional slot: the model returns the domain NAME
  the user named. It is NEVER a UUID — the business layer (``rag_search`` tool)
  resolves the name to a real ``assets.domain_id``; the model never fabricates
  an id. When the sentence names no domain the model OMITS it.

No natural-language extraction rule (regex / quotes / fixed phrases) runs in
this module. The returned draft (``{slot: value}``) is passed through the SAME
Binder / ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .slot_plan import SlotPlan


class RagSearchHandler:
    """``cap-rag-search`` parameter handler (stateless)."""

    capability_id = "cap-rag-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``rag_search`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — RAG takes nothing from turn context.
        The verbatim sentence is the fallback the model may clean."""
        message = str(query or "").strip()
        if not message:
            return None
        return {"query": message}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for the natural-language slots it owns: the
        cleaned ``query`` topic, the scoping ``domain`` name and the stated
        ``top_k`` count. An absent count stays the tool's own default; an
        unnamed domain stays absent (the tool applies no scope)."""
        return SlotPlan(model_slots=("query", "domain", "top_k"),
                        default_slots=("top_k",))

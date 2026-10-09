"""RagSearchHandler — argument acquisition for ``cap-rag-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the RAG pipeline. The three schema slots of
``cap-rag-search``:

* ``query``  — the user's sentence, taken VERBATIM by ``acquire()`` as the
  deterministic FALLBACK. ``slot_plan()`` additionally authorizes the unified
  orchestrator to have the shared extractor return a CLEANED search topic; a
  valid, constraint-passing model value replaces the verbatim sentence, and an
  empty / invalid / unavailable one leaves the verbatim sentence in place.
* ``top_k``  — a RULE-obtained value: extracted by the shared DET count rule when
  the sentence states a count (``acquire()``), so the model never re-derives it;
  otherwise it is deliberately left to the ``rag_search`` tool's own default (5).
* ``domain`` — the model is authorized (only the sentence can name a scope), but
  must OMIT it when the sentence names none; it is never emitted otherwise.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .search_args import result_count
from .slot_plan import SlotPlan


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
        draft: dict[str, object] = {"query": message}
        # A stated count is a RULE-obtained value (DET), kept as-is; an absent one
        # stays the tool's default (slot_plan resolves which). `domain` has no DET
        # source (only a semantic scope read) -> the model's job (below).
        count = result_count(message)
        if count is not None:
            draft["top_k"] = count
        return draft

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for ``query`` (a cleaned topic) and ``domain``
        (only the sentence can name a scope). ``top_k`` is the DET count rule's
        value when the sentence states one (already in the draft), and otherwise
        stays the tool's default — never asked of a model. The verbatim ``query``
        from ``acquire()`` remains the fallback."""
        if "top_k" in draft:
            return SlotPlan(model_slots=("query", "domain"))
        return SlotPlan(model_slots=("query", "domain"), default_slots=("top_k",))

"""WebSearchHandler — argument acquisition for ``cap-web-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the web-search pipeline. The two schema slots of
``cap-web-search``:

* ``query`` — the user's sentence, taken VERBATIM by ``acquire()`` as the
  deterministic FALLBACK only. ``slot_plan()`` authorizes the unified extractor
  to return a CLEANED search TOPIC (the acquisition contract: the model owns
  natural-language understanding); a valid, constraint-passing model value
  replaces the verbatim sentence, and an empty / invalid / unavailable one
  leaves the verbatim sentence in place.
* ``top_k`` — a MODEL-owned optional slot: the result count the sentence states
  is understood by the MODEL (natural-language understanding), never by a
  deterministic rule. When the model omits it, the ``web_search`` tool applies
  its own default (5).

No ``scope`` / ``domain`` / ``engine`` slot exists in the tool schema, so none is
invented here. No natural-language extraction rule (regex / quotes / fixed
phrases) runs in this module. The returned draft (``{slot: value}``) is passed
through the SAME Binder / ActionExecutor / ToolRuntime handoff every other
capability uses.
"""
from __future__ import annotations

from .slot_plan import SlotPlan


class WebSearchHandler:
    """``cap-web-search`` parameter handler (stateless)."""

    capability_id = "cap-web-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``web_search`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — web search takes nothing from turn
        context. The verbatim sentence is the fallback the model may clean."""
        message = str(query or "").strip()
        if not message:
            return None
        return {"query": message}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for the natural-language slots it owns: the
        cleaned ``query`` topic and the stated ``top_k`` count. An absent count
        stays the tool's own default — the model is never forced to invent one."""
        return SlotPlan(model_slots=("query", "top_k"),
                        default_slots=("top_k",))

"""SocialSearchHandler — argument acquisition for ``cap-social-search``.

Owns ONLY this capability's parameter preparation; it never executes a tool and
never re-implements the social/search adapters. The turn's sentence is the only
value source. This handler runs NO natural-language extraction rule (no regex /
token / quoted-phrase detection): every natural-language slot is understood by
the unified extractor (the acquisition contract's MODEL lane).

The two MODEL-owned schema slots of ``cap-social-search``:

* ``query``    — the user's sentence, taken VERBATIM by ``acquire()`` as the
  deterministic FALLBACK. ``slot_plan()`` authorizes the shared extractor to
  return the search TOPIC with only the execution-instruction frame stripped;
  only a valid, constraint-passing model value replaces the verbatim sentence.
* ``platform`` — a MODEL-owned optional slot: the model returns the platform the
  user NAMED (reddit / x / zhihu) and OMITS it when the sentence names none. A
  generic request ("what are people saying about ...") therefore emits NO
  platform — the model must never invent ``auto``. An omitted platform and
  ``auto`` are equivalent downstream: ``search_social`` fans out to every
  available platform, so omission loses nothing.

``subreddit`` — a TOOL-owned optional slot (the ``search_social`` backend still
  accepts it) that is NOT model-authorized and NEVER a PE model slot. An explicit,
  user-typed ``r/<name>`` scope in the sentence is extracted DETERMINISTICALLY (a
  fixed structured token, not natural-language guessing) and handed to the tool's
  ``subreddit`` argument; when it is present the ``platform`` is pinned to
  ``reddit`` so the executor can never silently widen the search to platforms that
  cannot honour the scope. This is an EXECUTION-layer value only — it is absent
  from ``slot_plan().model_slots``, from the training target and from the PE
  scoring (the model's own ``query`` prediction is still scored independently).
``limit``    — a real ``search_social`` parameter but NOT model-owned: it is a
  tool default (5, clamped to 1..25) applied downstream, never a model output.

The deterministic scope shape and the slot narrowing live in ONE place
(:mod:`.search_scope`) — the SAME rule the dataset builder and both scorers read,
so the runtime face can never drift from the training / scoring face.

The returned draft (``{slot: value}``) is passed through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .search_scope import effective_model_slots, execution_scope_args
from .slot_plan import SlotPlan

# The static MODEL-owned slots, in render order. Equal to
# ``pe_contract.model_slots("cap-social-search")`` (the protocol test enforces the
# tie); the deterministic scope is applied on top of this list.
_STATIC_MODEL_SLOTS = ("query", "platform")


class SocialSearchHandler:
    """``cap-social-search`` parameter handler (stateless)."""

    capability_id = "cap-social-search"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``search_social`` argument draft, or ``None`` when no query
        exists (the Agent then owns the turn). ``facts`` is accepted for the
        common handler contract but unused — social search takes nothing from
        turn context. The verbatim sentence is the fallback the model may clean.

        An explicit ``r/<name>`` scope is added HERE, deterministically, into the
        tool-only ``subreddit`` arg (with ``platform`` pinned to ``reddit``) so the
        scope can never be silently widened downstream."""
        message = str(query or "").strip()
        if not message:
            return None
        draft: dict[str, object] = {"query": message}
        draft.update(execution_scope_args(self.capability_id, message))
        return draft

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for the natural-language slots it owns: the
        ``query`` topic (execution frame + result-count control stripped) and,
        unless an explicit ``r/<sub>`` scope pinned ``platform`` to reddit, the
        named ``platform``. ``subreddit`` is never a model slot; ``limit`` is the
        tool's own default (5), never a model slot. The narrowing is delegated to
        :mod:`.search_scope` — the one rule shared with the builder and scorers."""
        slots = effective_model_slots(self.capability_id, query, _STATIC_MODEL_SLOTS)
        return SlotPlan(model_slots=slots, default_slots=())

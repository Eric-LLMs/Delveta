"""TranslateHandler — argument acquisition for ``cap-translate``.

Owns the two ``cap-translate`` schema slots, ``text`` (required) and
``target_language`` (optional). ``cap-translate`` (tool binding ``translate``)
translates a payload INTO a target language; the executor defaults an absent
``target_language`` to **English** (the confirmed contract — the former
always-Chinese behavior is retired, not preserved).

Per the acquisition contract, BOTH slots are MODEL-owned natural-language slots
and this handler runs NO extraction rule (no delimiters, quotes, code-block or
colon pre-parsing — that was implementation with no requirement basis):

* ``text`` — the payload to translate; the MODEL extracts it from the sentence
  (never the instruction frame). A payload neither the model resolves honestly
  lands MISSING for the Binder gate.
* ``target_language`` — a semantic slot the model fills ONLY when the sentence
  names a target and OMITS otherwise (never invents one). The deterministic
  English default is the EXECUTOR's own constant, not a model output — so a
  model-unavailable / omitted reply simply leaves the slot absent and the
  executor renders into English.

This handler is the pure MODEL lane: ``acquire()`` returns the EMPTY ``{}`` draft
(never ``None`` unless the query is blank) and ``slot_plan()`` authorizes both
slots to the unified extractor. The returned draft goes through the SAME Binder /
ActionExecutor / ToolRuntime handoff every other capability uses.
"""
from __future__ import annotations

from .slot_plan import SlotPlan


class TranslateHandler:
    """``cap-translate`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but unused: the payload
    lives ONLY in the sentence.
    """

    capability_id = "cap-translate"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the EMPTY ``{}`` draft (never a rule-extracted payload) so the
        model is authorized to extract ``text`` / ``target_language``. Only a
        blank query returns ``None`` (no input at all)."""
        message = str(query or "")
        if not message.strip():
            return None
        return {}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize both MODEL-owned slots. ``target_language`` is filled only
        when a target is named; ``text`` is the payload the model extracts. The
        English default for an absent ``target_language`` is the executor's, not
        a model slot."""
        return SlotPlan(model_slots=("text", "target_language"))

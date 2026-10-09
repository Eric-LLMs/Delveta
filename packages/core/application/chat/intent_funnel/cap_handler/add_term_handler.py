"""AddTermHandler — argument acquisition for ``cap-add-term``.

Owns ONLY this capability's public schema slots, ``term`` and ``domain``.
``cap-add-term`` (tool binding ``add_term``) inserts ONE word into a vocabulary
domain addressed BY NAME (``VocabularyService.add_term``). The tool resolves the
domain by an EXACT case-insensitive name match over the caller's visible domains
(0 → "not found", >1 → "ambiguous"), so the ``domain`` value must be the user's
literal domain NAME — never translated, slugified or otherwise "understood".

Per the acquisition contract, BOTH slots are MODEL-owned natural-language slots
and this handler runs NO extraction rule (no quoted-span, ``把``/``将`` clause or
English insert-verb regex — that was implementation with no requirement basis):

* ``term`` — the word to add; the MODEL extracts it from the sentence verbatim
  (quotes dropped). A deictic reference with no antecedent (``这个词`` / ``this
  word``) yields no term; a required term neither the model resolves lands
  MISSING for the Binder gate.
* ``domain`` — the vocabulary domain the user named; the MODEL returns the NAME
  verbatim. The tool owns the name→entity resolution (name match / not-found /
  ambiguous preflight).

``acquire()`` returns the EMPTY ``{}`` draft (never ``None`` unless the query is
blank) and ``slot_plan()`` authorizes both slots to the unified local-Qwen slot
extractor. A model value is only folded in after ``_certify``'s SAME kind gate +
Binder validate; a required slot neither the model resolves honestly lands
MISSING (the Binder gate, then the Agent).

``definition`` is NEVER emitted (tool-owned, optional, no producer — an emitted
value would be a fabrication).

``facts`` is accepted for the common handler contract but ignored ON PURPOSE: both
values come from the sentence, never from the turn's asset context.
"""
from __future__ import annotations

from .slot_plan import SlotPlan


class AddTermHandler:
    """``cap-add-term`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but ignored ON PURPOSE:
    ``term`` and ``domain`` both come from the sentence, never from the turn's
    asset context — so no viewer/attachment id can leak in as a term or domain.
    """

    capability_id = "cap-add-term"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the EMPTY ``{}`` draft (never a rule-extracted term/domain) so
        the model is authorized to extract both slots. Only a blank query returns
        ``None`` (no input at all)."""
        message = str(query or "").strip()
        if not message:
            return None
        return {}

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model for both MODEL-owned slots, ``term`` and
        ``domain``. A slot the model cannot resolve honestly stays MISSING for the
        Binder gate. Never a fabricated ``term`` / ``domain``."""
        return SlotPlan(model_slots=("term", "domain"))

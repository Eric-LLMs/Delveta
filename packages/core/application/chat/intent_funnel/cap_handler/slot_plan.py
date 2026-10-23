"""SlotPlan — the handler's EXPLICIT per-slot disposition beyond its draft.

A capability handler's ``acquire()`` returns the slots it resolved itself
(engineering rules + trusted turn context). ``slot_plan()`` then states, per
slot, what still has to happen — WITHOUT the caller ever inferring it from
``slot not in draft``:

* a key present in ``acquire()``'s draft  → ACQUIRED (rule / context) — the
  authoritative value; a model result must never overwrite it UNLESS the key is
  explicitly named in ``SlotPlan.fallback_slots``;
* ``SlotPlan.model_slots``                → PENDING_MODEL — the handler
  authorizes ONE model extraction for exactly these slots;
* ``SlotPlan.fallback_slots``             → REPLACEABLE — a draft value that is
  only a deterministic FALLBACK (today: the search trio's verbatim sentence as
  ``query``), so the model MAY replace it;
* ``SlotPlan.default_slots``              → DEFAULTED — deliberately left to the
  tool's own legal default (e.g. an unspecified result count), so neither a rule
  nor the model produces a value;
* ``SlotPlan.required_model_slots``       → the subset of ``model_slots`` that
  MUST come back from the extractor on THIS turn. When the model is unavailable
  or returns no valid value for one of them, the acquisition hop exits to the
  Agent instead of certifying a draft that silently lost a requested argument
  (today: ``cap-summary``'s explicitly requested ``output_dir``);
* anything else (a required schema slot in none of the above) → MISSING — the
  existing Binder gate owns it (never fabricated here).

This is an OPTIONAL extension: a handler without ``slot_plan`` (every handler
except the search trio) plans nothing, so the unified orchestrator keeps the
pure-deterministic behaviour and calls no model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SlotPlan:
    """The handler's explicit request for the slots it did NOT resolve itself.

    ``model_slots`` — the ONLY slots the unified orchestrator may hand to the
    shared extractor (an unauthorized slot is never sent, so a model value can
    never leak onto a rule/context slot). ``fallback_slots`` — the subset of
    draft keys whose value is only a deterministic fallback and may therefore be
    REPLACED by a non-blank model value; every other draft key is protected.
    ``default_slots`` — the slots the handler deliberately leaves to the tool's
    legal default. ``required_model_slots`` — the model slots whose absence on
    THIS turn is an acquisition failure (never a silent omission). All default
    to empty: a handler that plans nothing keeps the pure-deterministic flow,
    and a draft value is protected unless named."""

    model_slots: tuple[str, ...] = ()
    fallback_slots: tuple[str, ...] = ()
    default_slots: tuple[str, ...] = ()
    required_model_slots: tuple[str, ...] = ()


@runtime_checkable
class SlotPlanningHandler(Protocol):
    """The OPTIONAL handler extension. Presence of ``slot_plan`` is the explicit
    opt-in to model-assisted slot acquisition; absence keeps the handler
    deterministic and model-free."""

    capability_id: str

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan: ...

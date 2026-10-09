"""SlotPlan — the handler's EXPLICIT per-slot disposition beyond its draft.

A capability handler's ``acquire()`` returns the slots it resolved itself
(engineering rules + trusted turn context). ``slot_plan()`` then states, per
slot, what still has to happen — WITHOUT the caller ever inferring it from
``slot not in draft``:

* a key present in ``acquire()``'s draft  → ACQUIRED (rule / context) — the
  authoritative value; a model result must never overwrite it;
* ``SlotPlan.model_slots``                → PENDING_MODEL — the handler
  authorizes ONE model extraction for exactly these slots;
* ``SlotPlan.default_slots``              → DEFAULTED — deliberately left to the
  tool's own legal default (e.g. an unspecified result count), so neither a rule
  nor the model produces a value;
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
    never leak onto a rule/context slot). ``default_slots`` — the slots the
    handler deliberately leaves to the tool's legal default. Both default to
    empty: a handler that plans nothing keeps the pure-deterministic flow."""

    model_slots: tuple[str, ...] = ()
    default_slots: tuple[str, ...] = ()


@runtime_checkable
class SlotPlanningHandler(Protocol):
    """The OPTIONAL handler extension. Presence of ``slot_plan`` is the explicit
    opt-in to model-assisted slot acquisition; absence keeps the handler
    deterministic and model-free."""

    capability_id: str

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan: ...

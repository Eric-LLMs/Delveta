"""Accept MODEL values into a deterministic handler draft (Phase-1 unified flow).

The handler path's half of the contract: a handler's ``acquire()`` draft is the
authoritative rule/context side, and ``slot_plan().model_slots`` names the ONLY
slots a model extraction is authorized to touch. This module folds the model's
reply into that draft under three invariants:

* **authorization** — a slot that is not in ``model_slots`` is NEVER written, so
  a model value can never leak onto a rule/context slot (the deterministic value
  is truth);
* **validation** — an empty, malformed or type-invalid model value is REJECTED,
  so the draft's own value (the verbatim fallback) stands untouched when the
  model has nothing usable;
* **no invention** — a slot absent from the Registry schema is dropped.

This is a pure, side-effect-free function: it calls no model, touches no Binder,
and the merged draft still passes through the SAME ``_certify -> Binder`` gate.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

_INT_RE = re.compile(r"^\d+$")


def _coerce(spec: Mapping, raw: object) -> object | None:
    """The accepted value for one slot, or ``None`` when the raw model value is
    empty / malformed / type-invalid (the caller then keeps the draft's value)."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if str(spec.get("type") or "string") == "integer":
        return int(text) if _INT_RE.match(text) else None
    max_len = spec.get("max_len")
    if max_len is not None and len(text) > int(max_len):
        return None
    enum = spec.get("enum")
    if enum and text not in enum:
        return None
    return text


def accept_model_values(draft: Mapping, model_values: Mapping, *,
                        model_slots, parameters) -> dict[str, object]:
    """Return ``draft`` with the authorized, valid model values folded in.

    ``model_slots`` is the handler's explicit authorization; ``parameters`` is
    the Registry schema truth for the capability. A slot the model was not asked
    for, does not exist in the schema, or whose reply is unusable leaves the
    draft exactly as it was."""
    out: dict[str, object] = dict(draft or {})
    params = dict(parameters or {})
    values = dict(model_values or {})
    for slot in model_slots or ():
        spec = params.get(slot)
        if not isinstance(spec, dict):
            continue  # never invent an unknown slot
        accepted = _coerce(spec, values.get(slot))
        if accepted is not None:
            out[slot] = accepted
    return out

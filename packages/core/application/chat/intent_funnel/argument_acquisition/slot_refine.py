"""Fold MODEL values into a deterministic handler draft (Phase-1 unified flow).

The handler path's half of the contract: a handler's ``acquire()`` draft is the
authoritative rule/context side, and ``slot_plan().model_slots`` names the ONLY
slots a model extraction is authorized to touch. When folding the model's reply
into that draft this module enforces exactly ONE thing:

* **authorization** — a slot outside ``model_slots`` is NEVER written (a model
  value can never leak onto a rule/context slot), and a slot absent from the
  Registry schema is never invented.

The model's value is written EXACTLY as produced: NO strip, NO type coercion, NO
length / enum filtering, NO silent drop. Validation is REPORTED separately
(:func:`validation_report`) — a bad value stays observable and is never silently
rewritten into a valid-looking one. The merged draft still passes through the
SAME ``_certify -> Binder`` gate.

This is a pure, side-effect-free module: it calls no model, touches no Binder.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

_INT_RE = re.compile(r"^\d+$")


def _has_value(raw: object) -> bool:
    """True when the model actually produced a value for the slot. An absent,
    ``None`` or blank (empty / whitespace-only) value is NOT a value — the draft's
    own deterministic fallback stays. This is a PRESENCE check, never a rewrite of
    a produced value."""
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    return raw != ""


def accept_model_values(draft: Mapping, model_values: Mapping, *,
                        model_slots, parameters) -> dict[str, object]:
    """Return ``draft`` with the AUTHORIZED model values folded in UNCHANGED.

    ``model_slots`` is the handler's explicit authorization; ``parameters`` is
    the Registry schema truth for the capability. A slot the model was not asked
    for, or that does not exist in the schema, is never written. A value the
    model DID produce is carried verbatim (no strip / coerce / drop); a blank or
    absent value leaves the draft's own fallback in place. Validation is a
    SEPARATE report (:func:`validation_report`), never applied by rewriting."""
    out: dict[str, object] = dict(draft or {})
    params = dict(parameters or {})
    values = dict(model_values or {})
    for slot in model_slots or ():
        if slot not in params:
            continue  # never invent an unknown slot
        if slot in values and _has_value(values[slot]):
            out[slot] = values[slot]
    return out


def validation_report(model_values: Mapping, *, model_slots,
                      parameters) -> dict[str, str]:
    """Report every model value that fails the Registry schema, WITHOUT altering
    it. Reasons: ``empty`` / ``type`` / ``max_len`` / ``enum`` (fails the schema),
    ``unauthorized`` (emitted but not in ``model_slots``) or ``unknown`` (no
    schema for the slot). An empty report means every value passed its checks."""
    params = dict(parameters or {})
    values = dict(model_values or {})
    allowed = set(map(str, model_slots or ()))
    report: dict[str, str] = {}
    for key, raw in values.items():
        slot = str(key)
        if slot not in allowed:
            report[slot] = "unauthorized"
            continue
        spec = params.get(slot)
        if not isinstance(spec, dict):
            report[slot] = "unknown"
            continue
        if raw is None or raw == "":
            report[slot] = "empty"
            continue
        if str(spec.get("type") or "string") == "integer" and not _INT_RE.match(str(raw)):
            report[slot] = "type"
            continue
        max_len = spec.get("max_len")
        if max_len is not None and len(str(raw)) > int(max_len):
            report[slot] = "max_len"
            continue
        enum = spec.get("enum")
        if enum and raw not in enum:
            report[slot] = "enum"
    return report

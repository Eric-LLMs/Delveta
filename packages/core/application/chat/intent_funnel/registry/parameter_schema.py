"""Registry parameter-schema normalization — the two-shape intake adapter.

The live ``capabilities.parameters`` column carries the canonical parameter
schema in ONE of two shapes:

* the **canonical flat** form the admin plane writes and every consumer is
  documented against (``entry.CapabilityEntry.parameters`` /
  :func:`..snapshot._parameter_issues`) —
  ``{slot: {"type": str, "description": str, "required": bool, "max_len": int?}}``;
* a raw **JSON-Schema** form seeded for a few tool-bound capabilities —
  ``{"type": "object", "required": [slot, ...], "properties": {slot: {...}}}``.

Only the flat form is defined by the Registry / Argument-Path-Router / Binder
contract, so the JSON-Schema form is normalized to it HERE, once, at the
Registry read boundary — every downstream consumer (Binder, ARP, declaration
bridge, extractor) then sees a single contract and never has to branch on the
stored shape.

The transform is pure, lossless (unknown per-slot keys such as ``enum`` are
preserved) and **idempotent** (a canonical flat input round-trips unchanged),
so it is safe to re-run on an already-normalized value (``from_rows`` /
``from_payload`` / ``dataclasses.replace``).
"""
from __future__ import annotations

from typing import Any


def _flat_spec(raw: Any) -> dict:
    """One flat slot spec: a dict, with the JSON-Schema ``maxLength`` keyword
    folded onto the contract's ``max_len`` when the latter is absent."""
    spec = dict(raw) if isinstance(raw, dict) else {}
    if "maxLength" in spec and "max_len" not in spec:
        spec["max_len"] = spec.pop("maxLength")
    return spec


def _from_json_schema(raw: Any, *, required: bool) -> dict:
    """One JSON-Schema property -> the canonical flat spec. ``required`` comes
    from the SCHEMA's top-level ``required`` list (a property is not marked
    inline in JSON-Schema), so it is passed in by the caller."""
    spec = _flat_spec(raw)
    spec["type"] = str(spec.get("type") or "string")
    spec["description"] = str(spec.get("description") or "")
    spec["required"] = bool(required)
    return spec


def normalize_parameters(raw: Any) -> dict[str, dict]:
    """Coerce a Registry ``parameters`` value to the canonical flat
    ``{slot: spec}`` shape. Empty / malformed input yields ``{}`` (the entry is
    then undeclared — the ARP routes to the Agent, unchanged).

    Detection is structural, not name-based: the JSON-Schema wrapper is the
    ONLY shape whose top-level values are not ALL per-slot dicts (its ``type``
    is a str and its ``required`` a list), so a canonical flat map — every value
    a dict — is never mistaken for a schema."""
    if not isinstance(raw, dict) or not raw:
        return {}
    properties = raw.get("properties")
    is_schema = isinstance(properties, dict) and (
        raw.get("type") == "object"
        or not all(isinstance(v, dict) for v in raw.values())
    )
    if is_schema:
        declared = raw.get("required")
        required = set(declared) if isinstance(declared, (list, tuple, set)) else set()
        return {str(slot): _from_json_schema(spec, required=str(slot) in required)
                for slot, spec in properties.items()}
    return {str(slot): _flat_spec(spec) for slot, spec in raw.items()}

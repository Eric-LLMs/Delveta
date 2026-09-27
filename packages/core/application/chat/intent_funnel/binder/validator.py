"""Active Binder validation — schema normalization and slot gates.

The mechanics :func:`..binder.validate` composes: one canonical-parameters
schema (Registry ``parameters``, 0007 ruling) checked per slot. This module
is the strategy boundary — a future richer validator (types, enums, formats)
evolves here without touching the binding flow. It extracts nothing, calls no
model, decides no intent; the executor's own runtime-side gate is the SEPARATE
legacy ``validate_action`` (see :mod:`.legacy`), and the publish gate
guarantees the two schemas agree.
"""
from __future__ import annotations

from ..contract import BIND_COMPLETE, BIND_INVALID, BIND_MISSING, BoundArguments


def validate_against_schema(schema: dict, args) -> BoundArguments:
    """Pure gate: unknown slot -> INVALID; missing/blank required slot ->
    MISSING (Agent owns the clarification); non-str coerced to str and
    stripped; over-length -> INVALID. A capability with an empty schema needs
    no arguments at all: any draft (or none, e.g. the stub backend's)
    normalizes to COMPLETE {}."""
    if not schema:
        return BoundArguments(BIND_COMPLETE, {})
    if not isinstance(args, dict):
        return BoundArguments(BIND_MISSING)
    if set(args) - set(schema):
        return BoundArguments(BIND_INVALID, args)
    out: dict[str, str] = {}
    for name, raw in schema.items():
        spec = raw if isinstance(raw, dict) else {}
        val = args.get(name)
        if val is None or (isinstance(val, str) and not val.strip()):
            if spec.get("required", True):
                return BoundArguments(BIND_MISSING)
            continue
        if not isinstance(val, str):
            val = str(val)
        val = val.strip()
        max_len = spec.get("max_len")
        if max_len is not None and len(val) > int(max_len):
            return BoundArguments(BIND_INVALID, args)
        out[name] = val
    return BoundArguments(BIND_COMPLETE, out)

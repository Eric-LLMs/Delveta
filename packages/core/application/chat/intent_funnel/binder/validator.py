"""Active Binder validation — schema normalization and slot gates.

The mechanics :func:`..binder.validate` composes: one canonical-parameters
schema (Registry ``parameters``, 0007 ruling) checked per slot. This module
is the strategy boundary — a future richer validator (types, enums, formats)
evolves here without touching the binding flow. It extracts nothing, calls no
model, decides no intent; the executor's own runtime-side gate is the SEPARATE
legacy ``validate_action`` (see :mod:`.legacy`), and the publish gate
guarantees the two schemas agree.

:func:`diagnose_against_schema` is the SAME gate with a structured report:
state plus every missing/invalid/unknown slot. It exists so an Agent fallback
can be told WHY the contract failed, without a second, drifting validator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contract import BIND_COMPLETE, BIND_INVALID, BIND_MISSING, BoundArguments


@dataclass(frozen=True)
class SchemaDiagnosis:
    """One pass of the schema gate, with the evidence kept.

    ``state`` is EXACTLY the state the gate returns (first problem in schema
    order wins, matching the historical early return). ``normalized`` is the
    coercible subset even when the state is not COMPLETE; ``missing`` and
    ``invalid`` collect the remaining evidence for reporting. ``invalid`` values
    are stable reason slugs: ``unknown_slot`` / ``max_len``.
    """

    state: str
    normalized: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    invalid: dict[str, str] = field(default_factory=dict)
    unknown: tuple[str, ...] = ()


def diagnose_against_schema(schema: dict, args) -> SchemaDiagnosis:
    """The schema gate plus its evidence. Semantics are identical to the
    historical :func:`validate_against_schema`:

    * empty schema -> COMPLETE ``{}`` (any draft is ignored);
    * non-mapping draft -> MISSING;
    * a slot outside the schema -> INVALID;
    * required slot empty -> MISSING (the FIRST such slot decides the state);
    * non-str value coerced to str and stripped;
    * over-length -> INVALID (the FIRST such slot decides the state);
    * otherwise COMPLETE with the normalized values.
    """
    if not schema:
        return SchemaDiagnosis(BIND_COMPLETE, {})
    if not isinstance(args, dict):
        return SchemaDiagnosis(BIND_MISSING)

    unknown = tuple(sorted(set(args) - set(schema)))
    if unknown:
        return SchemaDiagnosis(
            BIND_INVALID,
            invalid={slot: "unknown_slot" for slot in unknown},
            unknown=unknown,
        )

    out: dict[str, str] = {}
    missing: list[str] = []
    invalid: dict[str, str] = {}
    state: str | None = None
    for name, raw in schema.items():
        spec = raw if isinstance(raw, dict) else {}
        val = args.get(name)
        if val is None or (isinstance(val, str) and not val.strip()):
            if spec.get("required", True):
                missing.append(name)
                if state is None:
                    state = BIND_MISSING
            continue
        if not isinstance(val, str):
            val = str(val)
        val = val.strip()
        max_len = spec.get("max_len")
        if max_len is not None and len(val) > int(max_len):
            invalid[name] = "max_len"
            if state is None:
                state = BIND_INVALID
            continue
        out[name] = val

    if state is None:
        return SchemaDiagnosis(BIND_COMPLETE, out)
    return SchemaDiagnosis(
        state,
        normalized=out,
        missing=tuple(missing),
        invalid=invalid,
    )


def validate_against_schema(schema: dict, args) -> BoundArguments:
    """Pure gate: unknown slot -> INVALID; missing/blank required slot ->
    MISSING (Agent owns the clarification); non-str coerced to str and
    stripped; over-length -> INVALID. A capability with an empty schema needs
    no arguments at all: any draft (or none, e.g. the stub backend's)
    normalizes to COMPLETE {}."""
    diagnosis = diagnose_against_schema(schema, args)
    if diagnosis.state == BIND_COMPLETE:
        return BoundArguments(BIND_COMPLETE, diagnosis.normalized)
    if diagnosis.state == BIND_INVALID:
        return BoundArguments(BIND_INVALID, args)
    return BoundArguments(BIND_MISSING)

"""Node 4 — active Binder: ToolIntentModel draft + Registry schema -> BoundArguments.

Four states, never a naked None (8.7): the argument truth is a STATE, and the
non-COMPLETE states exit to the Agent (chain ruling: the recheck
hop is gone — on the active path the Binder VALIDATES ToolIntentModel's
extraction (:func:`validate`) and never extracts itself from the SENTENCE).
Context-sourced slots are the deliberate second source (E2E-matrix ruling
see :func:`_resolve_context_slots`): asset identity comes from
TurnFacts, never from the model. The Binder executes nothing (8.8: routing
metadata is all the funnel ever produces).

The validation mechanics live in :mod:`.validator` (the strategy boundary);
the legacy extract-then-validate lanes are physically separated into
:mod:`.legacy` and are reachable only through the package façade.
"""
from __future__ import annotations

from .validator import validate_against_schema

# Context-sourced slots: slot name -> the TurnFacts fields that may answer it,
# in precedence order. A Registry schema carrying one of these slot names is
# declaring "this argument is settled by the turn's context, not by the
# sentence" — the same fact the parameter description states in prose
# ("from turn facts, not the sentence"). One table, keyed by SLOT, so every
# capability with an asset-id slot shares one resolution rule (no per-tool
# special cases; a new context-sourced slot is one line here + its facts field).
_CONTEXT_SLOT_SOURCES = {
    "asset_id": ("attachment_asset_id", "path_asset_id", "viewer_asset_id"),
}


def _resolve_context_slots(args, facts, schema: dict) -> dict:
    """Deterministically source context slots from TurnFacts.

    Facts present -> the draft value is OVERWRITTEN with the settled one (a
    model-copied UUID is never trusted; a hallucinated one dies here).
    Facts absent -> any model-supplied value is STRIPPED, so the required slot
    lands MISSING and the Agent owns the clarification — the model is never
    the source of an asset identity, in either direction.
    """
    out = dict(args) if isinstance(args, dict) else {}
    for slot, fields in _CONTEXT_SLOT_SOURCES.items():
        if slot not in schema:
            continue
        truth = ""
        if facts is not None:
            for field in fields:
                truth = str(getattr(facts, field, "") or "")
                if truth:
                    break
        if truth:
            out[slot] = truth
        else:
            out.pop(slot, None)
    return out


def validate(entry, args, facts=None):
    """Normalize/validate ToolIntentModel's argument DRAFT against the
    Registry's CANONICAL parameter schema (``entry.parameters``, 0007
    ruling), with context-sourced slots resolved from the turn's settled
    FACTS before the schema gate runs. The Binder still extracts nothing from
    the sentence on this path — it is a pure gate plus a deterministic fact
    lookup; the executor's own ``validate_action`` stays the final
    runtime-side gate (legacy lane), and the publish gate guarantees the two
    schemas agree."""
    schema = dict(entry.parameters or {})
    return validate_against_schema(schema, _resolve_context_slots(args, facts, schema))

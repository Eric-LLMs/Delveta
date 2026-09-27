"""Node 4 — active Binder: ToolIntentModel draft + Registry schema -> BoundArguments.

Four states, never a naked None (8.7): the argument truth is a STATE, and the
non-COMPLETE states exit to the Agent (chain ruling 2026-09-24: the recheck
hop is gone — on the active path the Binder VALIDATES ToolIntentModel's
extraction (:func:`validate`) and never extracts itself). The Binder executes
nothing (8.8: routing metadata is all the funnel ever produces).

The validation mechanics live in :mod:`.validator` (the strategy boundary);
the legacy extract-then-validate lanes are physically separated into
:mod:`.legacy` and are reachable only through the package façade.
"""
from __future__ import annotations

from .validator import validate_against_schema


def validate(entry, args):
    """Normalize/validate ToolIntentModel's argument DRAFT against the
    Registry's CANONICAL parameter schema (``entry.parameters``, 0007
    ruling). The Binder extracts nothing on this path — it is a pure gate;
    the executor's own ``validate_action`` stays the final runtime-side gate
    (legacy lane), and the publish gate guarantees the two schemas agree."""
    schema = dict(entry.parameters or {})
    return validate_against_schema(schema, args)

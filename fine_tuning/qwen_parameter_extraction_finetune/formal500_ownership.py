"""formal500_ownership — the CORRECTED per-slot ownership resolver for Formal-500.

Both the dataset builder (``logs/_argbench/build_formal500_v2.py``) and the v2
scorer (``formal500_v2_score``) read this ONE module, so the corrected ownership
map cannot drift between generation and scoring.

Why this file exists
--------------------
``logs/_argbench/ownership.py`` (Phase-3b) is STALE: it folds the whole runtime
tool schema into one flat MODEL set and therefore marks ``top_k`` / ``limit`` /
``subreddit`` / ``count`` / ``pages`` / ``path`` as Qwen's responsibility. The
live contract moved on (commit ``935781e`` retired the C-class/DET lane; the
parameter-extraction fine-tune is now the source of truth), so that flat map no
longer describes what the extractor model is actually asked to fill.

That stale module is kept FROZEN and untouched so the legacy Formal-500
evaluator stays byte-for-byte reproducible. This module is the corrected
counterpart: it does NOT import or mutate ``ownership.py``; it re-derives
ownership from the CURRENT sources of truth.

Sources of truth (in priority order)
------------------------------------
1. ``fine_tuning/qwen_parameter_extraction_finetune/pe_contract.py`` — the frozen
   extraction contract. ``model_slots(cap)`` IS the online handler's
   ``slot_plan().model_slots`` for the six MODEL-extraction capabilities
   (the protocol test enforces the tie). Imported, never re-typed.
2. The production ``cap_handler`` modules — the handler-owned capabilities.
   A handler WITHOUT ``slot_plan`` (vision / read-document / read-file / pdf-*)
   plans nothing, so the extractor model is never asked to fill any slot on
   those capabilities: ``asset_id`` is SYSTEM_BINDER (TurnFacts), ``pages`` is
   CONTEXT (the viewer's declared range), ``question`` is HANDLER (the verbatim
   sentence), ``path`` (read-file) is HANDLER (a deterministic sentence token).
3. ``argument_acquisition/declaration.py`` — for capabilities with NO handler
   the generic chain derives ownership from the slot NAME (default MODEL), so
   ``edit_file`` slots are MODEL, ``bash.timeout`` is TOOL_DEFAULT and
   ``artifact.{project_id,run_id}`` are UNAVAILABLE. A capability whose Registry
   ``parameters`` is empty (slides / summary / mindmap) is UNDECLARED: it has
   no argument-bearing slots at all.

The bucket vocabulary is the four-way split the Phase-4 contract uses, plus the
not-yet-modeled ``HANDLER`` bucket (a deterministic handler-owned value):

  PE_MODEL         the extractor model's job — the ONLY PE extraction label.
  HANDLER          a deterministic handler value (never the model, never Qwen).
  CONTEXT          a settled-context value (viewer page range) — not the model.
  SYSTEM_BINDER    a value the Binder injects from TurnFacts (asset_id).
  TOOL_DEFAULT     a function/tool default the model must never supply.
  UNAVAILABLE      declared but with no producer on the funnel path.

Stdlib only; imports ``pe_contract`` by path so the verifier runs anywhere.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))  # fine_tuning/qwen_parameter_extraction_finetune
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pe_contract  # noqa: E402  (sibling module)

# ── ownership vocabulary ─────────────────────────────────────────────────────
PE_MODEL = "PE_MODEL"
HANDLER = "HANDLER"
CONTEXT = "CONTEXT"
SYSTEM_BINDER = "SYSTEM_BINDER"
TOOL_DEFAULT = "TOOL_DEFAULT"
UNAVAILABLE = "UNAVAILABLE"
UNDECLARED = "UNDECLARED"

# ── PER-SLOT ownership for every capability the Formal-500 touches ───────────
# For the six MODEL-extraction capabilities the PE_MODEL slots are READ FROM
# ``pe_contract`` at call time (never re-typed here). The rest is the handler /
# declaration truth spelled out per capability, each row carrying its basis.
#
# (capability_id -> {slot: bucket, ...}). A capability absent from this table is
# UNDECLARED (no argument-bearing slots).
_SLOT_BUCKETS: dict[str, dict[str, str]] = {
    # ── handler-owned, NO slot_plan → the model is never asked ────────────────
    "cap-vision": {"asset_id": SYSTEM_BINDER, "question": HANDLER},
    "cap-read-document": {"asset_id": SYSTEM_BINDER, "pages": CONTEXT},
    "cap-pdf-extract-text": {"asset_id": SYSTEM_BINDER, "pages": CONTEXT},
    "cap-pdf-table-to-text": {"asset_id": SYSTEM_BINDER, "pages": CONTEXT},
    "cap-read-file": {"path": HANDLER, "max_chars": TOOL_DEFAULT},
    # ── generic chain (declaration.py: slot-name table, default MODEL) ────────
    "cap-edit-file": {"path": PE_MODEL, "new_text": PE_MODEL, "old_text": PE_MODEL},
    "cap-bash": {"command": PE_MODEL, "timeout": TOOL_DEFAULT},
    "cap-artifact": {"action": PE_MODEL, "project_id": UNAVAILABLE, "run_id": UNAVAILABLE},
    # ── undeclared (Registry ``parameters`` empty → nothing to fill) ──────────
    "cap-slides": {},
    "cap-summary": {},
    "cap-mindmap": {},
}

# Non-model slots that the frozen Formal-500 v1 wrongly folded into the PE
# target. Kept here as the explicit REMOVAL list with its basis.
NON_MODEL_REMOVALS: dict[str, str] = {
    "top_k": "web/rag function default (pe_contract: real backend param, NOT a model slot)",
    "limit": "social function default (pe_contract: real backend param, NOT a model slot)",
    "subreddit": "social reddit-only scope (pe_contract: real backend param, NOT a model slot)",
    "count": "cap-slides is UNDECLARED (Registry parameters == {}) — no such slot exists",
}


def _contract_slots(capability_id: str) -> dict[str, str]:
    """The PE_MODEL slots for one of the six pe_contract capabilities."""
    return {slot: PE_MODEL for slot in pe_contract.model_slots(capability_id)}


def slot_buckets(capability_id: str) -> dict[str, str]:
    """{slot: bucket} for ``capability_id`` — the authoritative ownership map.

    pe_contract drives the six MODEL-extraction capabilities; ``_SLOT_BUCKETS``
    drives every handler / generic / undeclared capability.
    """
    if capability_id in pe_contract.SCHEMA:
        buckets = _contract_slots(capability_id)
        # surface the contract's NON-model params too (top_k / limit / subreddit)
        # so a v1 value for them can be classified rather than silently kept.
        for slot, spec in pe_contract.SCHEMA[capability_id]["params"].items():
            buckets.setdefault(slot, TOOL_DEFAULT)
        return buckets
    return dict(_SLOT_BUCKETS.get(capability_id, {}))


def pe_model_slots(capability_id: str) -> tuple[str, ...]:
    """The slots the extractor model is asked to fill for ``capability_id``."""
    return tuple(s for s, b in slot_buckets(capability_id).items() if b == PE_MODEL)


def classify(capability_id: str, slot: str) -> str:
    """The bucket for one (capability, slot); UNDECLARED when the capability is
    undeclared or the slot is not on its schema."""
    return slot_buckets(capability_id).get(slot, UNDECLARED)

"""Declaration bridge — Registry entry -> ``{slot: SlotDecl}`` (Phase 4).

The live Registry declares acquisition the LEGACY way: ``arg_slots`` is a
``{slot: {source: str}}`` (or ``{slot: str}``) map whose ``source`` comes from
the frozen enum in :mod:`..registry.snapshot`. The Phase-4 contract forbids
reusing that shape — a :class:`SlotDecl` carries ``ownership`` /
``allowed_sources`` / ``escalation`` and NEVER a schema fact. This module is the
ONE explicit ADAPTER between the two: it changes no DB row, no admin schema and
no public Registry API.

Ownership is a property of the SLOT NAME, not of the capability that happens to
declare it, so the primary rule is a frozen slot-name table mirroring the
validated audit in ``logs/_argbench/ownership.py`` (default ``MODEL``). A slot
that ALSO appears in the legacy ``arg_slots`` is overwritten by the legacy
source's ownership — an explicit declaration always wins over the name
heuristic.

A capability with NO ``parameters`` yields ``{}`` — the ARP's undeclared
boundary stays exactly as frozen (§G): nothing is synthesized, nothing is
reinterpreted.
"""
from __future__ import annotations

from .contract import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT,
    OWNERSHIP_UNAVAILABLE,
    SlotDecl,
)

# Frozen slot-name -> ownership table. Mirrors logs/_argbench/ownership.py:56-72;
# the ``why`` for each row is that audit's rationale. Any slot name NOT listed
# here defaults to MODEL (extracted from the sentence / settled facts).
_RULES: dict[str, str] = {
    # Binder injects asset_id from TurnFacts and OVERWRITES the draft.
    "asset_id": OWNERSHIP_SYSTEM_BINDER,
    # Tool-owned effective value when absent; the model must never supply it.
    "parent_path": OWNERSHIP_TOOL_DEFAULT,
    "definition": OWNERSHIP_TOOL_DEFAULT,
    "output_dir": OWNERSHIP_TOOL_DEFAULT,
    "timeout": OWNERSHIP_TOOL_DEFAULT,
    "max_chars": OWNERSHIP_TOOL_DEFAULT,
    # Declared on the runtime schema but with NO producer on the funnel path.
    "project_id": OWNERSHIP_UNAVAILABLE,
    "run_id": OWNERSHIP_UNAVAILABLE,
    "paths": OWNERSHIP_UNAVAILABLE,
}
_DEFAULT_OWNERSHIP = OWNERSHIP_MODEL

# Legacy ``arg_slots`` source -> the contract ownership it means. The value is
# a WORLD, not a wire detail: a source that names the user's own text is MODEL,
# a source that names settled UI/session context is SYSTEM_BINDER, and a
# tool-fixed value is TOOL_DEFAULT. ``plugin:<name>`` (the seventh legacy form,
# an extractor plugin) is MODEL — the extractor produces the value from the
# user's text.
_LEGACY_OWNERSHIP: dict[str, str] = {
    "user_input": OWNERSHIP_MODEL,
    "viewer.current_page": OWNERSHIP_SYSTEM_BINDER,
    "viewer.selection": OWNERSHIP_SYSTEM_BINDER,
    "attachment": OWNERSHIP_SYSTEM_BINDER,
    "turn_context": OWNERSHIP_SYSTEM_BINDER,
    "fixed": OWNERSHIP_TOOL_DEFAULT,
}


def _legacy_source(raw: object) -> str:
    """The source string of one legacy ``arg_slots`` entry, accepting both the
    ``{slot: {source: str}}`` and ``{slot: str}`` shapes."""
    if isinstance(raw, dict):
        return str(raw.get("source") or "").strip()
    return str(raw or "").strip()


def _ownership_from_legacy(source: str) -> str | None:
    if source in _LEGACY_OWNERSHIP:
        return _LEGACY_OWNERSHIP[source]
    if source.startswith("plugin:"):
        return OWNERSHIP_MODEL
    return None


def declaration_for(entry) -> dict[str, SlotDecl]:
    """Derive the capability's acquisition declaration from its Registry schema
    (``parameters``) and its legacy ``arg_slots``. Slots are taken from the
    CANONICAL schema only — a legacy ``arg_slots`` key absent from the schema
    declares nothing the Binder would validate, so it is ignored. An entry with
    no ``parameters`` returns ``{}`` (undeclared, §G, unchanged)."""
    parameters = dict(getattr(entry, "parameters", None) or {})
    legacy = dict(getattr(entry, "arg_slots", None) or {})
    out: dict[str, SlotDecl] = {}
    for slot in parameters:
        source = _legacy_source(legacy.get(slot))
        ownership = (_ownership_from_legacy(source)
                     or _RULES.get(slot, _DEFAULT_OWNERSHIP))
        out[slot] = SlotDecl(ownership=ownership)
    return out

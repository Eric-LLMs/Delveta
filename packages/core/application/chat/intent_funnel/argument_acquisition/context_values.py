"""System-value resolution — settled turn facts -> SYSTEM_BINDER slot values.

Gate 1 of the Argument Path Router decides readiness from the system side of a
capability's declaration: the SYSTEM_BINDER slots that are ALREADY resolved this
turn. The one truth for "which fact answers which slot" lives in the Binder
(``binder._CONTEXT_SLOT_SOURCES``) — this module REUSES that table rather than
reproducing it, so a new context-sourced slot stays a one-line Binder change
(no second copy to drift).

Only SYSTEM_BINDER slots can be resolved here. A TOOL_DEFAULT slot has no
Registry-side value (its effective default lives in the tool), so it stays
empty — a required one then honestly lands MISSING and the Agent owns the
clarification. Sources are recorded as ``UI_CONTEXT`` (the value came from the
turn's settled UI/session facts).
"""
from __future__ import annotations

from ..binder.binder import _CONTEXT_SLOT_SOURCES
from .contract import OWNERSHIP_SYSTEM_BINDER, SOURCE_UI_CONTEXT


def system_values_for(entry, declaration, *, facts):
    """Return ``(values, sources)`` for the capability's SYSTEM_BINDER slots
    that a turn FACT can answer. ``values[slot]`` is the settled value;
    ``sources[slot]`` is the contract source it was resolved from
    (``UI_CONTEXT``). A slot with no fact behind it is omitted entirely — no
    empty value is fabricated."""
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for slot, decl in (declaration or {}).items():
        if decl.ownership != OWNERSHIP_SYSTEM_BINDER:
            continue
        fields = _CONTEXT_SLOT_SOURCES.get(slot)
        if not fields or facts is None:
            continue
        for field in fields:
            truth = str(getattr(facts, field, "") or "")
            if truth:
                values[slot] = truth
                sources[slot] = SOURCE_UI_CONTEXT
                break
    return values, sources

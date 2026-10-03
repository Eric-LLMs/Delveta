"""argument_acquisition — the post-selection argument acquisition stage (§28).

Map (ruling): :mod:`.contract` holds the vocabulary (ownership /
source / strategies / SlotDecl / Readiness / provenance); :mod:`.path_router` is
the ARP (declaration + inputs -> strategy label); :mod:`.declaration` /
``evidence`` / ``context_values`` bridge Registry ``arg_slots`` and turn facts
into acquisition inputs; :mod:`.provider` composes the production per-turn
``AcquisitionInputProvider``; :mod:`.extractor` is the extraction-only MODEL-slot
adapter; :mod:`.context_bundle` builds the sanctioned bundle; :mod:`.merge`
combines MIXED system + MODEL values. The stage runs on the split lane after a
capability is selected (§25.6/§28)."""
from __future__ import annotations

from .contract import (
    OWNERSHIP_MODEL,
    OWNERSHIP_SYSTEM_BINDER,
    OWNERSHIP_TOOL_DEFAULT,
    OWNERSHIP_UNAVAILABLE,
    OWNERSHIPS,
    SOURCES,
    SOURCES_OF,
    SOURCE_CALLBACK_CONTEXT,
    SOURCE_CONVERSATION_5_USER_TURNS,
    SOURCE_DEFAULT,
    SOURCE_QUERY,
    SOURCE_RESOLVER,
    SOURCE_UI_CONTEXT,
    STRATEGIES,
    STRATEGY_CONTEXT_DIRECT,
    STRATEGY_MISSING,
    STRATEGY_MIXED,
    STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR,
    STRATEGY_QUERY_TO_EXTRACTOR,
    ArgumentProvenance,
    Readiness,
    SlotDecl,
    is_declared,
)
from .inputs import AcquisitionInputs, AcquisitionInputProvider, no_acquisition_inputs

__all__ = [
    "OWNERSHIPS", "SOURCES", "SOURCES_OF", "STRATEGIES",
    "OWNERSHIP_MODEL", "OWNERSHIP_SYSTEM_BINDER",
    "OWNERSHIP_TOOL_DEFAULT", "OWNERSHIP_UNAVAILABLE",
    "SOURCE_QUERY", "SOURCE_UI_CONTEXT", "SOURCE_CONVERSATION_5_USER_TURNS",
    "SOURCE_CALLBACK_CONTEXT", "SOURCE_RESOLVER", "SOURCE_DEFAULT",
    "STRATEGY_CONTEXT_DIRECT", "STRATEGY_QUERY_TO_EXTRACTOR",
    "STRATEGY_QUERY_PLUS_5TURNS_TO_EXTRACTOR", "STRATEGY_MIXED", "STRATEGY_MISSING",
    "SlotDecl", "Readiness", "ArgumentProvenance", "is_declared",
    "AcquisitionInputs", "AcquisitionInputProvider", "no_acquisition_inputs",
]

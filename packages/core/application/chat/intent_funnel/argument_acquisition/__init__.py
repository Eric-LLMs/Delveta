"""argument_acquisition — the Argument Path Router package (Phase 2 scaffold).

Map (ruling 2026-10-01): :mod:`.contract` (vocabulary: ownership/source/
strategies/SlotDecl/Readiness/provenance), plus ``path_router`` / ``context_bundle``
/ ``callback`` / ``merge`` added in Phase 4-6. Nothing here is wired into the
cascade yet; the contract surface ships first so Phase 4 lands on a fixed vocabulary.
"""
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
    STRATEGY_QUERY_PLUS_5_USER_TURNS,
    STRATEGY_QUERY_TO_QWEN,
    ArgumentProvenance,
    Readiness,
    SlotDecl,
    is_declared,
)

__all__ = [
    "OWNERSHIPS", "SOURCES", "SOURCES_OF", "STRATEGIES",
    "OWNERSHIP_MODEL", "OWNERSHIP_SYSTEM_BINDER",
    "OWNERSHIP_TOOL_DEFAULT", "OWNERSHIP_UNAVAILABLE",
    "SOURCE_QUERY", "SOURCE_UI_CONTEXT", "SOURCE_CONVERSATION_5_USER_TURNS",
    "SOURCE_CALLBACK_CONTEXT", "SOURCE_RESOLVER", "SOURCE_DEFAULT",
    "STRATEGY_CONTEXT_DIRECT", "STRATEGY_QUERY_TO_QWEN",
    "STRATEGY_QUERY_PLUS_5_USER_TURNS", "STRATEGY_MIXED", "STRATEGY_MISSING",
    "SlotDecl", "Readiness", "ArgumentProvenance", "is_declared",
]

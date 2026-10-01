"""TRANSITION SHIM — REMOVE IN PHASE 4.

Phase 3 wires ONLY the MISS/AMBIGUOUS lane of the new selection node. In the
final architecture cap_router's output (:class:`..cap_router.CapabilityRoute`)
feeds the **Argument Path Router**, which decides HOW each argument is acquired
(CONTEXT_DIRECT / QUERY_TO_QWEN / QUERY_PLUS_5_USER_TURNS / MIXED / MISSING)
before any Binder runs. That node does not exist yet.

So Phase 3 adapts the route back into the LEGACY ``ToolIntentVerdict`` shape so
the existing Binder/downstream can consume it unchanged. This module is
**TEMPORARY compatibility only** — it is NOT the final architecture. Phase 4
introduces the Argument Path Router and **this module MUST be deleted**; a
``CapabilityRoute`` must never again be laundered into a ToolIntentModel verdict.

Phase 3 never extracts arguments here: ``arguments=None`` is carried through, so
a schema'd capability exits ``BIND_MISSING`` (the deliberate limit of the phase,
NOT a cap_router failure).
"""
from __future__ import annotations

from .cap_router import CapabilityRoute
from .contract import (
    TOOL_INTENT_CONFIDENT,
    TOOL_INTENT_REJECT,
    ToolIntentVerdict,
)


def route_to_verdict(route: CapabilityRoute) -> ToolIntentVerdict:
    """TRANSITION SHIM — REMOVE IN PHASE 4.

    A SELECTED route becomes a CONFIDENT verdict with NO arguments; a NONE route
    becomes a REJECT. The caller reads ``route.selected`` for the cap_router-
    specific fallback reason and only the SELECTED shape reaches the Binder.
    """
    if route.selected:
        return ToolIntentVerdict(
            TOOL_INTENT_CONFIDENT, route.capability_id,
            arguments=None, confidence=route.confidence)
    return ToolIntentVerdict(TOOL_INTENT_REJECT, None, arguments=None)

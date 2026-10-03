"""cap_router — capability selection node (cap_router service | stub | off).

Implementation map: :mod:`.base` (backend-independent contract). The selection
orchestration and the concrete backends are added in Phase 3/4; this package
holds the contract surface only (Phase 2 scaffold, no runtime wiring yet).
"""
from __future__ import annotations

from .backends import selector_for
from .base import (
    BACKEND_CAP_ROUTER,
    BACKEND_OFF,
    BACKEND_STUB,
    BACKENDS,
    ROUTE_NONE,
    ROUTE_REJECT,
    ROUTE_SELECTED,
    CapabilityRoute,
    CapabilityRouterUnavailable,
    CapabilitySelector,
)

__all__ = [
    "BACKENDS", "BACKEND_OFF", "BACKEND_STUB", "BACKEND_CAP_ROUTER",
    "ROUTE_SELECTED", "ROUTE_NONE", "ROUTE_REJECT",
    "CapabilityRoute", "CapabilityRouterUnavailable", "CapabilitySelector",
    "selector_for",
]

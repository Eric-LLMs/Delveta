"""cap_router — capability selection node (LayaChoice | stub | off).

Implementation map: :mod:`.base` (backend-independent contract). The selection
orchestration and the concrete backends are added in Phase 3/4; this package
holds the contract surface only (Phase 2 scaffold, no runtime wiring yet).
"""
from __future__ import annotations

from .backends import CapabilitySelector
from .base import (
    BACKEND_LAYA,
    BACKEND_OFF,
    BACKEND_STUB,
    BACKENDS,
    ROUTE_NONE,
    ROUTE_SELECTED,
    CapabilityRoute,
    CapabilityRouterUnavailable,
)

__all__ = [
    "BACKENDS", "BACKEND_OFF", "BACKEND_STUB", "BACKEND_LAYA",
    "ROUTE_SELECTED", "ROUTE_NONE",
    "CapabilityRoute", "CapabilityRouterUnavailable", "CapabilitySelector",
]

"""cap_router — capability selection node (cap_router service | stub | off).

Implementation map: :mod:`.base` holds the backend-independent contract
(``CapabilitySelector`` + ``CapabilityRoute`` with SELECTED / NONE / REJECT);
:mod:`.card_renderer` renders the 4-slot payload (3 capability cards + the
frozen REJECT card); :mod:`.backends` resolves the configured backend name to a
concrete selector via :func:`selector_for`. The node is wired into the funnel
orchestrator on the split lane (§25.6/§26)."""
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

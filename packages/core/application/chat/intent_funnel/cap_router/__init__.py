"""cap_router — capability selection node (the deployed cap_router service).

Implementation map: :mod:`.base` holds the backend-independent contract
(``CapabilitySelector`` + ``CapabilityRoute`` with SELECTED / NONE / REJECT);
:mod:`.card_renderer` renders the variable-length card payload (K capability
cards + the frozen REJECT card); :mod:`.backends` resolves the ONE production
selector via :func:`default_selector` (``StubSelector`` is a test double,
injected through ``ChatDeps.cap_router_selector``). The node is wired into the
funnel orchestrator for every non-HIT turn."""
from __future__ import annotations

from .backends import default_selector
from .base import (
    ROUTE_NONE,
    ROUTE_REJECT,
    ROUTE_SELECTED,
    CapabilityRoute,
    CapabilityRouterUnavailable,
    CapabilitySelector,
)

__all__ = [
    "ROUTE_SELECTED", "ROUTE_NONE", "ROUTE_REJECT",
    "CapabilityRoute", "CapabilityRouterUnavailable", "CapabilitySelector",
    "default_selector",
]

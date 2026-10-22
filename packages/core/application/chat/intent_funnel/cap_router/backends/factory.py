"""cap_router production resolution — the ONE selector the funnel uses.

There is no backend ladder and no configuration switch: every production
MISS/AMBIGUOUS turn is adjudicated by the deployed ``cap_router`` service.
``StubSelector`` is a TEST DOUBLE only; a test injects it through
``ChatDeps.cap_router_selector`` and the cascade prefers that injection.

Service availability is NOT resolved here: whether the endpoint is deployed is
discovered by :meth:`CapRouterSelector.select` (raising
``CapabilityRouterUnavailable``), which the caller maps to the Agent.
"""
from __future__ import annotations

from ..base import CapabilitySelector


def default_selector() -> CapabilitySelector:
    """The production selector. Constructed unconditionally — a missing endpoint
    surfaces as ``CapabilityRouterUnavailable`` from ``select``, never here."""
    from .service import CapRouterSelector

    return CapRouterSelector()

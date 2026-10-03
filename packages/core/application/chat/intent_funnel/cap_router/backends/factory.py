"""cap_router backend RESOLUTION — configured name -> selector, or None.

Backend resolution is a distinct responsibility from any backend itself: each
backend module (:mod:`.stub`, :mod:`.service`) defines its selector; THIS module
owns the ONE place that maps a configured backend name to it. The caller maps
``None`` to the Agent (``CAP_ROUTER_UNAVAILABLE``) — selection never falls
through to another model (ruling 2026-10-01).
"""
from __future__ import annotations

from ..base import BACKEND_CAP_ROUTER, BACKEND_STUB, CapabilitySelector


def selector_for(backend: str) -> CapabilitySelector | None:
    """Resolve a configured backend name to its selector, or None when the
    backend cannot serve (``off`` / unknown). The caller maps None to the Agent
    via ``CAP_ROUTER_UNAVAILABLE`` — selection never falls through to another
    model (ruling 2026-10-01).

    ``cap_router`` ALWAYS resolves to a :class:`CapRouterSelector` — backend
    resolution and service availability are separate concerns: whether the
    service is deployed is discovered by :meth:`CapRouterSelector.select`
    (raising ``CapabilityRouterUnavailable``), never by returning None here."""
    if backend == BACKEND_STUB:
        from .stub import StubSelector

        return StubSelector()
    if backend == BACKEND_CAP_ROUTER:
        from .service import CapRouterSelector

        return CapRouterSelector()
    return None

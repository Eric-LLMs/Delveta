"""cap_router backend RESOLUTION — configured name -> selector, or None.

Backend resolution is a distinct responsibility from any backend itself: a
backend module (:mod:`.stub`, and later ``laya``) defines its selector; THIS
module owns the ONE place that maps a configured backend name to it. The caller
maps ``None`` to the Agent (``CAP_ROUTER_UNAVAILABLE``) — selection never falls
through to another model (ruling 2026-10-01).
"""
from __future__ import annotations

from ..base import BACKEND_STUB, CapabilitySelector


def selector_for(backend: str) -> CapabilitySelector | None:
    """Resolve a configured backend name to its selector, or None when the
    backend cannot serve (``off`` / not-yet-deployed ``laya`` / unknown). The
    caller maps None to the Agent via ``CAP_ROUTER_UNAVAILABLE`` — selection
    never falls through to another model (ruling 2026-10-01). Phase 3 ships the
    deterministic ``stub`` only; ``laya`` is added behind this same factory."""
    if backend == BACKEND_STUB:
        from .stub import StubSelector

        return StubSelector()
    return None

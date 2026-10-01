"""cap_router backends — the selector INTERFACE (Phase 2 scaffold).

The concrete backends land in Phase 3: ``stub`` (deterministic) and ``laya``
(the real LayaChoice model). This module defines only the seam every backend
honors, so Phase 3 fills in implementations without touching the contract.
Nothing here is wired into the cascade yet.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..base import CapabilityRoute


@runtime_checkable
class CapabilitySelector(Protocol):
    """One backend's selection call: given the user query and the
    capability-level candidate cards, return the ONE capability (or NONE).

    A backend that cannot serve raises :class:`..base.CapabilityRouterUnavailable`
    — the caller maps that to the Agent; selection never falls through to another
    model (ruling 2026-10-01)."""

    async def select(self, query: str, candidates: Sequence, *,
                     entries_by_id: dict, facts=None) -> CapabilityRoute: ...

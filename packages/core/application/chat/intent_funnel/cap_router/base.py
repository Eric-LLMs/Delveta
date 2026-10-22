"""cap_router — the capability-SELECTION node contract (backend-independent seam).

Phase 2: capability selection is split out of the former
single-call ``tool_intent.select_and_extract`` into its own node. ``cap_router``
answers exactly ONE question — *which* capability the turn dispatches — and
nothing else: it never authors arguments, never touches the Binder, never
resolves context, never calls Qwen. Argument acquisition is a separate node
(:mod:`..argument_acquisition`).

There is exactly ONE production selector: the deployed cap_router service
(current model impl: LayaChoice). ``StubSelector`` exists ONLY as a test double
and is injected through ``ChatDeps.cap_router_selector``; it is never selected
by configuration in production.

Failure contract (ruling): a ``cap_router`` failure — timeout, service
unavailable, malformed output, or an off-candidate capability id — exits to the
Agent. Selection NEVER falls back to the argument extractor or to a second
selection model; otherwise the cap_router metrics would be polluted by the very
model under test.

The backend seam is :class:`CapabilitySelector`, defined here as the node's
abstract contract. The production resolution (``default_selector``) and the
concrete selectors live in :mod:`.backends`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

ROUTE_SELECTED = "SELECTED"
ROUTE_NONE = "NONE"
# REJECT is a NORMAL decision, never a threshold or a fallback placeholder. It
# routes to the system's REAL no-capability path (Agent), not to a fake tool /
# the argument chain. The option list is variable-length — REJECT is appended
# after however many capability cards the business layer handed over.
ROUTE_REJECT = "REJECT"


@dataclass(frozen=True)
class CapabilityRoute:
    """cap_router's single output: the ONE capability the turn dispatches, NONE,
    or REJECT. ``provenance`` is a candidate-side note (which card won and why)
    for telemetry only — it carries no execution authority."""

    decision: str = ROUTE_NONE
    capability_id: str | None = None
    confidence: float | None = None
    provenance: str = field(default="", repr=False)

    @property
    def selected(self) -> bool:
        return self.decision == ROUTE_SELECTED and self.capability_id is not None

    @property
    def rejected(self) -> bool:
        return self.decision == ROUTE_REJECT


class CapabilityRouterUnavailable(Exception):
    """The configured cap_router backend cannot serve (not deployed / transport
    down / malformed output). The CALLER maps this to the Agent — selection
    never falls through to another model (ruling)."""


@runtime_checkable
class CapabilitySelector(Protocol):
    """The backend seam: one selection call — given the user query and the
    capability-level candidate cards, return the ONE capability (or NONE).

    A backend that cannot serve raises :class:`CapabilityRouterUnavailable`
    — the caller maps that to the Agent; selection never falls through to
    another model (ruling)."""

    async def select(self, query: str, candidates: Sequence, *,
                     entries_by_id: dict, facts=None) -> CapabilityRoute: ...

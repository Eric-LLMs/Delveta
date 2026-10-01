"""cap_router — the capability-SELECTION node contract (backend-independent seam).

Phase 2 (2026-10-01): capability selection is split out of the former
single-call ``tool_intent.select_and_extract`` into its own node. ``cap_router``
answers exactly ONE question — *which* capability the turn dispatches — and
nothing else: it never authors arguments, never touches the Binder, never
resolves context, never calls Qwen. Argument acquisition is a separate node
(:mod:`..argument_acquisition`).

Backend ladder (``settings.chat_cap_router_backend``):
  ``off``   — the new split chain is NOT used; the legacy single-call hop stays
              byte-identical (rollback / compatibility lane);
  ``stub``  — deterministic selector (wiring/E2E tests only, no model);
  ``laya``  — the real LayaChoice decision model.

Failure contract (ruling 2026-10-01): a ``laya`` failure — timeout, service
unavailable, malformed output, or an off-candidate capability id — exits to the
Agent. Selection NEVER falls back to Qwen; otherwise the Laya E2E metrics would
be polluted by the very model under test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BACKEND_OFF = "off"
BACKEND_STUB = "stub"
BACKEND_LAYA = "laya"
BACKENDS = (BACKEND_OFF, BACKEND_STUB, BACKEND_LAYA)

ROUTE_SELECTED = "SELECTED"
ROUTE_NONE = "NONE"


@dataclass(frozen=True)
class CapabilityRoute:
    """cap_router's single output: the ONE capability the turn dispatches, or
    NONE. ``provenance`` is a candidate-side note (which card won and why) for
    telemetry only — it carries no execution authority."""

    decision: str = ROUTE_NONE
    capability_id: str | None = None
    confidence: float | None = None
    provenance: str = field(default="", repr=False)

    @property
    def selected(self) -> bool:
        return self.decision == ROUTE_SELECTED and self.capability_id is not None


class CapabilityRouterUnavailable(Exception):
    """The configured cap_router backend cannot serve (not deployed / transport
    down / malformed output). The CALLER maps this to the Agent — selection
    never falls through to another model (ruling 2026-10-01)."""

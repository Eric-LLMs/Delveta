"""cap_router backend: deterministic capability selection (Phase 3 wiring).

The transition stub of the cap_router lane. It inherits the leader-vs-runner-up
margin discipline the cosine corpus proved (the SAME rule the legacy
ToolIntentModel stub applied), but answers exactly ONE question — *which*
capability (or NONE). It authors nothing else: no arguments, no Binder contact,
no context resolution, no model call.

Phase 3 scope: this stub exists to verify the
``MISS/AMBIGUOUS -> Recall -> Aggregation -> cap_router -> ONE|NONE`` wiring
with ``backend=stub``. The real LayaChoice backend lands later and replaces it
behind the SAME :class:`..CapabilitySelector` protocol, without touching the
contract or the cascade.

Margins: a single candidate with a trustworthy provenance (a calibrated cosine
``recall`` or a deterministic ``matcher_hit``) selects; a race selects only when
BOTH leaders are trustworthy AND the gap clears ``chat_funnel_margin``;
matcher-AMBIGUOUS escalations (``score=0``) are inherently "two table patterns
claiming the turn" and answer NONE — the stub never resolves them.
"""
from __future__ import annotations

from ..base import ROUTE_NONE, ROUTE_SELECTED, CapabilityRoute

_TRUSTED = ("recall", "matcher_hit")


def _margin() -> float:
    from core.config import settings

    return settings.chat_funnel_margin


class StubSelector:
    """Deterministic :class:`..CapabilitySelector`: no model, no arguments."""

    async def select(self, query: str, candidates, *, entries_by_id: dict,
                     facts=None) -> CapabilityRoute:
        if not candidates:
            return CapabilityRoute(ROUTE_NONE, provenance="no candidates")
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        head = ranked[0]
        if len(ranked) == 1:
            if head.origin not in _TRUSTED:
                return CapabilityRoute(
                    ROUTE_NONE, provenance="matcher_ambiguous without scores")
            return CapabilityRoute(
                ROUTE_SELECTED, head.capability_id, confidence=head.score,
                provenance="single trusted candidate")
        second = ranked[1]
        margin = _margin()
        if (head.origin in _TRUSTED and second.origin in _TRUSTED
                and head.score - second.score >= margin):
            return CapabilityRoute(
                ROUTE_SELECTED, head.capability_id, confidence=head.score,
                provenance=f"margin {head.score - second.score:.3f} >= {margin}")
        return CapabilityRoute(
            ROUTE_NONE, provenance="race too close / mixed provenance")

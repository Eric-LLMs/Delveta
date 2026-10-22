"""cap_router test double: deterministic capability selection.

This is NOT a production backend — production has exactly one selector, the
deployed cap_router service. :class:`StubSelector` exists so tests can drive the
``MISS/AMBIGUOUS -> Recall -> Aggregation -> cap_router -> ONE|NONE`` wiring
without a model call; tests inject it through ``ChatDeps.cap_router_selector``.
It inherits the leader-vs-runner-up margin discipline the cosine corpus proved,
but answers exactly ONE question — *which* capability (or NONE). It authors
nothing else: no arguments, no Binder contact, no context resolution, no model
call.

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

"""Funnel-level rollout policy — what may run, and how failures are named.

One home for the enablement questions the cascade asks but never answers:
the funnel-wide gate (:func:`funnel_live`), the per-kind capability gate
(:func:`kind_enabled`) and the stage->8.10 reason mapping (:func:`stage_reason`).
Policy decides WHETHER a stage may run and how to EXPLAIN an exit; it never
implements a node and nodes never read scattered rollout switches themselves.

Single path (ruling): the funnel IS the product's formal routing
lane — the old dark-launch rollout switches were deleted. What remains here
is the fail-open door (missing deps) and the per-kind gate: unknown intent
kinds route nothing. There is no entry veto — every wired turn reaches the
Matcher.
"""
from __future__ import annotations

from .contract import (
    REASON_CAP_ROUTER_TIMEOUT,
    REASON_CASCADE_ERROR,
    REASON_CASCADE_TIMEOUT,
    REASON_RECALL_TIMEOUT,
    REASON_RECALL_UNAVAILABLE,
    REASON_REGISTRY_UNAVAILABLE,
)


def funnel_live(deps) -> bool:
    """The funnel-wide door: open whenever the cascade's dependencies are wired.
    There is no Funnel-vs-Agent mode switch and no entry veto any more —
    abstention inside the cascade is what routes, and the Matcher is the first
    node that sees the turn. Missing deps fail open to the Agent."""
    return deps is not None


def kind_enabled(kind: str) -> bool:
    """Per-kind capability gate: ACTION is the shipped lane (the caller already
    passed the funnel-wide door); an intent kind outside the shipped set routes
    NOTHING — being IN the table was never the same as being routable. Widened
    kinds (private/web) re-enter only with their own explicit rollout switch."""
    return kind in ("", "action")


def stage_reason(stage: str, *, timed_out: bool) -> str:
    """8.10 reason code for the exit point the cascade died at."""
    if timed_out:
        return {
            "recall": REASON_RECALL_TIMEOUT,
            "cap_router": REASON_CAP_ROUTER_TIMEOUT,  # the ONE selection hop owns the budget
        }.get(stage, REASON_CASCADE_TIMEOUT)
    return {
        "registry": REASON_REGISTRY_UNAVAILABLE,
        "recall": REASON_RECALL_UNAVAILABLE,
    }.get(stage, REASON_CASCADE_ERROR)

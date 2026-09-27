"""Funnel-level rollout policy — what may run, and how failures are named.

One home for the enablement questions the cascade asks but never answers:
the funnel-wide gate (:func:`funnel_live`), the per-kind rollout gate
(:func:`kind_enabled`, P3 逐开关灰度) and the stage->8.10 reason mapping
(:func:`stage_reason`). Policy decides WHETHER a stage may run and how to
EXPLAIN an exit; it never implements a node and nodes never read scattered
rollout switches themselves.
"""
from __future__ import annotations

from core.config import settings

from .contract import (
    REASON_CASCADE_ERROR,
    REASON_CASCADE_TIMEOUT,
    REASON_RECALL_TIMEOUT,
    REASON_RECALL_UNAVAILABLE,
    REASON_REGISTRY_UNAVAILABLE,
    REASON_TOOL_INTENT_TIMEOUT,
)


def funnel_live(requirements, deps, ctx) -> bool:
    """Gate for the P2 target cascade: master funnel switch + the plan-level
    action sub-gate + the common-layer guardrails (:mod:`guardrails`). From P3
    on, non-ACTION kinds additionally pass :func:`kind_enabled` per candidate;
    this gate stays the funnel-wide door."""
    if not (settings.chat_funnel_enabled and settings.chat_fast_paths_enabled
            and settings.chat_action_fast_path_enabled and deps is not None):
        return False
    from . import guardrails  # late import: keeps the module chain light

    return guardrails.turn_veto(ctx.body.message or "", requirements, ctx) is None


def kind_enabled(kind: str) -> bool:
    """P3 per-kind rollout gate (逐开关灰度): ACTION rides the master funnel gate
    (the caller already passed it); each widened kind needs its own switch, and
    an unknown kind routes nothing. Being IN the table was never the same as
    being ON."""
    if kind in ("", "action"):
        return True
    if kind == "private":
        return settings.chat_funnel_private_enabled
    if kind == "web":
        return settings.chat_funnel_web_enabled
    return False


def stage_reason(stage: str, *, timed_out: bool) -> str:
    """8.10 reason code for the exit point the cascade died at."""
    if timed_out:
        return {
            "recall": REASON_RECALL_TIMEOUT,
            "tool_intent": REASON_TOOL_INTENT_TIMEOUT,  # the ONE model hop owns the budget
        }.get(stage, REASON_CASCADE_TIMEOUT)
    return {
        "registry": REASON_REGISTRY_UNAVAILABLE,
        "recall": REASON_RECALL_UNAVAILABLE,
    }.get(stage, REASON_CASCADE_ERROR)

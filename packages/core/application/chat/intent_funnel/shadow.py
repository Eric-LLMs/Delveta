"""Shadow lanes — observation with zero execution authority.

The 2026-09-28 single-path ruling deleted the 8.15 Matcher dark-launch hook
(``chat_matcher_mode`` + ``observe``): the Matcher node inside the cascade is
the formal consumer, so a second observation-only copy of it no longer has a
product role. What remains is the Phase-E full-cascade shadow:

:func:`cascade_shadow` — one dry-run turn through the SAME orchestrator body
as production (zero cascade duplication, so the shadow can never drift from
shipped semantics). Used by the preview console lane and offline tooling.

Invariants this module owns:

1. observation, never behavior — every failure is fail-quiet; the only
   observable effect of a broken shadow run is absent log lines (8.15: "结果
   不得影响当前 Agent 行为");
2. cost isolation — the whole observation runs under an ``execution_mode=shadow``
   pin (8.14), so any LLM/embedding usage a shadow stage records settles
   outside real-user billing. The pin is always reset, exception included.
"""
from __future__ import annotations

import logging

from core.application.chat.understanding import (
    Complexity,
    Confidence,
    Signal,
    TurnRequirements,
)
from core.infrastructure.request_context import (
    reset_request_execution_mode,
    set_request_execution_mode,
)

from . import observability
from .orchestrator import run_cascade

logger = logging.getLogger(__name__)


# ── Phase-E full-cascade shadow (SAME orchestrator body, raw-score seam open) ────

async def cascade_shadow(ctx, *, deps, requirements=None,
                         recall_min_score: float = 0.0,
                         model_candidate_floor: float = 0.58,
                         persist_event: bool = False,
                         turn_key: str = "") -> dict:
    """Phase-E Cascade Shadow: one dry-run turn through the SAME node body as
    production (:func:`orchestrator.run_nodes` — zero orchestration
    duplication, so the shadow can never drift from shipped semantics), with
    the raw-score seam opened: Recall keeps every candidate it scored
    (min_score=0; the live-table ruling already keeps EVERY hit >= threshold as
    its own candidate) and the model-facing set re-applies a floor, so ALL
    threshold buckets are recomputable OFFLINE from the captured raw scores —
    recall is never re-run per threshold. The chain stops at Binder: routing
    metadata only (8.8), no dispatch, no Runtime, no event row; usage is pinned
    ``execution_mode=shadow`` (8.14) and observability is log-only.
    ``would_execute`` is the certified turn's metadata, NOT a permission —
    by construction the cascade cannot execute anything from here.
    Never raises: faults surface as 8.10 fallback reasons, exactly as in
    production.

    Persistence seam for offline tooling (both defaults = byte-identical
    replay): ``persist_event=True`` also writes the 8.12 event row (still
    pinned ``execution_mode=shadow``) so observation accumulates in the same
    table the observability admin reads; ``turn_key`` rides the row's
    trace_json (plus the stage captures — never the query) as the join key."""
    if requirements is None:
        requirements = TurnRequirements(
            complexity=Complexity.LOW, confidence=Confidence.LOW,
            needs_web=Signal.LOW, needs_memory=False,
        )
    trace = observability.new_trace()
    capture: dict = {}
    token = set_request_execution_mode("shadow")
    try:
        out = await run_cascade(ctx, deps, requirements, trace,
                                recall_min_score=recall_min_score,
                                model_candidate_floor=model_candidate_floor,
                                capture=capture)
        if persist_event:
            # Inside the pin: the row says execution_mode=shadow (8.14). The
            # trace_json carries the turn_key join anchor + stage captures,
            # never the raw query (same no-query rule as production rows).
            _act = (out.requested_action or {}) if out is not None else {}
            await observability.persist_event(deps, ctx, trace, {
                "turn_key": turn_key or None,
                "matcher": capture.get("matcher"),
                "candidates": capture.get("candidates", []),
                "model_verdict": capture.get("tool_intent"),
                "would_execute": {
                    k: _act.get(k) for k in
                    ("capability_id", "args", "funnel_stage", "funnel_kind")
                } if _act else None,
            })
    finally:
        reset_request_execution_mode(token)
    observability.log_trace(trace)
    result = {
        "deepest_stage": trace["stage"], "matcher": trace["matcher"],
        "recall_count": trace["recall_count"], "recall_top": trace["recall_top"],
        "tool_intent": trace["tool_intent"],
        "final_route": trace["final_route"], "fallback_reason": trace["fallback"],
        "registry_version": trace["registry"], "index_version": trace["index"],
        "total_ms": trace["total_ms"], "execution_mode": "shadow",
        "capture": capture,
    }
    act = (out.requested_action or {}) if out is not None else {}
    result["would_execute"] = ({k: act.get(k) for k in (
        "capability_id", "args", "funnel_stage", "funnel_kind",
    )} if out is not None else None)
    return result

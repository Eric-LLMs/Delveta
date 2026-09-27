"""§8.5 full-chain query preview — dry-run over the SAME orchestrator.

Preview is an execution mode, not a second funnel: it drives
:func:`orchestrator.run_cascade` with the production node body and returns the
trace as a verdict, executing nothing. Side-effect-free by construction (the
chain only produces routing metadata, 8.8); usage is pinned
``execution_mode=preview`` (8.14) and the routing event lands with the same
mode (8.12). Never raises: faults surface as the 8.10 fallback_reason, exactly
as they would in production.
"""
from __future__ import annotations

import types

from core.application.chat.understanding import (
    Complexity,
    Confidence,
    Signal,
    TurnRequirements,
)

from . import observability
from .orchestrator import run_cascade


def _preview_ctx(message: str):
    """The minimal ctx a console can express as a one-off test query: pure
    text, no viewer/attachment/research/handoff. (Drafts with context facts
    ride the same contract later; the Matcher only consumes TurnFacts fields.)"""
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, research_turn=False, effective_handoff=None,
        session_id="",
    )


async def preview(message: str, *, deps) -> dict:
    """§8.5: run the ACTIVE (Registry, Index) pair end to end for one query —
    Registry → Matcher → (Recall) → ToolIntentModel → Binder → Final Route —
    and return the trace as a verdict, executing nothing."""
    from core.infrastructure.request_context import (
        reset_request_execution_mode,
        set_request_execution_mode,
    )

    requirements = TurnRequirements(
        complexity=Complexity.LOW, confidence=Confidence.LOW,
        needs_web=Signal.LOW, needs_memory=False,
    )
    ctx = _preview_ctx(message)
    trace = observability.new_trace()
    token = set_request_execution_mode("preview")
    capture: dict = {}
    try:
        out = await run_cascade(ctx, deps, requirements, trace, capture=capture)
        await observability.persist_event(deps, ctx, trace)  # inside the pin: the event says "preview"
    finally:
        reset_request_execution_mode(token)
    observability.log_trace(trace)
    result = {
        "deepest_stage": trace["stage"], "matcher": trace["matcher"],
        "recall_count": trace["recall_count"], "recall_top": trace["recall_top"],
        "tool_intent": trace["tool_intent"],
        "final_route": trace["final_route"], "fallback_reason": trace["fallback"],
        "registry_version": trace["registry"], "index_version": trace["index"],
        "total_ms": trace["total_ms"], "execution_mode": "preview",
        # Console dry-run detail (Phase 5): what the model actually saw — each
        # candidate's origin, matched corpus sentence and kind — plus the raw
        # verdict and Binder state. Read-only projection of `capture`.
        "candidates": capture.get("candidates", []),
        "recall_raw": capture.get("recall_raw", []),
        "model_verdict": capture.get("tool_intent"),
        "binder_state": capture.get("binder"),
    }
    if out is not None:
        act = out.requested_action or {}
        result["route"] = {k: act.get(k) for k in (
            "capability_id", "tool", "args", "funnel_stage", "funnel_kind",
            "binding_integrity",
        )}
    return result

"""The per-run trace record: its shape and its capture-blob projection."""
from __future__ import annotations


def new_trace() -> dict:
    """The shared per-run trace record: production routing and the 8.5 query
    preview fill the SAME fields — one observability shape (8.12), one event
    row shape, so a preview can be diffed against real traffic line for line.
    The retired Decision node's ``decision`` field was dropped outright
    (migration 0008) — trace lines and event rows carry no historical name."""
    return {"stage": "registry", "matcher": "-", "recall_count": 0,
            "recall_top": "-", "tool_intent": "-",
            "final_route": "agent", "fallback": "-", "registry": "-",
            "index": "-", "capability": None, "total_ms": 0}


def trace_json(capture: dict | None, ctx) -> dict | None:
    """The observability blob (never the full prompt): what the model saw and
    what it decided, rebuilt from the capture seam. None = capture off.

    ``cap_router`` carries the selection decision (decision / capability_id /
    confidence / option_slots / v2_ineligible_reason) and ``acquisition`` the
    Argument Path Router's outcome (declared / strategy / model_slots /
    system_slots / bundle_source / readiness). ``timings`` carries the
    per-stage milliseconds the funnel measured. The row shape is a JSONB blob,
    so a new stage adds a key here, never a column."""
    if capture is None:
        return None
    return {
        "query": getattr(getattr(ctx, "body", None), "message", None),
        "matcher": capture.get("matcher"),
        "candidates": capture.get("candidates", []),
        "model_verdict": capture.get("tool_intent"),
        "cap_router": capture.get("cap_router"),
        "acquisition": capture.get("acquisition"),
        "entry": capture.get("entry"),
        "binder_state": capture.get("binder"),
        "timings": capture.get("timings", {}),
    }

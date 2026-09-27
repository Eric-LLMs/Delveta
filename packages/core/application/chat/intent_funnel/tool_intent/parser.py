"""Reply parsing: raw model output -> structured ToolIntentVerdict.

The correctness gate every backend's reply passes through (shared by local,
online and the ladder): candidate-set membership FIRST (off-card invention is
UNCERTAIN, never a verdict), then the confidence floor, with the raw
confidence kept as telemetry even when the floor downgrades it (Phase E
offline sweeps). "NONE" -> REJECT; a missing/garbage id -> REJECT/UNCERTAIN.
"""
from __future__ import annotations

from ..contract import (
    TOOL_INTENT_CONFIDENT,
    TOOL_INTENT_REJECT,
    TOOL_INTENT_UNCERTAIN,
    ToolIntentVerdict,
)


def verdict_from_reply(data: dict, candidates) -> ToolIntentVerdict:
    cap_id = str(data.get("capability_id") or "").strip()
    valid = {c.capability_id for c in candidates}
    args = data.get("arguments")
    args = dict(args) if isinstance(args, dict) else None
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not cap_id or cap_id.upper() == "NONE":
        return ToolIntentVerdict(TOOL_INTENT_REJECT, None, "tool_intent chose NONE",
                                 confidence=None)
    if cap_id not in valid:
        # off-card invention stays an uncertainty, it is never a verdict
        return ToolIntentVerdict(TOOL_INTENT_UNCERTAIN, None, f"off-card id {cap_id!r}",
                                 confidence=None)
    from core.config import settings

    if confidence < settings.chat_tool_intent_min_confidence:
        return ToolIntentVerdict(
            TOOL_INTENT_UNCERTAIN, cap_id, f"confidence {confidence:.2f} below floor",
            confidence=confidence,  # telemetry: the raw value, floor kept honest
        )
    return ToolIntentVerdict(
        TOOL_INTENT_CONFIDENT, cap_id, f"confidence {confidence:.2f}", arguments=args,
        confidence=confidence,
    )

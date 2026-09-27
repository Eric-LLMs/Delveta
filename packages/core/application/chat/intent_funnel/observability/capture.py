"""Trace output channels: the structured log line and the 8.12 event row."""
from __future__ import annotations

import logging

# OBSERVABILITY COMPAT: the trace line and the fail-open line keep the
# historic ``...intent_funnel.funnel`` logger NAME although the code now lives
# here — log filters, caplog assertions and any external log routing key on
# that string; renaming it would be an observability-semantics change, not a
# structural move.
logger = logging.getLogger("core.application.chat.intent_funnel.funnel")


def log_trace(trace: dict) -> None:
    logger.info(
        "funnel_trace deepest_stage=%s matcher=%s recall_count=%d recall_top=%s "
        "tool_intent=%s final_route=%s fallback_reason=%s "
        "registry_version=%s index_version=%s total_ms=%d",
        trace["stage"], trace["matcher"], trace["recall_count"], trace["recall_top"],
        trace["tool_intent"], trace["final_route"], trace["fallback"],
        trace["registry"], trace["index"], trace["total_ms"],
    )


async def persist_event(deps, ctx, trace: dict, trace_json: dict | None = None) -> None:
    """8.12: one row per route decision, best-effort. Telemetry must never sink
    a turn or delay a preview, and an unwired session factory (unit tests,
    dark lanes) is a silent no-op. The raw query is deliberately NOT stored;
    ``execution_mode`` (8.14) separates production from shadow/preview/test.
    ``trace_json`` (Phase 6, settings.chat_funnel_trace_capture) is the single
    exception to the no-query rule: a dark-launched, admin-gated observability
    blob carrying query + rebuilt cards + verdict — never the full prompt."""
    factory = getattr(deps, "session_factory", None)
    if factory is None:
        return
    try:
        from core.infrastructure.db import ChatFunnelEventModel
        from core.infrastructure.request_context import (
            get_request_execution_mode,
            get_request_user_id,
        )

        async with factory() as session:
            session.add(ChatFunnelEventModel(
                execution_mode=get_request_execution_mode(),
                user_id=get_request_user_id(),
                session_id=str(getattr(ctx, "session_id", "") or "") or None,
                deepest_stage=trace["stage"], matcher=trace["matcher"],
                recall_count=trace["recall_count"], recall_top=trace["recall_top"],
                tool_intent=trace["tool_intent"],
                final_route=trace["final_route"], fallback_reason=trace["fallback"],
                registry_version=trace["registry"], index_version=trace["index"],
                capability_id=trace["capability"], total_ms=trace["total_ms"],
                trace_json=trace_json,
            ))
            await session.commit()
    except Exception as exc:
        logger.info("funnel event persist skipped: %r", exc)

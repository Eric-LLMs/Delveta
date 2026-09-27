"""Funnel observability — the trace record, its log line and its event row.

Observer only (8.12/8.15): every function here reads the routing outcome and
records it; none may change it. Persistence failures are swallowed to an info
line by design (a telemetry fault must never sink a turn or alter a route).
"""
from .trace import new_trace, trace_json
from .capture import log_trace, persist_event

__all__ = ["new_trace", "trace_json", "log_trace", "persist_event"]

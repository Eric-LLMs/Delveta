"""Deferred capture broker — pause ONE streaming turn until the client supplies the image.

The Intent Funnel decides that this turn needs the ``vision`` tool BEFORE any
screenshot exists (a local file lives only on the user's PC, so the server cannot
capture it). The turn must therefore WAIT: the orchestrator opens a ticket, emits a
``capture-request`` SSE frame, and blocks on the ticket's future. The client captures
the page/frame, uploads it through the existing chat-asset route, and POSTs the asset
id back to ``/chat/captures/{ticket_id}``; the future resolves and the SAME turn
resumes with the plan the funnel already certified — no second funnel run, no second
argument extraction, no duplicate tool execution.

A ticket is single-use and scoped to the user+session that opened it. The broker is
in-process state (the API runs one event loop): tickets are created and resolved on
that loop, so no locking is needed. Closed/expired tickets resolve nothing; the
orchestrator owns the retry/timeout frames.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from uuid import uuid4

# submit() outcomes. The route maps them to HTTP; the orchestrator only ever sees the
# payload the future resolves with.
STATUS_OK = "ok"
STATUS_NOT_FOUND = "not_found"
STATUS_FORBIDDEN = "forbidden"
STATUS_CLOSED = "closed"


@dataclass
class CaptureTicket:
    """One pending client capture: identity for authorization + the resume future."""

    id: str
    user_id: str
    session_id: str
    future: asyncio.Future


class CaptureBroker:
    """Create/submit/close capture tickets for paused streaming turns."""

    def __init__(self) -> None:
        self._tickets: dict[str, CaptureTicket] = {}

    def open(self, *, user_id, session_id) -> CaptureTicket:
        ticket = CaptureTicket(
            id=uuid4().hex,
            user_id=str(user_id),
            session_id=str(session_id),
            future=asyncio.get_running_loop().create_future(),
        )
        self._tickets[ticket.id] = ticket
        return ticket

    def submit(self, ticket_id: str, *, user_id, payload: dict) -> str:
        """Resolve a ticket with the client's capture result. Never raises."""
        ticket = self._tickets.get(str(ticket_id))
        if ticket is None:
            return STATUS_NOT_FOUND
        if str(user_id) != ticket.user_id:
            return STATUS_FORBIDDEN
        if ticket.future.done():
            return STATUS_CLOSED
        ticket.future.set_result(dict(payload))
        return STATUS_OK

    def close(self, ticket_id: str) -> None:
        """Drop a ticket; an unresolved future resolves as failed so no waiter hangs."""
        ticket = self._tickets.pop(str(ticket_id), None)
        if ticket is not None and not ticket.future.done():
            ticket.future.set_result({"status": "failed", "reason": "closed"})

    def pending(self) -> int:
        """Number of open tickets (tests/diagnostics only)."""
        return len(self._tickets)


def merge_payload(spec: dict | None, submitted: dict | None) -> dict:
    """Combine the capture-request spec with the client's submission.

    The spec (kind/name/page/t_ms) is the server's authority for HOW to interpret
    the submission; the client's non-null fields (asset_id, attachment_id, text,
    reason) win where they exist. Never mutates either input.
    """
    merged = dict(spec or {})
    merged.update({k: v for k, v in (submitted or {}).items() if v is not None})
    return merged


def resume_plan(plan, payload: dict | None):
    """Return the paused plan resumed with the client's uploaded capture.

    Only ``asset_id`` is injected into the certified action's args — the capability,
    tool, registry fingerprint and kind were already certified by the funnel, so this
    is a pure patch, never a re-route or a second argument extraction. The capture
    marker is cleared so the resumed turn cannot pause again.
    """
    payload = payload or {}
    asset_id = str(payload.get("asset_id") or "").strip()
    if plan is None or not isinstance(plan.action, dict) or not asset_id:
        return plan
    action = dict(plan.action)
    args = dict(action.get("args") or {})
    args["asset_id"] = asset_id
    action["args"] = args
    return replace(plan, action=action, capture_request=None)


def apply_capture_to_context(ctx, payload: dict | None, viewer_deps) -> None:
    """Fold a successful capture into the SAME turn's context before execution.

    The viewer assembly gains the capture block (image asset + prepared text) so the
    ``vision`` tool can send text + picture + question in ONE call, and the uploaded
    draft attachment is bound to this turn's user message. The Agent context is
    rebuilt from the updated assembly — ``_run_tool`` binds it for the tool.
    """
    payload = payload or {}
    with_capture = getattr(viewer_deps, "with_capture", None)
    if callable(with_capture):
        ctx.viewer_assembly = with_capture(ctx.viewer_assembly, payload)
    if ctx.viewer_assembly is not None:
        ctx.agent_context = {**(ctx.agent_context or {}),
                             "viewer": ctx.viewer_assembly}
    attachment_id = payload.get("attachment_id")
    if attachment_id and ctx.session_memory is not None:
        add = getattr(ctx.session_memory, "add_attachment_ids", None)
        if callable(add):
            add([attachment_id])

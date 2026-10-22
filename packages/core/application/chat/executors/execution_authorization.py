"""Final execution-boundary authorization for the Chat ACTION lane.

Routing answers "which capability does this turn mean"; execution authorization
answers "may this process actually run it from the Chat plane, right now". They
are deliberately different questions: recall/card exposure is wider than the
execution allowlist, and a model pick or a routing-time stamp is not authority.

Every field is read from the AUTHORITATIVE live Registry view resolved by the
caller immediately before dispatch. This module is pure: it never loads the
registry, never calls a tool, and never falls back to a default capability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Structured refusal reasons (stable machine-readable codes).
REASON_CAPABILITY_NOT_REGISTERED = "CAPABILITY_NOT_REGISTERED"
REASON_CAPABILITY_INACTIVE = "CAPABILITY_INACTIVE"
REASON_CAPABILITY_NOT_ALLOWLISTED = "CAPABILITY_NOT_ALLOWLISTED"
REASON_TOOL_BINDING_MISMATCH = "TOOL_BINDING_MISMATCH"


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str | None = None
    entry: Any = None


def authorize_execution(view: Any, *, capability_id: str, tool: str) -> AuthorizationDecision:
    """Decide whether ``capability_id`` -> ``tool`` may execute from the Chat plane.

    Ordering matters for diagnostics: unknown, then lifecycle, then the chat-plane
    allowlist, then binding. Hidden capabilities are enabled/active rows, so they
    must be rejected by ``chat_plane_candidate`` rather than by the lifecycle
    checks — this is the distinction between "may be recalled/offered as a card"
    and "may actually execute".
    """
    from core.application.chat.intent_funnel.registry.entry import (
        STATUS_ACTIVE,
        chat_plane_candidate,
    )

    cid = str(capability_id or "")
    entry = next(
        (e for e in (view.entries if view is not None else ())
         if e.capability_id == cid),
        None,
    )
    if entry is None:
        return AuthorizationDecision(False, REASON_CAPABILITY_NOT_REGISTERED)
    if not entry.enabled or entry.status != STATUS_ACTIVE:
        return AuthorizationDecision(False, REASON_CAPABILITY_INACTIVE)
    if not chat_plane_candidate(entry):
        return AuthorizationDecision(False, REASON_CAPABILITY_NOT_ALLOWLISTED)
    if entry.tool_binding != tool:
        return AuthorizationDecision(False, REASON_TOOL_BINDING_MISMATCH)
    return AuthorizationDecision(True, None, entry)

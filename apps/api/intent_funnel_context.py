"""Intent Funnel routing note — turn-scoped prompt context for an admission reject.

When the Intent Funnel declines to certify the capability the selector picked
(``INTENT_FUNNEL_ALLOWLIST_REJECT``), the turn falls open to the Agent. This module
renders that fact as trusted, app-generated context in the DYNAMIC_SUFFIX zone, so the
Agent understands WHY the funnel did not handle the request — and that the refusal is
scoped to the funnel's own capability admission, never a system-wide execution ban.

The note comes from ``AgentTurn.context["intent_funnel"]``, sunk by the control plane
from ``ChatTurnContext.intent_funnel_reject``. It renders "" whenever the turn carries
no reject record, so every other turn assembles a byte-identical prompt.

Hard rules:
- This section is advisory routing context, not policy: it must not add, remove,
  expose or authorize any tool. The Agent's own registry, permissions, approvals and
  sandbox remain the only authority over what may run.
- Values are app-generated registry identifiers, still flattened to one line and
  capped so a malformed row can never forge extra prompt lines.
"""
from __future__ import annotations

from agent.engine.context import current_turn

_MAX_FIELD_CHARS = 120


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())[:_MAX_FIELD_CHARS]


def render_intent_funnel_reject(reject: dict) -> str:
    """Render the reject record into the dynamic-suffix section body ('' when empty)."""
    reason = _one_line(reject.get("reason"))
    if not reason:
        return ""
    capability = _one_line(reject.get("capability_id"))
    tool = _one_line(reject.get("tool_binding"))
    detail = f"reason: {reason}"
    if capability:
        detail += f"; capability: {capability}"
    if tool:
        detail += f"; tool: {tool}"
    return (
        "## Intent Funnel routing note\n"
        "The Intent Funnel considered a capability for this turn but declined to certify\n"
        f"it ({detail}). That decision applies to the Intent Funnel's own capability\n"
        "admission only. It is NOT a system-wide prohibition and it does NOT remove or\n"
        "restrict any tool registered to you now.\n"
        "The user's original request is unchanged and is yours to handle: decide under\n"
        "your own tool registry, permissions, approvals and sandbox rules whether and how\n"
        "to act, or answer directly when no tool is needed. Do not tell the user the\n"
        "request is forbidden, and do not refuse merely because the Intent Funnel abstained."
    )


async def intent_funnel_reference_section(context: dict) -> str:
    """Prompt section callable registered on the kernel assembler (DYNAMIC_SUFFIX).

    Reads this turn's sunk reject record from ``AgentTurn.context["intent_funnel"]``.
    No turn, no record, or no reason → ``""`` (zero prompt delta).
    """
    turn = (context or {}).get("turn") or current_turn()
    if turn is None or not getattr(turn, "context", None):
        return ""
    reject = turn.context.get("intent_funnel")
    if not isinstance(reject, dict):
        return ""
    return render_intent_funnel_reject(reject)

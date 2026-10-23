"""Intent Funnel routing note — turn-scoped prompt context for a funnel fallback.

When the Intent Funnel does not certify a turn, the original turn falls open to
the Agent. A reason code alone is not enough: the Agent needs the stage, the
capability, the acquisition strategy, the arguments already resolved (and their
sources), the required contract, what is missing/invalid, and the execution
state. This module renders that record as trusted, app-generated context in the
DYNAMIC_SUFFIX zone.

The note comes from ``AgentTurn.context["intent_funnel"]``, sunk by the control
plane from ``ChatTurnContext.intent_funnel_failure``. It renders "" whenever the
turn carries no record, so every other turn assembles a byte-identical prompt.

Hard rules:
- This section is advisory routing context, not policy: it must not add, remove,
  expose or authorize any tool. The Agent's own registry, permissions, approvals
  and sandbox remain the only authority over what may run.
- Every field is app-generated or already part of the user's turn; values are
  flattened to one line and capped so a malformed row can never forge extra
  prompt lines.
- It never claims an action succeeded. Execution-boundary records carry their
  own ``execution_state``; only a proven pre-body failure is published.
"""
from __future__ import annotations

from agent.engine.context import current_turn
from core.application.chat.intent_funnel.failure import (
    KIND_ACQUISITION_FAILURE,
    KIND_CONTRACT_MISMATCH,
    KIND_EXECUTION_PREFLIGHT,
    KIND_FUNNEL_ABSTAIN,
)

_MAX_FIELD_CHARS = 120
_MAX_VALUE_CHARS = 100
_MAX_QUERY_CHARS = 300
_MAX_ITEMS = 8
_MAX_SECTION_CHARS = 4000

_SCOPE = (
    "This is routing context, not a system-wide prohibition: it does not add,\n"
    "remove, expose or restrict any tool. Your own tool registry, permissions,\n"
    "approvals and sandbox remain the only authority. The user's original request\n"
    "is unchanged and is yours to handle."
)


def _one_line(value: object, limit: int = _MAX_FIELD_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _code(value: object) -> str:
    text = _one_line(value).replace("`", "'")
    return f"`{text}`" if text else "(unrecorded)"


def _list(value: object) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        return "none"
    return ", ".join(_one_line(v) for v in list(value)[:_MAX_ITEMS]) or "none"


def _map(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts = [
        f"{_one_line(k)}={_one_line(v)}"
        for k, v in sorted(value.items())[:_MAX_ITEMS]
    ]
    return "; ".join(parts) or "none"


def _required(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    for name in sorted(value):
        spec = value[name] if isinstance(value[name], dict) else {}
        bits = [_one_line(spec.get("type") or "string")]
        bits.append("required" if spec.get("required", True) else "optional")
        if spec.get("max_len"):
            bits.append(f"max_len={spec['max_len']}")
        if spec.get("items"):
            bits.append(f"items={_one_line(spec['items'])}")
        if spec.get("enum"):
            bits.append("enum=" + "|".join(
                _one_line(v) for v in list(spec["enum"])[:_MAX_ITEMS]))
        parts.append(f"{_one_line(name)} ({', '.join(bits)})")
    return "; ".join(parts) or "none"


def _args(value: object, sources: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    source_map = sources if isinstance(sources, dict) else {}
    parts: list[str] = []
    for name in sorted(value)[:_MAX_ITEMS]:
        raw = value[name]
        if isinstance(raw, (list, tuple)):
            rendered = "[" + ", ".join(
                _one_line(v, _MAX_VALUE_CHARS) for v in list(raw)[:_MAX_ITEMS]) + "]"
        else:
            rendered = _one_line(raw, _MAX_VALUE_CHARS)
        source = _one_line(source_map.get(name))
        parts.append(f"{_one_line(name)}={rendered}" + (f" [{source}]" if source else ""))
    return "; ".join(parts) or "none"


def _context(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    parts: list[str] = []
    if value.get("has_viewer"):
        kind = _one_line(value.get("viewer_kind"))
        name = _one_line(value.get("viewer_name"))
        label = "viewer" + (f" ({kind})" if kind else "")
        if name:
            label += f' "{name}"'
        parts.append(label)
        page_from = value.get("viewer_page_from")
        page_to = value.get("viewer_page_to")
        if page_from is not None:
            parts.append(f"pages {page_from}-{page_to if page_to is not None else page_from}")
    if value.get("has_attachment"):
        parts.append("attachment present")
    if value.get("has_viewer_selection"):
        parts.append("viewer selection present")
    if value.get("has_resolved_path_asset"):
        parts.append("sentence path resolved to an asset")
    if value.get("has_turn_context"):
        parts.append("session context present")
    return "; ".join(parts) or "none"


def _header(reason: str, stage: str, state: str) -> list[str]:
    detail = f"reason: {reason}"
    if stage:
        detail += f"; stage: {stage}"
    if state:
        detail += f"; execution state: {state}"
    return [f"({detail})."]


def _render_admission(record: dict) -> str:
    reason = _one_line(record.get("reason"))
    if not reason:
        return ""
    capability = _one_line(record.get("capability_id"))
    tool = _one_line(record.get("tool_binding"))
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


def _render_acquisition(record: dict) -> str:
    reason = _one_line(record.get("reason"))
    stage = _one_line(record.get("stage"))
    state = _one_line(record.get("execution_state"))
    lines = [
        "## Intent Funnel routing note",
        (f"The Intent Funnel selected capability {_code(record.get('capability_id'))} "
         f"(tool {_code(record.get('tool_binding'))}) for this turn but could not acquire "
         "the arguments it requires."),
        *_header(reason, stage, state),
        f"- acquisition strategy: {_one_line(record.get('strategy')) or 'none'}",
        f"- required parameters: {_required(record.get('required'))}",
        f"- existing arguments: {_args(record.get('args'), record.get('arg_sources'))}",
        f"- missing parameters: {_list(record.get('missing'))}",
    ]
    invalid = _map(record.get("invalid"))
    if invalid != "none":
        lines.append(f"- invalid parameters: {invalid}")
    query = _one_line(record.get("query"), _MAX_QUERY_CHARS)
    if query:
        lines.append(f"- original request: {query}")
    lines.append(f"- available context: {_context(record.get('context'))}")
    detail = _one_line(record.get("detail"), 240)
    if detail:
        lines.append(f"- detail: {detail}")
    lines.extend([
        "Nothing was executed. Treat the missing arguments as a question you may need",
        "to resolve from the user's request or the available context before acting.",
        _SCOPE,
    ])
    return "\n".join(lines)[:_MAX_SECTION_CHARS]


def _render_contract(record: dict) -> str:
    reason = _one_line(record.get("reason"))
    stage = _one_line(record.get("stage"))
    state = _one_line(record.get("execution_state"))
    lines = [
        "## Intent Funnel routing note",
        (f"The Intent Funnel selected capability {_code(record.get('capability_id'))} "
         f"(tool {_code(record.get('tool_binding'))}), but the runtime rejected its "
         "arguments before execution."),
        *_header(reason, stage, state),
        f"- required parameters: {_required(record.get('required'))}",
        f"- actual arguments: {_args(record.get('args'), record.get('arg_sources'))}",
        f"- missing parameters: {_list(record.get('missing'))}",
        f"- unexpected parameters: {_list(record.get('extra'))}",
        f"- invalid arguments: {_map(record.get('invalid'))}",
    ]
    query = _one_line(record.get("query"), _MAX_QUERY_CHARS)
    if query:
        lines.append(f"- original request: {query}")
    original = _one_line(record.get("original_error"), 240)
    if original:
        lines.append(f"- original error: {original}")
    lines.extend([
        "Nothing was executed, so acting on the user's request now cannot duplicate a",
        "prior run. Correct the arguments (or ask the user for the missing value) using",
        "your own tools, permissions and approvals.",
        _SCOPE,
    ])
    return "\n".join(lines)[:_MAX_SECTION_CHARS]


def _render_preflight(record: dict) -> str:
    reason = _one_line(record.get("reason"))
    stage = _one_line(record.get("stage"))
    state = _one_line(record.get("execution_state"))
    lines = [
        "## Intent Funnel routing note",
        (f"The Intent Funnel selected capability {_code(record.get('capability_id'))} "
         f"(tool {_code(record.get('tool_binding'))}), but it failed a preflight check "
         "before executing anything."),
        *_header(reason, stage, state),
        f"- actual arguments: {_args(record.get('args'), record.get('arg_sources'))}",
    ]
    original = _one_line(record.get("original_error") or record.get("detail"), 240)
    if original:
        lines.append(f"- preflight result: {original}")
    lines.extend([
        "No side effect occurred. This is not a parameter-contract mismatch: a",
        "permission, approval or runtime preflight rule decided the outcome. You may",
        "explain it or ask the user for the missing authority; do not claim the action",
        "ran.",
        _SCOPE,
    ])
    return "\n".join(lines)[:_MAX_SECTION_CHARS]


def render_intent_funnel_context(record: dict) -> str:
    """Render one structured fallback record ('' when it carries no reason)."""
    if not isinstance(record, dict):
        return ""
    reason = _one_line(record.get("reason"))
    if not reason:
        return ""
    kind = _one_line(record.get("kind"))
    if kind == KIND_CONTRACT_MISMATCH:
        return _render_contract(record)
    if kind == KIND_EXECUTION_PREFLIGHT:
        return _render_preflight(record)
    if kind in (KIND_ACQUISITION_FAILURE, KIND_FUNNEL_ABSTAIN):
        return _render_acquisition(record)
    # ADMISSION_REJECT and historic records without a kind.
    return _render_admission(record)


async def intent_funnel_reference_section(context: dict) -> str:
    """Prompt section callable registered on the kernel assembler (DYNAMIC_SUFFIX).

    Reads this turn's sunk fallback record from ``AgentTurn.context["intent_funnel"]``.
    No turn, no record, or no reason → ``""`` (zero prompt delta).
    """
    turn = (context or {}).get("turn") or current_turn()
    if turn is None or not getattr(turn, "context", None):
        return ""
    record = turn.context.get("intent_funnel")
    if not isinstance(record, dict):
        return ""
    return render_intent_funnel_context(record)

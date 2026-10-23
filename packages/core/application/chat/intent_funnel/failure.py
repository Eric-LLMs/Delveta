"""Turn-scoped structured context for an Intent Funnel fallback.

Every non-certified funnel outcome hands the ORIGINAL turn to the Agent. A bare
reason code is not enough context for that decision: the Agent needs the stage,
the capability, the acquisition strategy, the arguments already resolved (and
where each came from), what is missing or invalid, the required parameter
contract, and which turn context exists.

This module builds that record as plain, JSON-safe data and publishes it on the
turn (``ctx.intent_funnel_failure`` + ``ctx.agent_context["intent_funnel"]``).
It is routing context only:

* it never executes, authorizes, grants or removes a tool;
* it never changes the user's original request;
* it carries no credential, executor handle or permission bypass.

The Agent's own registry, permissions, approvals and sandbox remain the only
authority over what may run. A record is consumed by the Agent prompt section
(``apps/api/intent_funnel_context.py``) and by evaluation capture; it is never
read back by the request path.
"""
from __future__ import annotations

from typing import Any

# ── record kinds ────────────────────────────────────────────────────────────────
KIND_ADMISSION_REJECT = "ADMISSION_REJECT"
KIND_ACQUISITION_FAILURE = "ACQUISITION_FAILURE"
KIND_CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
KIND_EXECUTION_PREFLIGHT = "EXECUTION_PREFLIGHT"
# A funnel abstain that is neither argument acquisition nor a binder contract
# failure (today: the capability's kind gate is closed). It still explains WHY
# the turn fell through, without mislabelling it as a parameter problem.
KIND_FUNNEL_ABSTAIN = "FUNNEL_ABSTAIN"

# ── stages ──────────────────────────────────────────────────────────────────────
STAGE_ADMISSION = "admission"
STAGE_ACQUISITION = "acquisition"
STAGE_BINDER = "binder"
STAGE_EXECUTOR = "executor"

# ── execution state ─────────────────────────────────────────────────────────────
# The executor only hands the turn to the Agent when it can PROVE the tool body
# was not entered. State-unknown failures stay terminal in the executor and are
# never published here (a blind Agent replay could duplicate the side effect).
EXECUTION_NOT_STARTED = "not_started"
EXECUTION_UNKNOWN = "unknown"

# Structured-context reasons. These are NOT funnel trace fallback codes: the
# trace keeps its existing node codes (BIND_*, ACQUISITION_*), while this layer
# names the category the Agent is told about. One executor code covers every
# ``validate_action`` rejection; the original message rides ``original_error``.
REASON_EXECUTOR_PARAMETER_MISMATCH = "EXECUTOR_PARAMETER_MISMATCH"
REASON_EXECUTOR_PREFLIGHT = "EXECUTOR_PREFLIGHT"

_MAX_TEXT_CHARS = 600
_MAX_QUERY_CHARS = 1000
_MAX_VALUE_CHARS = 200
_MAX_ITEMS = 12


def _text(raw: object, limit: int = _MAX_TEXT_CHARS) -> str:
    if raw is None:
        return ""
    return " ".join(str(raw).split())[:limit]


def _value(raw: object) -> Any:
    """A JSON-safe scalar/array for the record. Model values are reported as
    produced (no validation rewrite); only representation is normalized."""
    if raw is None or isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw
    if isinstance(raw, str):
        return _text(raw, _MAX_VALUE_CHARS)
    if isinstance(raw, (list, tuple)):
        return [_value(v) for v in list(raw)[:_MAX_ITEMS]]
    return _text(raw, _MAX_VALUE_CHARS)


def _args(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): _value(v) for k, v in raw.items()}


def _sources(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): _text(v, 80) for k, v in raw.items()}


def required_parameters(schema: object) -> dict[str, dict]:
    """Project a Registry/tool schema into the Agent-facing required contract.

    The schema is reported, never interpreted: ``required`` defaults to True
    exactly as the Binder and ``validate_action`` treat it. Optional slots are
    included with their flag so the Agent can tell "not required" apart from
    "not declared".
    """
    if not isinstance(schema, dict):
        return {}
    out: dict[str, dict] = {}
    for name, raw in schema.items():
        spec = raw if isinstance(raw, dict) else {}
        item: dict[str, Any] = {
            "type": _text(spec.get("type") or "string", 40),
            "required": bool(spec.get("required", True)),
        }
        if spec.get("description"):
            item["description"] = _text(spec["description"], 160)
        if spec.get("max_len") is not None:
            item["max_len"] = spec["max_len"]
        if spec.get("enum"):
            item["enum"] = [_text(v, 60) for v in list(spec["enum"])[:_MAX_ITEMS]]
        if isinstance(spec.get("items"), dict) and spec["items"].get("type"):
            item["items"] = _text(spec["items"]["type"], 40)
        out[str(name)] = item
    return out


def missing_required_parameters(schema: object, args: object) -> list[str]:
    """Required slots with no non-blank value. Presence only — a value is never
    rewritten or "repaired" here."""
    values = args if isinstance(args, dict) else {}
    out: list[str] = []
    specs = schema if isinstance(schema, dict) else {}
    for name, raw in specs.items():
        spec = raw if isinstance(raw, dict) else {}
        if not spec.get("required", True):
            continue
        value = values.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            out.append(str(name))
    return out


def turn_context_summary(facts: object) -> dict:
    """The legal turn context the funnel could see, as booleans/identifiers.
    No asset ids, tokens or transcript content are copied into the record."""
    if facts is None:
        return {}
    out: dict[str, Any] = {
        "has_viewer": bool(getattr(facts, "has_viewer", False)),
        "has_attachment": bool(getattr(facts, "has_attachment", False)),
        "has_viewer_selection": bool(getattr(facts, "has_viewer_selection", False)),
        "has_turn_context": bool(getattr(facts, "has_turn_context", False)),
        "has_resolved_path_asset": bool(getattr(facts, "path_asset_id", "")),
    }
    kind = _text(getattr(facts, "viewer_kind", ""), 40)
    name = _text(getattr(facts, "viewer_name", ""), 160)
    if kind:
        out["viewer_kind"] = kind
    if name:
        out["viewer_name"] = name
    page_from = getattr(facts, "viewer_page_from", None)
    page_to = getattr(facts, "viewer_page_to", None)
    if page_from is not None:
        out["viewer_page_from"] = page_from
    if page_to is not None:
        out["viewer_page_to"] = page_to
    return out


def build_failure(
    *,
    kind: str,
    stage: str,
    reason: str,
    capability_id: str = "",
    tool_binding: str = "",
    strategy: str = "",
    query: str = "",
    args: object = None,
    arg_sources: object = None,
    missing: object = (),
    invalid: object = None,
    extra: object = (),
    required: object = None,
    detail: str = "",
    facts: object = None,
    execution_state: str = EXECUTION_NOT_STARTED,
    original_error: str = "",
) -> dict:
    """Build the plain record consumed by the Agent prompt / evaluation capture."""
    record: dict[str, Any] = {
        "kind": _text(kind, 60),
        "stage": _text(stage, 40),
        "reason": _text(reason, 80),
        "capability_id": _text(capability_id, 120),
        "tool_binding": _text(tool_binding, 120),
        "strategy": _text(strategy, 60),
        "query": _text(query, _MAX_QUERY_CHARS),
        "args": _args(args),
        "arg_sources": _sources(arg_sources),
        "missing": sorted({_text(s, 80) for s in (missing or ()) if _text(s, 80)}),
        "invalid": _sources(invalid),
        "extra": sorted({_text(s, 80) for s in (extra or ()) if _text(s, 80)}),
        "required": required_parameters(required),
        "context": turn_context_summary(facts),
        "detail": _text(detail),
        "execution_state": _text(execution_state, 40) or EXECUTION_NOT_STARTED,
    }
    if original_error:
        record["original_error"] = _text(original_error)
    return record


def publish_failure(ctx: object, record: dict) -> None:
    """Attach the record to THIS turn and to the kernel's ``run(context=…)``
    payload. Never raises: a context-less test double simply gains the fields."""
    if ctx is None:
        return
    payload = dict(record)
    agent_context = dict(getattr(ctx, "agent_context", None) or {})
    agent_context["intent_funnel"] = dict(payload)
    # Duck-typed turn object (production ``ChatTurnContext``, test doubles): the
    # fields are plain attributes, so direct assignment is the honest spelling.
    ctx.intent_funnel_failure = payload  # type: ignore[attr-defined]
    ctx.agent_context = agent_context  # type: ignore[attr-defined]

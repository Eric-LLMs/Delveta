"""Structured acquisition failure a capability handler returns to the funnel.

A handler normally returns its argument draft. Some requests, however, cannot be
turned into a draft at all — the content source is genuinely undecided, the
online scope model violated its output contract, or the user asked for a save
destination that cannot be honoured. Those are NOT "no arguments"; they are
distinct business outcomes and the Agent must be told WHICH one happened, on
what evidence.

This object carries that evidence from the handler to the orchestrator:

* ``category`` — the Agent-facing failure class (never a funnel trace code);
* ``stage`` — where the handler stopped (scope resolution / content selection /
  destination);
* ``detail`` — the specific reason, in one sentence;
* ``scope`` — the page-scope resolver's verdict when one was obtained;
* ``context`` — the trusted turn facts the Agent needs to continue: the viewer /
  attachment asset ids, the visible page(s), and whether the conversation holds
  a previous assistant reply.

It is routing context only. It never authorizes, executes, or fabricates an
argument; the Agent still goes through the normal Intent Funnel / acquisition /
Binder / authorization / execution chain before anything runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── categories ────────────────────────────────────────────────────────────────
# The resolver's reply broke its own output contract (unknown field, bad enum,
# ``all`` carrying pages, an identity field, an unparsable page spec, ...).
# Per the business rules this goes STRAIGHT to the Agent: it is not ordinary
# ambiguity and must never be auto-fallen-back to a page or the whole document.
CATEGORY_SCOPE_CONTRACT = "scope_contract_violation"
# The ONLINE argument extractor could not serve: the call failed / timed out /
# returned non-JSON, the extractor is not wired, or it produced no valid value
# for a slot this turn explicitly requires. Per the business rules this hands
# the turn to the Agent — it must NOT be treated as ordinary ambiguity and must
# NOT fall back to a page or a previous reply.
CATEGORY_EXTRACTION_FAILED = "argument_extraction_failed"
# The scope could not be settled at all (ambiguous / unknown / resolver
# unavailable / not wired) and the conversation has no assistant reply to fall
# back to.
CATEGORY_SCOPE_UNSETTLED = "scope_unsettled"
# The scope WAS settled, but the selected source has no usable content
# (an empty request, or a conversation with no non-empty user/assistant rows).
CATEGORY_NO_CONTENT = "no_content"
# The user asked to save the summary, but the named destination is not a usable
# workspace directory. Returning the summary in chat instead would silently
# ignore the request, so the turn exits.
CATEGORY_INVALID_DESTINATION = "invalid_destination"

# ── stages ────────────────────────────────────────────────────────────────────
STAGE_SCOPE_RESOLUTION = "scope_resolution"
STAGE_CONTENT_SELECTION = "content_selection"
STAGE_ARGUMENT_EXTRACTION = "argument_extraction"
STAGE_DESTINATION = "destination"


def history_snapshot(history) -> dict[str, object]:
    """Availability facts about the turn's chat history (no content copied)."""
    messages = list(history or ())
    has_reply = False
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role, content = msg.get("role"), msg.get("content")
        else:
            role, content = getattr(msg, "role", ""), getattr(msg, "content", "")
        if str(role or "") == "assistant" and str(content or "").strip():
            has_reply = True
            break
    return {
        "has_history": bool(messages),
        "history_messages": len(messages),
        "has_last_assistant_reply": has_reply,
    }


def turn_context(facts, history=()) -> dict[str, object]:
    """The trusted turn facts the Agent needs to continue the dialogue.

    Asset ids and the viewer's page come from the turn's settled facts (never
    from model output); the history is reported by availability only. This is
    the SAME evidence the summary handler attaches to its own failures.
    """
    get = (lambda name, default=None: getattr(facts, name, default)
           if facts is not None else default)
    record: dict[str, object] = {
        "viewer_asset_id": str(get("viewer_asset_id", "") or ""),
        "attachment_asset_id": str(get("attachment_asset_id", "") or ""),
        "path_asset_id": str(get("path_asset_id", "") or ""),
        "viewer_current_page": get("viewer_current_page"),
        "viewer_page_from": get("viewer_page_from"),
        "viewer_page_to": get("viewer_page_to"),
        "viewer_kind": str(get("viewer_kind", "") or ""),
        "viewer_name": str(get("viewer_name", "") or ""),
    }
    record.update(history_snapshot(history))
    return record


@dataclass(frozen=True)
class AcquisitionFailure:
    """A handler-level acquisition exit, carried as data (never raised)."""

    category: str
    stage: str
    detail: str
    scope: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection for :func:`..failure.build_failure`."""
        record: dict[str, Any] = {
            "category": str(self.category),
            "stage": str(self.stage),
            "detail": str(self.detail),
        }
        if self.scope:
            record["scope"] = dict(self.scope)
        if self.context:
            record["context"] = dict(self.context)
        return record

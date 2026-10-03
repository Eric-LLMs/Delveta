"""Context bundle — the deterministic extractor-context assembly (Phase 4 Step 1).

The contract §F/§B: a MODEL acquisition is fed EITHER the current query alone,
OR the current query plus the last ``5`` ``role == "user"`` messages. Never the
assistant/system/Agent history, never the full transcript, never a callback.

When the strategy does not need history (CONTEXT_DIRECT, QUERY_TO_EXTRACTOR), the
history argument is not read at all — no full transcript is ever constructed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contract import SOURCE_CONVERSATION_5_USER_TURNS, SOURCE_QUERY

# The frozen escalation window (§B/§I): the last FIVE user messages, no more.
LAST_N_USER_TURNS = 5


@dataclass(frozen=True)
class ContextBundle:
    """Exactly what a MODEL acquisition may be shown: the current query and,
    only when the strategy requires it, the last-5 user turns (chronological)."""

    query: str
    user_turns: tuple[str, ...] = ()
    source: str = SOURCE_QUERY


def _role(msg) -> str:
    return str((msg.get("role") if isinstance(msg, dict)
                else getattr(msg, "role", "")) or "")


def _content(msg) -> str:
    return str((msg.get("content") if isinstance(msg, dict)
                else getattr(msg, "content", "")) or "").strip()


def user_turns(history, *, limit: int = LAST_N_USER_TURNS) -> tuple[str, ...]:
    """The last ``limit`` ``role == "user"`` message texts, in order. Every
    other role (assistant / system / Agent) is excluded — the only history a
    MODEL acquisition may see is the user's own recent turns."""
    kept = [c for c in (_content(m) for m in (history or ())
                        if _role(m) == "user") if c]
    return tuple(kept[-limit:])


def build(query: str, *, history=(), source: str = SOURCE_QUERY) -> ContextBundle:
    """Assemble the bundle for one strategy. ``source`` is the router's
    ``bundle_source``; only ``CONVERSATION_5_USER_TURNS`` reads history."""
    text = str(query or "")
    if source == SOURCE_CONVERSATION_5_USER_TURNS:
        return ContextBundle(query=text, user_turns=user_turns(history),
                             source=source)
    return ContextBundle(query=text, source=SOURCE_QUERY)

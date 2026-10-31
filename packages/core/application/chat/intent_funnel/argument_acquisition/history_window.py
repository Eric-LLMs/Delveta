"""Deterministic conversation window for the post-funnel verify+extract call.

The post-funnel call resolves anaphora, ellipsis, short follow-ups and
multi-turn parameter supplements, so it must see the recent conversation. It
must never see an unbounded transcript, and it must never ask a model to
summarize one. This module owns the single rule set:

* the user's own messages are kept VERBATIM;
* an assistant message is abbreviated to its head and tail — the head carries
  the topic, the tail carries the last question the user may be answering —
  with an explicit omission marker in between;
* turns are taken from the newest backwards, at most ``MAX_USER_TURNS`` user
  turns, then the OLDEST turns are dropped until the rendered window fits the
  budget;
* abbreviating or dropping anything sets ``truncated``.

The current user message is NOT part of the window: the caller keeps it
verbatim and outside this budget. No model is called here.
"""
from __future__ import annotations

from dataclasses import dataclass

# A reference window, not a hard cap on the caller's query: the query and the
# capability definition are charged separately. 300–500 characters is the
# working range for a supplement/follow-up; the oldest turns are dropped first.
MAX_USER_TURNS = 5
HISTORY_BUDGET_CHARS = 500
ASSISTANT_HEAD_CHARS = 150
ASSISTANT_TAIL_CHARS = 250
OMISSION_TEMPLATE = "\n…[{omitted} characters omitted]…\n"

_ROLES = ("user", "assistant")


def _role(msg) -> str:
    return str((msg.get("role") if isinstance(msg, dict)
                else getattr(msg, "role", "")) or "")


def _content(msg) -> str:
    return str((msg.get("content") if isinstance(msg, dict)
                else getattr(msg, "content", "")) or "").strip()


def _abbreviate(text: str) -> tuple[str, bool]:
    limit = ASSISTANT_HEAD_CHARS + ASSISTANT_TAIL_CHARS
    if len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    return (text[:ASSISTANT_HEAD_CHARS]
            + OMISSION_TEMPLATE.format(omitted=omitted)
            + text[-ASSISTANT_TAIL_CHARS:]), True


@dataclass(frozen=True)
class HistoryWindow:
    """The rendered window plus the facts the caller records in its log."""

    text: str
    chars: int
    truncated: bool
    user_turns: int


def build(history=(), *, budget: int = HISTORY_BUDGET_CHARS,
          max_user_turns: int = MAX_USER_TURNS) -> HistoryWindow:
    """Select and render the recent conversation (chronological order)."""
    messages = [(role, text) for role, text in
                ((_role(m), _content(m)) for m in (history or ()))
                if role in _ROLES and text]

    selected: list[tuple[str, str]] = []
    user_turns = 0
    truncated = False
    for role, text in reversed(messages):
        if role == "user":
            if user_turns >= max_user_turns:
                truncated = True
                break
            user_turns += 1
        selected.append((role, text))
    selected.reverse()

    rendered: list[tuple[str, str]] = []
    for role, text in selected:
        if role == "assistant":
            text, cut = _abbreviate(text)
            truncated = truncated or cut
        rendered.append((role, text))

    def _chars(items: list[tuple[str, str]]) -> int:
        return sum(len(role) + len(text) + 2 for role, text in items)

    while len(rendered) > 1 and _chars(rendered) > budget:
        rendered.pop(0)
        truncated = True

    text = "\n".join(f"{role}: {body}" for role, body in rendered)
    return HistoryWindow(text=text, chars=len(text), truncated=truncated,
                         user_turns=user_turns)

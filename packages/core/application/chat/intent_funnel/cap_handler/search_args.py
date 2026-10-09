"""Shared DET rule for the search handlers — the STATED result count.

The search capabilities (``cap-web-search`` / ``cap-rag-search`` /
``cap-social-search``) own an optional integer slot (``top_k`` / ``limit``). The
ruling for it follows the acquisition waterfall (rules + trusted context FIRST):

* a count the DET rule can read ("5 条", "top 3", "前 5 条") is a RULE-obtained
  value — :func:`result_count` extracts the integer and the handler keeps it;
  the model is never asked to re-derive a rule-obtained value;
* only when the sentence states NO count does the slot fall to the tool's own
  legal default (never a model guess).

This module is that one rule, kept in one place so the three handlers cannot
drift. A stated count is an explicit number bound to a count noun ("5 条",
"3 results") or a "top N" / "前N" form — NOT a bare number that happens to appear
(a year, a version, a technology name like "Python 3" is NOT a result count).
"""
from __future__ import annotations

import re

# A count cue: a number bound to a count noun, or a leading "top N" / "前N" /
# "取N" / "要N" / "返回N" form. A bare number with no such cue never matches.
_COUNT_RE = re.compile(
    r"(?:\btop\s+|\bfirst\s+|\bgive me\s+|\bshow\s+|前|取|要|返回|列出)\s*(\d{1,3})\b"
    r"|(\d{1,3})\s*(?:个|条|项|篇|results?|items?|hits?)\b",
    re.IGNORECASE,
)


def result_count(message: str) -> int | None:
    """The explicit result count stated in ``message``, or ``None``.

    The DET half of the acquisition waterfall: when the sentence states a count
    the handler keeps THIS rule-obtained integer (the model is never asked to
    re-derive it); a message with no count cue returns ``None`` so the slot falls
    to the tool's own legal default."""
    match = _COUNT_RE.search(str(message or ""))
    if match is None:
        return None
    raw = match.group(1) or match.group(2)
    try:
        return int(raw)
    except (TypeError, ValueError):  # pragma: no cover - the regex guarantees digits
        return None


def states_result_count(message: str) -> bool:
    """True when ``message`` states an explicit result count."""
    return result_count(message) is not None

"""Shared DET rule for the search handlers — is a result count STATED?

The search capabilities (``cap-web-search`` / ``cap-rag-search`` /
``cap-social-search``) own an optional integer slot (``top_k`` / ``limit``). The
ruling for it is: a count slot is authorized for MODEL extraction ONLY when the
user's sentence states a count explicitly; otherwise the value stays the tool's
own legal default (never a model guess). This module is that one rule, kept in
one place so the three handlers cannot drift.

A stated count is an explicit number bound to a count noun ("5 条", "3 results")
or a "top N" / "前N" form — NOT a bare number that happens to appear (a year, a
version, a technology name like "Python 3" is NOT a result count).
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


def states_result_count(message: str) -> bool:
    """True when ``message`` states an explicit result count."""
    return bool(_COUNT_RE.search(str(message or "")))

"""Deterministic page-scope RULE — the first half of combined acquisition.

The rule runs BEFORE any model call on every page-addressed file turn. It reads
ONLY the user's sentence and the trusted turn facts, and answers one question:
which pages does this request name explicitly?

* ``page``    — the sentence states a page, a page range or a page list
                (Arabic or Chinese numerals, English ordinals/cardinals,
                ``page 4`` / ``p.3`` / ``3-5 页`` forms);
* ``all``     — the sentence states no page restriction at all (the whole
                document). The viewer's current page is NEVER substituted: a
                whole-document request must not become a current-page request
                merely because a viewer page exists;
* ``unknown`` — the sentence refers to pages it does not spell out (``some
                pages``, ``部分页``) or a page noun carries no resolvable number.
                This is the ONLY state in which the online LLM is REQUIRED to
                supplement the rule; it is never rounded to ``all``.

Authority boundaries:

* the rule never produces or changes ``asset_id`` — identity is a turn fact;
* every page value is rendered through :mod:`.pages_spec`, so the strict tool
  grammar validates it before it can enter a draft;
* the rule never guesses: ``unknown`` is an honest outcome, not a failure to be
  papered over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pages_spec import canonical_list, canonical_range

DECISION_PAGE = "page"
DECISION_ALL = "all"
DECISION_UNKNOWN = "unknown"

# ── number vocabulary ─────────────────────────────────────────────────────────
_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_EN_NUMBERS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_EN_RE = "|".join(sorted(_EN_NUMBERS, key=len, reverse=True))
_NUM_RE = rf"(?:\d+|[一二两三四五六七八九十]+|{_EN_RE})"

# A page noun that is not followed by any resolvable number is an UNKNOWN page
# reference: the LLM must supply what the sentence left implicit.
_AMBIGUOUS_CUES = (
    "some pages", "a few pages", "several pages", "certain pages",
    "部分页", "某些页", "几页", "若干页", "某一页", "某页",
)

# "this page" / "current page" (singular) resolve to the viewer's settled page.
_DEICTIC = (
    "当前页", "当前这页", "当前这一页", "这一页", "这页", "本页",
    "current page", "this page",
)

_PAGE_NOUN = re.compile(r"\bpages?\b|页", re.IGNORECASE)
_BARE_RANGE = re.compile(rf"\b({_NUM_RE})\s*[-–—~]\s*({_NUM_RE})\b", re.IGNORECASE)

# ── explicit forms (checked in this order: most specific first) ────────────────
_RANGE_PATTERNS = (
    # 第1到2页 / 第 1 到第 3 页 / 第2页到第3页
    re.compile(rf"第\s*({_NUM_RE})\s*页?\s*(?:到|至|~|—|–|-)\s*第?\s*({_NUM_RE})\s*页",
               re.IGNORECASE),
    # 2 到 4 页
    re.compile(rf"({_NUM_RE})\s*(?:到|至|~|—|–|-)\s*({_NUM_RE})\s*页",
               re.IGNORECASE),
    # pages 10 through 12 / page 8 to page 10 / read pages 1-3
    re.compile(rf"\bpages?\s*({_NUM_RE})\s*(?:-|–|—|~|to|through)\s*(?:pages?\s*)?({_NUM_RE})\b",
               re.IGNORECASE),
    # p.3-5 / pp. 3–5
    re.compile(rf"\bpp?\.\s*({_NUM_RE})\s*(?:-|–|—|~)\s*({_NUM_RE})\b",
               re.IGNORECASE),
)

# 第 2,4,6 页  (one span, comma/、 list)
_CN_COMMA_SPAN = re.compile(r"第\s*([\d,，、\s]+)\s*页")
# 第 1 页和第 3 页 / 打开第 2 页和第 3 页
_CN_REF = re.compile(rf"第\s*({_NUM_RE})\s*页", re.IGNORECASE)
# pages 3 and 7 / open pages 4 and 6
_EN_LIST = re.compile(
    rf"\bpages?\s*({_NUM_RE})\s*(?:,|and|&)\s*({_NUM_RE})\b", re.IGNORECASE)

_SINGLE_PATTERNS = (
    re.compile(rf"第\s*({_NUM_RE})\s*页", re.IGNORECASE),
    # A bare Chinese numeral + 页 must not match the numeral inside a deictic
    # phrase ("这一页" / "那一页" / "当前一页"): those are resolved by the viewer.
    re.compile(r"(\d+)\s*页", re.IGNORECASE),
    re.compile(r"(?<![这那当])([一二两三四五六七八九十]+)\s*页"),
    re.compile(rf"\bpages?\s*({_NUM_RE})\b", re.IGNORECASE),
    re.compile(rf"\bpp?\.\s*({_NUM_RE})\b", re.IGNORECASE),
    re.compile(rf"\b({_NUM_RE})\s+page\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RuleScope:
    """The deterministic verdict. ``pages`` is a canonical tool spec or ``None``."""

    decision: str = DECISION_UNKNOWN
    pages: str | None = None
    reason: str = ""

    @property
    def decided(self) -> bool:
        return self.decision != DECISION_UNKNOWN

    def record(self) -> dict[str, object]:
        return {"decision": self.decision, "pages": self.pages,
                "reason": self.reason}


def _number(token: str) -> int | None:
    tok = str(token or "").strip().lower()
    if not tok:
        return None
    if tok.isdigit():
        return int(tok)
    if tok in _EN_NUMBERS:
        return _EN_NUMBERS[tok]
    if tok in _CN_DIGITS:
        return _CN_DIGITS[tok]
    # 十一..十九 / 二十..九十 (the dataset needs 十二 at most)
    if len(tok) == 2 and tok[0] == "十" and tok[1] in _CN_DIGITS:
        return 10 + _CN_DIGITS[tok[1]]
    if len(tok) == 2 and tok[1] == "十" and tok[0] in _CN_DIGITS:
        return _CN_DIGITS[tok[0]] * 10
    if len(tok) == 3 and tok[1] == "十" and tok[0] in _CN_DIGITS and tok[2] in _CN_DIGITS:
        return _CN_DIGITS[tok[0]] * 10 + _CN_DIGITS[tok[2]]
    return None


def _range_page(start: str, end: str) -> str | None:
    a, b = _number(start), _number(end)
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    try:
        return canonical_range(a, b)
    except ValueError:
        return None


def _list_pages(*tokens: str) -> str | None:
    nums = [_number(t) for t in tokens]
    if any(n is None or n <= 0 for n in nums):
        return None
    try:
        return canonical_list(nums)
    except ValueError:
        return None


def _single_page(token: str) -> str | None:
    number = _number(token)
    if number is None or number <= 0:
        return None
    try:
        return canonical_list([number])
    except ValueError:
        return None


def _viewer_page(facts) -> int | None:
    raw = getattr(facts, "viewer_current_page", None) if facts is not None else None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _unknown(reason: str) -> RuleScope:
    return RuleScope(decision=DECISION_UNKNOWN, pages=None, reason=reason)


def rule_scope(query: str, facts=None) -> RuleScope:
    """Classify the sentence's stated page scope. Pure and deterministic."""
    q = str(query or "").lower()
    if not q.strip():
        return _unknown("the request is empty")

    # 1) explicit range — the most specific form
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(q)
        if match:
            pages = _range_page(match.group(1), match.group(2))
            if pages:
                return RuleScope(DECISION_PAGE, pages, f"explicit range {pages!r}")
            return _unknown("a page range was named but its bounds are not valid pages")

    # 2) explicit list (Chinese comma span, repeated 第N页, English "and")
    match = _CN_COMMA_SPAN.search(q)
    if match and re.search(r"[,，、]", match.group(1)):
        nums = [int(x) for x in re.findall(r"\d+", match.group(1))]
        if nums:
            pages = _list_pages(*nums)
            if pages:
                return RuleScope(DECISION_PAGE, pages, f"explicit page list {pages!r}")
            return _unknown("a page list was named but it is not a valid spec")
    refs = _CN_REF.findall(q)
    if len(refs) >= 2:
        pages = _list_pages(*refs)
        if pages:
            return RuleScope(DECISION_PAGE, pages, f"explicit page list {pages!r}")
        return _unknown("several page references were named but they are not valid pages")
    match = _EN_LIST.search(q)
    if match:
        pages = _list_pages(match.group(1), match.group(2))
        if pages:
            return RuleScope(DECISION_PAGE, pages, f"explicit page list {pages!r}")
        return _unknown("a page list was named but it is not a valid spec")

    # 3) one explicit page
    for pattern in _SINGLE_PATTERNS:
        match = pattern.search(q)
        if match:
            pages = _single_page(match.group(1))
            if pages:
                return RuleScope(DECISION_PAGE, pages, f"explicit page {pages!r}")
            return _unknown("a page was named but its number is not a valid page")

    # 4) deictic: the viewer's settled current page, and ONLY that
    if any(marker in q for marker in _DEICTIC):
        page = _viewer_page(facts)
        if page is not None:
            return RuleScope(DECISION_PAGE, str(page),
                             f"deictic page -> viewer page {page}")
        return _unknown("the sentence refers to the current page, but no viewer page is set")

    # 5) an unspelled page reference: never widens to the whole document
    if any(cue in q for cue in _AMBIGUOUS_CUES):
        return _unknown("the sentence refers to pages it does not spell out")
    if _PAGE_NOUN.search(q) and _BARE_RANGE.search(q):
        return _unknown("a numeric range is attached to a page reference without a page number")

    # 6) no page restriction stated -> the whole document
    return RuleScope(DECISION_ALL, None, "no page restriction was stated")

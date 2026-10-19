"""formal500_rules — the generation rules the Formal-500 v2 builder applies.

Kept in the tracked PE scripts tree (not in ``logs/``) so the rules are unit
tested in CI: a tracked test may not import an untracked ``logs/`` module, and a
generation rule that silently drifts is exactly what produced the v1 drift.

Two rules live here:

* :func:`strip_container` — a ``domain`` value must be the bare domain NAME, not
  the container phrase. ``工程词汇库 -> 工程``,
  ``machine learning vocabulary -> machine learning``, ``信息论词库 -> 信息论``.
  The name is kept in the sentence's own language: no translation, no slug
  (``pe_contract`` add-term/rag ``domain`` = "the vocabulary domain";
  ``docs/architecture.md`` §25.16.10 — the literal name, never slugified).
* :func:`query_is_verbatim` — the gold ``query`` must be a verbatim span of the
  user's sentence (``pe_contract`` ``_SEARCH_QUERY_DESC``: "kept VERBATIM — do not
  rewrite, translate, summarize, expand or paraphrase").

Stdlib only.
"""
from __future__ import annotations

import re

# Container words that name a container, not a domain. Longest-first so a longer
# suffix is never eaten by a shorter one (``词汇表`` before ``词表``).
_ZH_SUFFIX = ["词汇库", "词汇表", "词汇本", "术语库", "术语表", "生词本", "词库", "词表", "词典", "词汇"]
_EN_SUFFIX = ["vocabulary", "glossary", "lexicon", "wordlist", "terms", "dictionary"]
_CONTAINER_TAIL = re.compile(
    r"[\s\u3000]*(" + "|".join(re.escape(s) for s in _EN_SUFFIX + _ZH_SUFFIX) + r")$",
    re.IGNORECASE,
)

# Function words / verbs that sit in front of a container word but are NOT a
# domain name. A capture equal to one of these must never be promoted to a label.
DOMAIN_STOPWORDS = {
    "加到", "加进", "加", "收进", "收录", "添加", "放入", "放进", "保存", "存在",
    "到", "进", "去",
    "the", "a", "an", "this", "that", "these", "those", "my", "our", "your",
    "to", "into", "in", "on", "at", "of", "for", "it", "word", "term", "terms",
}


def strip_container(value: str) -> str | None:
    """The bare domain name from a container-phrased surface, or ``None`` when
    nothing changed or stripping would empty the value (never destroy a value)."""
    if not isinstance(value, str):
        return None
    stripped = _CONTAINER_TAIL.sub("", value).strip()
    if not stripped or stripped == value:
        return None
    return stripped


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _unquote(s: str) -> str:
    for q in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019"):
        s = s.replace(q, "")
    return s


def query_is_verbatim(user_query: str, gold_query: str) -> bool:
    """True when ``gold_query`` (quotes/whitespace normalized) is a span of the
    user's sentence — i.e. the gold topic was copied, not rewritten."""
    return _norm(_unquote(gold_query)) in _norm(_unquote(user_query))

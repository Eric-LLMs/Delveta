"""AddTermHandler — argument acquisition for ``cap-add-term``.

Owns ONLY this capability's public schema slots, ``term`` and ``domain``.
``cap-add-term`` (tool binding ``add_term``) inserts ONE word into a vocabulary
domain addressed BY NAME (``VocabularyService.add_term``). The tool resolves the
domain by an EXACT case-insensitive name match over the caller's visible domains
(0 → "not found", >1 → "ambiguous"), so the ``domain`` value must be the user's
literal domain NAME — copied from the sentence, never translated, slugified or
otherwise "understood".

Both slots are resolved DET-FIRST (the rule ladder below is the authoritative
source when it reliably names the value); a slot the DET rules do NOT obtain is
authorized to the unified local-Qwen slot extractor via ``slot_plan()`` — a
semantic gap, never a bail to the Agent. The model value is only folded in after
``_certify``'s SAME kind gate + Binder validate; a model value is NEVER allowed to
overwrite a DET-obtained one (an ACQUIRED slot is truth), and a slot the model
also cannot resolve honestly lands MISSING (the Binder gate, then the Agent).

``term`` — the word being added — comes from a literal span, first hit wins:

1. a QUOTED span — ASCII ``"…"`` / ``'…'`` or the full-width ``“…”`` / ``‘…’`` /
   ``「…」`` / ``『…』`` pairs (the enclosing quotes are dropped, the inside copied
   verbatim and edge-trimmed);
2. an unquoted Chinese term — the token between ``把`` / ``将`` and the trailing
   insert verb (``把 arch 这个词收进 …`` → ``arch``; ``把 熵 加进 …`` → ``熵``);
3. an unquoted English term — the single token after an insert verb
   (``add`` / ``put`` / ``record`` / ``save`` / ``insert``) and before ``to`` /
   ``in`` / ``into``.

A deictic reference with no antecedent (``这个词`` / ``this word``) is NOT a term:
with no literal span, the term is MISSING.

``domain`` — the named vocabulary — is located by structure and copied VERBATIM:

1. the Chinese ``在 X 里加入`` clause (``在 算法 词库里加入 …`` → ``算法 词库``);
2. the Chinese verb-trailing clause (``… 加到 金融 词汇库`` → ``金融 词汇库``);
3. the English ``… to|in|into [determiner] X`` clause.

The region must carry a vocabulary indicator (``词库`` / ``词汇库`` / ``术语表`` /
``glossary`` / ``vocabulary`` / ``terms`` …) — without one it is not a domain.
Normalization is INTENTIONALLY minimal: strip a leading determiner
(``my`` / ``the`` / ``我的`` …), strip a trailing bare ``词库`` token and a
trailing bare English ``domain`` word. The generic carrier words are otherwise
KEPT whole when a modifier precedes them (``金融 词汇库`` keeps its space,
``machine learning vocabulary`` stays intact). A region that is ONLY a generic
carrier (``词库`` / ``glossary``) names no domain.

Draft contract (rule-1 DET-first, rule-2 model for the gap):

* a DET-obtained value is ACQUIRED and kept; only the slots DET did NOT obtain are
  handed to the model. So ``acquire()`` returns whatever the rules resolved — both,
  one, or NONE — and ``slot_plan()`` authorizes exactly the unfilled subset.
* ``acquire()`` returns ``{}`` (an EMPTY draft, NOT ``None``) when the rules name
  neither slot — that empty draft still reaches the orchestrator, whose plan
  authorizes both slots to the model. A DET miss never short-circuits the turn
  before the model runs; only a BLANK query (no input at all) returns ``None``.
* a required slot neither the rules nor the model resolves stays MISSING — the
  Binder gate owns it (never fabricated here).

``definition`` is NEVER emitted (tool-owned, optional, no producer — an emitted
value would be a fabrication).

``facts`` is accepted for the common handler contract but ignored ON PURPOSE: both
values come from the sentence, never from the turn's asset context.
"""
from __future__ import annotations

import re

from .slot_plan import SlotPlan

# ── term rule 1: quoted spans — the quotes are dropped, the inside copied ─────────
_QUOTED_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
)

# ── term rule 2: unquoted Chinese — the token between 把/将 and the insert verb ────
_ZH_TERM_RE = re.compile(
    r"(?:把|将)\s*([^\s，,。.！!？?]+?)\s*(?:这个|那个)?词?\s*"
    r"(?:加进|加入|加到|添加到|添到|收录到|收进|录入|存入|放进|放入|保存到"
    r"|添加|收录|写进|写)"
)

# ── term rule 3: unquoted English — one token after an insert verb, before to/in ──
_EN_TERM_RE = re.compile(
    r"\b(?:add|put|record|save|insert)\s+(?:the\s+)?(?:(?:word|term|token)\s+)?"
    r"([A-Za-z][A-Za-z0-9.+\-]*)\s+(?:into|to|in)\b",
    re.IGNORECASE,
)

# ── domain rule 1: Chinese "在 X 里加入" — the domain precedes the verb ────────────
_ZH_DOMAIN_BEFORE_RE = re.compile(
    r"在\s*(.+?)\s*里(?:面)?\s*(?:加入|加进|添加|录入|收录|加)"
)

# ── domain rule 2: Chinese verb-trailing — the domain follows the insert verb ─────
_ZH_DOMAIN_AFTER_RE = re.compile(
    r"(?:加进|加入|加到|添加到|添到|收录到|收进|录入|存入|放进|放入|保存到"
    r"|添加|收录|写进)\s*(.+)$"
)

# ── domain rule 3: English "… to|in|into [determiner] X" ─────────────────────────
_EN_DOMAIN_RE = re.compile(
    r"\b(?:into|to|in)\s+(?:my|the|our|your|a|an)\s+(.+)$",
    re.IGNORECASE,
)

# A deictic term with no antecedent — never a literal word.
_DEICTIC_TERM = frozenset(
    {"this", "that", "it", "this word", "this term", "the word", "the term",
     "a word", "word", "term", "词", "这个词", "那个词", "这个", "那个"}
)

# A region with none of these carries no vocabulary domain. Checked against the
# RAW region (before carrier stripping), case-insensitively for the English set.
_VOCAB_INDICATORS_ZH = ("词库", "词汇库", "词汇本", "术语库", "术语表", "生词本",
                        "单词库", "词汇", "词典", "词表")
_VOCAB_INDICATORS_EN = ("vocabulary", "glossary", "terms", "term", "word list",
                        "lexicon", "domain")

# A region that is ONLY a generic carrier names no domain (fail-closed).
_GENERIC_DOMAIN_EXACT = frozenset(
    {"词库", "词汇库", "词汇本", "术语库", "术语表", "生词本", "单词库", "词汇",
     "词典", "词表", "glossary", "vocabulary", "terms", "term", "word list",
     "lexicon", "domain", "list"}
)

_EDGE_PUNCT = " \t\r\n。.!?，,;；:：\"'“”‘’「」『』"


def _quoted(message: str) -> str | None:
    for open_q, close_q in _QUOTED_PAIRS:
        start = message.find(open_q)
        if start == -1:
            continue
        end = message.find(close_q, start + 1)
        if end > start + 1:
            return message[start + 1:end].strip() or None
    return None


def _term(message: str) -> str | None:
    quoted = _quoted(message)
    if quoted:
        return quoted
    for pattern in (_ZH_TERM_RE, _EN_TERM_RE):
        match = pattern.search(message)
        if match:
            return match.group(1)
    return None


def _clean_term(term: str | None) -> str | None:
    """Trim edge punctuation; reject empty and deictic (no-antecedent) values."""
    if term is None:
        return None
    term = term.strip(_EDGE_PUNCT).strip()
    if not term:
        return None
    if term.lower() in _DEICTIC_TERM:
        return None
    return term


def _has_indicator(region: str) -> bool:
    if any(ind in region for ind in _VOCAB_INDICATORS_ZH):
        return True
    low = region.lower()
    return any(ind in low for ind in _VOCAB_INDICATORS_EN)


def _clean_domain(region: str | None) -> str | None:
    """Verbatim domain name: strip a leading determiner, a trailing bare ``词库``
    token and a trailing bare English ``domain``; reject an indicator-less region
    or a bare generic carrier."""
    if region is None:
        return None
    region = region.strip(_EDGE_PUNCT).strip()
    if not region or not _has_indicator(region):
        return None
    region = re.sub(r"^(?:我的|我|你的|这个|那个|这些|那些)\s*", "", region)
    region = re.sub(r"^(?:my|the|our|your|a|an)\s+", "", region, flags=re.IGNORECASE)
    region = re.sub(r"\s*\bdomain\b\s*$", "", region, flags=re.IGNORECASE)
    region = re.sub(r"\s+词库$", "", region)  # only a whitespace-separated carrier
    region = region.strip(_EDGE_PUNCT).strip()
    if not region or region.lower() in _GENERIC_DOMAIN_EXACT:
        return None
    return region


def _domain(message: str) -> str | None:
    for pattern in (_ZH_DOMAIN_BEFORE_RE, _ZH_DOMAIN_AFTER_RE, _EN_DOMAIN_RE):
        match = pattern.search(message)
        if match:
            return _clean_domain(match.group(1))
    return None


class AddTermHandler:
    """``cap-add-term`` parameter handler (stateless).

    ``facts`` is accepted for the common handler contract but ignored ON PURPOSE:
    ``term`` and ``domain`` both come from the sentence, never from the turn's
    asset context — so no viewer/attachment id can leak in as a term or domain.
    """

    capability_id = "cap-add-term"

    async def acquire(self, *, query: str, facts) -> dict[str, object] | None:
        """Return the ``add_term`` DET draft — whichever of ``term`` / ``domain``
        the rules reliably resolved (both, one, or the empty ``{}``).

        A DET miss is NOT a bail: the empty ``{}`` draft still reaches the
        orchestrator, whose ``slot_plan`` hands the unfilled slots to the model.
        Only a blank query returns ``None`` (no input at all)."""
        message = str(query or "").strip()
        if not message:
            return None
        draft: dict[str, object] = {}
        term = _clean_term(_term(message))
        if term is not None:
            draft["term"] = term
        domain = _domain(message)
        if domain is not None:
            draft["domain"] = domain
        return draft

    def slot_plan(self, *, query: str, facts, draft: dict) -> SlotPlan:
        """Authorize the model ONLY for the slots the DET rules did NOT reliably
        obtain. A DET-obtained value is ACQUIRED and never re-asked (an
        unauthorized slot can never be overwritten); a slot the model also cannot
        resolve honestly stays MISSING for the Binder gate. Never a fabricated
        ``term`` / ``domain``."""
        model_slots = tuple(s for s in ("term", "domain") if s not in (draft or {}))
        return SlotPlan(model_slots=model_slots)


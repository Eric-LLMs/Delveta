"""Understanding Engine: the typed answer to "what does this turn need?".

``TurnRequirements`` is the OUTPUT contract; its owner is this module. It describes
CAPABILITY DEMAND only — it has no knowledge of executors, policies or plans (the
policy layer maps demand to a plan; the executors carry it out).

:func:`resolve_source_facts` derives ONLY the turn's source-restriction facts from
cheap lexical patterns. It never selects a route: capability selection belongs to the
Intent Funnel (a Matcher HIT certifies a table entry; a MISS goes through Recall and
LayaChoice-v2). The facts exist to fence the Agent's tools after an escalation
(``private_only`` / ``external_ok``) and to keep the half-execution guard for
certified web turns (``needs_web``).

All L0 intent-routing patterns — direct-tool recognition, web/private/action demand
classification, viewer eligibility, memory triggers and sequencing checks — have
been removed. Non-certified turns carry the default ``ABSTAIN`` confidence, so the
policy mapper sends them to the Agent while the funnel owns every selection attempt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Signal(str, Enum):
    """Certainty of one capability demand. AMBIGUOUS forces L1/agent arbitration."""

    HIGH = "high"
    LOW = "low"
    AMBIGUOUS = "ambiguous"


class Confidence(str, Enum):
    """Overall routing confidence of the requirement set."""

    HIGH = "high"
    LOW = "low"
    AMBIGUOUS = "ambiguous"
    ABSTAIN = "abstain"


class Complexity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass(frozen=True)
class TurnRequirements:
    """What capabilities the user's turn demands (never how to execute them).

    ``requested_action`` carries the *semantic intent* of an action request (a name +
    raw slot hints), NOT an instantiated ActionSpec — validation and authorization
    happen in Pre-flight / the ACTION executor, never here. Only the funnel's
    ``certified()`` sets it; the chat layer no longer recognises actions lexically.
    """

    needs_private: Signal = Signal.LOW
    needs_web: Signal = Signal.LOW
    needs_viewer: Signal = Signal.LOW
    needs_action: Signal = Signal.LOW
    needs_memory: bool = False
    requested_action: dict | None = None
    complexity: Complexity = Complexity.LOW
    confidence: Confidence = Confidence.ABSTAIN
    # SOURCE FACTS (independent of routing): how the ORIGINAL request restricts the
    # answer's sources. ``private_only`` = explicit "knowledge base only / no web"
    # restriction; ``external_ok`` = explicit permission to supplement with external
    # sources when the corpus falls short. Source policy comes from user intent —
    # never from the fact that a fast path failed. Detection is deliberately narrow:
    # a false positive would silently fence the Agent in.
    private_only: bool = False
    # Explicit user PERMISSION to supplement with external sources when the corpus
    # falls short ("如果查不到可以搜网络…"). Only this re-opens the web on an
    # escalated private-first turn; without it the escalated answer must disclose
    # the corpus gap honestly and stay on private sources.
    external_ok: bool = False
    # DEFERRED CAPTURE (funnel-certified vision only): the picture the turn needs is
    # not on the server — the PC client must capture the current page / video frame.
    # ``None`` on every ordinary turn; its presence means the ACTION must pause until
    # the client supplies the asset. It carries no execution authority: the certified
    # capability/tool/kind already live in ``requested_action``.
    capture_request: dict | None = None


# ── source facts (no routing authority) ────────────────────────────────────────────

# Anything time-sensitive or world-current belongs on the Agent path (which owns the
# ``web_search`` tool), never on a tool-less direct answer. Deliberately narrow: a
# false "needs_web" only costs the fast path, it never misroutes into a stale answer.
# NOTE ( shadow-A/B finding): the CJK alternatives ride WITHOUT the \b
# wrapper — between two CJK chars there is no word boundary, so "\b新闻\b" never
# matches "重要新闻吗" and every mid-sentence Chinese web cue was missed.
_WEB_PAT = re.compile(
    r"\b(today|yesterday|now|current(ly)?|latest|recent(ly)?|this week|this month|"
    r"this year|news|weather|score|price|stock|version|as of)\b"
    r"|今天|昨天|现在|目前|最新|最近|本周|本月|今年|新闻|天气|比分|股价|版本",
    re.IGNORECASE,
)
# Explicit SOURCE RESTRICTION: answer from the private corpus only / no external
# sources this turn. Narrow on purpose (see TurnRequirements.private_only).
_PRIVATE_ONLY_PAT = re.compile(
    r"(?:\b(?:only|just)\s+(?:use|from|based\s+on|answer(?:\s+from)?)\b[^.?;]{0,40}"
    r"\b(?:my|the)\s+\w*\s*(?:knowledge\s*base|knowledgebase|kb|corpus|library|notes?|documents?|files?)\b)"
    r"|(?:\b(?:without|no|don'?t|do\s+not|never)\s+(?:using\s+|calling\s+|the\s+|search\s+)?(?:web|internet|online|external)\b)"
    r"|(?:只[用从靠按查][^。;？!]{0,12}(?:知识库|文档|笔记|资料|文件))"
    r"|(?:不要|别|禁止|不准|不可|不得)[^。;？!]{0,6}(?:联网|上网|搜网|查网|用网络|网络搜索|外部资料|外部)",
    re.IGNORECASE,
)
# Explicit permission to fall back to external sources when the private corpus is
# insufficient — the ONLY thing that keeps web access on an escalated private-first
# turn (default after a private-first failure is: disclose honestly, stay private).
_ALLOW_EXTERNAL_PAT = re.compile(
    r"(?:\b(?:can|may|could|feel\s+free\s+to|otherwise)\b[^.?;]{0,30}"
    r"\b(?:search|check|look\s*up|use)\b[^.?;]{0,20}\b(?:the\s+)?(?:web|internet|online)\b)"
    r"|(?:可以|也可以|允许|不妨)[^。;？!]{0,8}(?:查|搜|联网|上网|网络)"
    r"|(?:没有|找不到|不足|不够)[^。;？!]{0,8}(?:就|再|可以)?[^。;？!]{0,4}(?:查|搜)[^。;？!]{0,6}(?:网络|网上|互联网)",
    re.IGNORECASE,
)


def resolve_source_facts(message: str) -> TurnRequirements:
    """Derive ONLY the source-restriction facts of one turn.

    This is not a classifier and never selects a capability: the returned requirements
    stay at the default ``ABSTAIN`` confidence (and every demand LOW), so the policy
    mapper hands the turn to the funnel/Agent as before. Routing starts at the
    Matcher; only its certification or the cascade's recall+selection can produce a
    HIGH-confidence requirement set.
    """
    text = message or ""
    needs_web = Signal.HIGH if _WEB_PAT.search(text) else Signal.LOW
    return TurnRequirements(
        needs_web=needs_web,
        # An explicit restriction that CO-OCCURS with a web demand is a contradiction,
        # not a fence — abstain rather than silently deny the Agent its network tools.
        private_only=(
            _PRIVATE_ONLY_PAT.search(text) is not None
            and needs_web is not Signal.HIGH
        ),
        external_ok=_ALLOW_EXTERNAL_PAT.search(text) is not None,
    )

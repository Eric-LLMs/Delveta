"""Funnel common layer — the system-wide safety guards (ruling 8.1-a).

These guards are deliberately CODE, not Registry data: the configuration table
is "meant to be read and edited by humans" and does not fit negation/context
veto logic. Nodes never duplicate them — :func:`funnel.route` applies
:func:`turn_veto` once before the cascade and the Matcher applies
:func:`negated` before certifying a HIT, exactly the two-call-site discipline
the legacy action layer already proved.
"""
from __future__ import annotations

import re

from core.application.chat.actions import is_negated_request
from core.application.chat.sanitization import is_pure_user_text
from core.application.chat.understanding import Signal

# Entry veto "referenced_input_absent" (shadow-A/B follow-up 2026-09-27, suspects
# a403c4b341e1 / d795e47fe617): a turn whose input OBJECT is named only by a
# demonstrative ("这份笔记 / 这个术语 / 这些资料") demands content that lives
# on-screen (viewer/attachment) or in the conversation — without either, the
# target simply is not there and only the Agent can ask the user for it. This
# matters because Registry standard queries are DELIBERATELY deictic
# ("把这份笔记做成思维导图" is a cap-mindmap sentence): the lift must come from
# CONTEXT FACTS (viewer/attach present) or an EXPLICIT inline input (the object
# quoted in the message), never from editing the table (ruling 8.1-a) and never
# from the Binder or the model.
_ONSCREEN_REF_PAT = re.compile(
    r"(?:这个|这份|这篇|这条|这些)[^。;？!]{0,10}"
    r"(?:笔记|文档|文件|论文|资料|内容|术语|词条|报告|稿子|条款|段落|句子|页面|网页|文章|链接|选中|词)"
    r"|(?:this|these)\s+(?:above\s+)?"
    r"(?:note|documents?|files?|papers?|terms?|entries|reports?|passages?|pages?"
    r"|articles?|links?|selections?|content)",
    re.IGNORECASE,
)
# The object quoted inside the message IS the explicit input — no veto.
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))


def _has_explicit_span(message: str) -> bool:
    return any(a in message and b in message for a, b in _QUOTE_PAIRS)


def turn_veto(message: str, requirements, ctx) -> str | None:
    """Prefixed reason when the funnel must not own this turn; None = proceed.

    Mirrors the frozen gate semantics (design §4.3 zero-pollution + ruling a):
    web/memory demand belongs to the Agent's planning loop, research/handoff
    turns are inherently multi-step chains, and non-pure text (attachment
    markers, control payloads) must never be pattern-matched as user intent."""
    if not is_pure_user_text(message or ""):
        return "input_not_pure_text"
    if requirements.needs_web is not Signal.LOW or requirements.needs_memory:
        return "turn_demands_web_or_memory"
    if getattr(ctx, "research_turn", False) or getattr(ctx, "effective_handoff", None):
        return "context_research_or_handoff"
    # Deictic-input veto LAST: an existing veto keeps its (more specific) reason;
    # this one only fires for turns that passed every guard but name their input
    # object nowhere except on a screen/conversation that is not present.
    if _ONSCREEN_REF_PAT.search(message or "") and not _has_explicit_span(message or ""):
        body = getattr(ctx, "body", None)
        viewer = getattr(ctx, "viewer_assembly", None) or getattr(body, "viewer", None)
        attach = getattr(body, "attach", None)
        if viewer is None and not attach:
            return "referenced_input_absent"
    return None


def negated(message: str) -> bool:
    """The single global negation guard (8.1-a): "不要新建文件夹" is not a request."""
    return is_negated_request(message or "")

"""Per-capability extractor prompt registry (Phase-1 unified contract).

The extractor's DEFAULT system prompt (:data:`..extractor.SYSTEM_EXTRACT`) is
unchanged — a capability with no entry here is extracted exactly as before, so
adding this registry is backward-compatible by construction.

A capability whose extraction discipline DIFFERS from the default registers its
own prompt here. The three search tools are the first such case: their ``query``
must be a CLEANED search TOPIC (the user's instruction frame stripped), whereas
the default prompt copies each slot value literally — which is why the old
default measured only 9.1% on ``query``.

Selection is by ``capability_id`` and happens in the unified orchestrator (never
in a handler): ``prompt_for(entry.capability_id)`` is passed to
:func:`..extractor.extract`.
"""
from __future__ import annotations

from .extractor import SYSTEM_EXTRACT

# The search-capability system prompt. Its ONE job beyond the default: turn the
# user's sentence into a clean search TOPIC, while every optional slot is filled
# ONLY when the sentence states it explicitly (never invented).
SEARCH_QUERY_PROMPT = (
    "You fill the argument slots for ONE already-chosen search tool call. The "
    "tool is fixed: never choose, change or question it.\n"
    "Rules for each slot you are asked to fill:\n"
    "- query: the SEARCH TOPIC — a clean, self-contained phrase distilled from "
    "the user's sentence. Remove the instruction frame, politeness and language "
    "directives (e.g. \"帮我搜一下\", \"search for\", \"look up\", \"查一下\", "
    "\"帮我找\"), but keep the user's own wording and language: do NOT translate, "
    "expand, summarise or invent. If the topic is already clean, copy it as-is.\n"
    "- top_k / limit: the result count, ONLY when the sentence states a count "
    "explicitly (e.g. \"前 5 条\", \"top 3\", \"10 results\"); otherwise OMIT it.\n"
    "- domain: a scope, ONLY when the sentence names one; otherwise OMIT it.\n"
    "- platform: ONLY when the sentence names one (reddit / x / zhihu); "
    "otherwise OMIT it.\n"
    "OMIT any slot with no value. NEVER invent, guess or fabricate a value.\n"
    'Respond with a single JSON object and no other text: '
    '{"arguments": {<slot>: <value>}}.\n'
    'Example: sentence "帮我搜一下注意力机制的最新进展" with slots query, top_k '
    '-> {"arguments": {"query": "注意力机制的最新进展"}}.\n'
    'Example: sentence "search for python asyncio best practices, top 3" with '
    'slots query, top_k -> '
    '{"arguments": {"query": "python asyncio best practices", "top_k": 3}}.'
)

# capability_id -> system prompt. Absent -> the default (backward-compatible).
_PROMPTS: dict[str, str] = {
    "cap-web-search": SEARCH_QUERY_PROMPT,
    "cap-rag-search": SEARCH_QUERY_PROMPT,
    "cap-social-search": SEARCH_QUERY_PROMPT,
}


def prompt_for(capability_id: str) -> str:
    """The system prompt for ``capability_id``, or the extractor default when the
    capability registers none."""
    return _PROMPTS.get(str(capability_id or ""), SYSTEM_EXTRACT)

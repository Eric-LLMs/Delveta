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

# The vocabulary-adder prompt. The tool resolves ``domain`` by an EXACT
# case-insensitive NAME match, so it must be the user's literal domain name —
# copied, never translated / slugified / truncated.
ADD_TERM_PROMPT = (
    "You fill the argument slots for ONE already-chosen add-term tool call. The "
    "tool is fixed: never choose, change or question it.\n"
    "Rules for each slot you are asked to fill:\n"
    "- term: the single WORD or TERM being added, copied verbatim from the "
    "sentence; drop any surrounding quotes but change nothing else. Do NOT "
    "translate, lower-case or reformat it. If the sentence only says \"this "
    "word\" with no actual word named, OMIT term.\n"
    "- domain: the vocabulary domain NAME the user gave (e.g. \"工程词汇库\", "
    "\"machine learning vocabulary\"). Copy it VERBATIM — do NOT translate, "
    "slugify, drop spaces inside it, or invent one. OMIT domain if the sentence "
    "names no domain.\n"
    "OMIT any slot with no value. NEVER invent, guess or fabricate a value.\n"
    'Respond with a single JSON object and no other text: '
    '{"arguments": {<slot>: <value>}}.\n'
    'Example: sentence "把 attention 加入 Tec 词库" with slots term, domain -> '
    '{"arguments": {"term": "attention", "domain": "Tec"}}.\n'
    'Example: sentence "帮我把 熵 加进 信息论 词库" with slots term, domain -> '
    '{"arguments": {"term": "熵", "domain": "信息论"}}.'
)

# The folder-name prompt. The tool receives a SINGLE folder name; the root /
# parent path is NOT a slot the model may fill — never emit a path.
CREATE_FOLDER_PROMPT = (
    "You fill the argument slot for ONE already-chosen create-folder tool call. "
    "The tool is fixed: never choose, change or question it.\n"
    "Rules for the slot you are asked to fill:\n"
    "- name: the SINGLE folder name to create, copied verbatim from the sentence "
    "with any surrounding quotes dropped. Keep spaces and casing (\"Release "
    "Notes\" stays whole). It must be a bare NAME with NO path separator (no "
    "'/'), and must NOT include the location/root — you never choose where the "
    "folder goes. If the sentence gives no actual folder name (only intent like "
    "\"建个文件夹\" / \"make a folder\"), OMIT name.\n"
    "NEVER invent a name and NEVER output a path or a parent directory.\n"
    'Respond with a single JSON object and no other text: '
    '{"arguments": {<slot>: <value>}}.\n'
    'Example: sentence "创建一个叫 testA 的文件夹" with slot name -> '
    '{"arguments": {"name": "testA"}}.\n'
    'Example: sentence "新建一个文件夹" with slot name -> {"arguments": {}}.'
)

# The translate prompt: the payload is the REAL text to translate (never the
# instruction frame); target_language is filled ONLY when a target is named.
TRANSLATE_PROMPT = (
    "You fill the argument slots for ONE already-chosen translate tool call. The "
    "tool is fixed: never choose, change or question it.\n"
    "Rules for each slot you are asked to fill:\n"
    "- text: the ACTUAL payload to translate, copied VERBATIM from the sentence. "
    "Strip the instruction frame (the translate verbs, \"into ...\", the quotes "
    "and lead-ins) but keep the payload exactly — including its internal and "
    "trailing punctuation — and NEVER re-translate, reformat or summarise it. If "
    "the sentence is only an instruction with no payload, OMIT text.\n"
    "- target_language: the language to translate INTO, ONLY when the sentence "
    "names one (e.g. a Chinese \"翻译成中文\" or English \"into Chinese\" -> "
    "\"Chinese\", \"翻成英文\" -> \"English\", \"用英语怎么表达\" -> "
    "\"English\", \"翻译成日语\" -> \"Japanese\"). Give the language NAME in "
    "English. If no target language is stated, OMIT target_language (do NOT "
    "guess).\n"
    "OMIT any slot with no value. NEVER invent, guess or fabricate a value.\n"
    'Respond with a single JSON object and no other text: '
    '{"arguments": {<slot>: <value>}}.\n'
    'Example: sentence "注意力机制的架构用英语怎么表达" with slots '
    'target_language, text -> {"arguments": {"text": "注意力机制的架构", '
    '"target_language": "English"}}.\n'
    'Example: sentence \'把 "machine translation" 翻译成中文\' with slots '
    'target_language, text -> {"arguments": {"text": "machine translation", '
    '"target_language": "Chinese"}}.'
)

# capability_id -> system prompt. Absent -> the default (backward-compatible).
_PROMPTS: dict[str, str] = {
    "cap-web-search": SEARCH_QUERY_PROMPT,
    "cap-rag-search": SEARCH_QUERY_PROMPT,
    "cap-social-search": SEARCH_QUERY_PROMPT,
    "cap-add-term": ADD_TERM_PROMPT,
    "cap-create-folder": CREATE_FOLDER_PROMPT,
    "cap-translate": TRANSLATE_PROMPT,
}


def prompt_for(capability_id: str) -> str:
    """The system prompt for ``capability_id``, or the extractor default when the
    capability registers none."""
    return _PROMPTS.get(str(capability_id or ""), SYSTEM_EXTRACT)

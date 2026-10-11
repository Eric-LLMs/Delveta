"""The frozen parameter-extraction contract (single source of truth).

Both the builder (:mod:`prepare_dataset`) and the verifier (:mod:`validate_dataset`)
read this module, so the capability schema, the model-owned slot set and the
human-readable surface map can never drift between the two.

Scope of this snapshot: the six argument-bearing capabilities whose extraction is
MODEL-owned end to end after commit ``935781e`` (the retired C-class / DET lane).
Assets/CTX capabilities (vision / read / pdf / ...) are SYSTEM-owned (``asset_id``
comes from turn facts, never the sentence) and are intentionally absent: the model
is never asked to fill a slot for them.

Descriptions are the exact registry strings the model reads, frozen at
``SOURCE_COMMIT`` (migrations 0002 seed + 0019 translate ``target_language`` +
0020 clean-topic ``query`` / human-readable ``domain``). ``max_len`` mirrors the
registry so the rendered slot block matches the live contract byte-for-byte.

Stdlib only — no production import — so the verifier runs anywhere.
"""
from __future__ import annotations

SOURCE_COMMIT = "fdf96cf"

# ── the clean-topic query contract (migration 0020) ──────────────────────────────
_SEARCH_QUERY_DESC = (
    "the search query — a clean, self-contained TOPIC distilled from the user "
    "sentence (instruction frame removed); the model understands the topic, it is "
    "NOT copied verbatim")
_RAG_DOMAIN_DESC = (
    "optional domain NAME to scope the search to (resolved to a real domain id by "
    "the tool); only when the sentence names one")

# ── the contract, one entry per MODEL-extraction capability ───────────────────────
# params: the registry parameter table (type / required / description / max_len?).
# model_slots: the slots the extractor model is allowed to emit for the capability,
#              in the order the slot block is rendered. This order MUST equal the
#              production handler's ``slot_plan().model_slots`` (the online source of
#              truth), because it fixes both the rendered slot block AND the
#              ``function.parameters`` key order. The protocol test enforces the tie.
SCHEMA: dict[str, dict] = {
    "cap-web-search": {
        "tool": "web_search",
        "description": (
            "Search the public web for external information, including webpages, "
            "reference material, news, and current information. Use it when the user "
            "asks to search or look up information on the internet. It does not apply "
            "to searching the user's own materials or searches restricted to a specific "
            "social platform.\n\n中文：在公开互联网上搜索外部信息，包括网页、资料、新闻和时事。"
            "当用户要求上网查找公开网络信息时适用。搜索用户自己的资料或限定在特定社交平台上搜索，"
            "不属于它。"),
        "params": {
            "query": {"type": "string", "required": True, "description": _SEARCH_QUERY_DESC},
            "top_k": {"type": "integer", "required": False,
                      "description": "optional number of results; only when the sentence states one"},
        },
        "model_slots": ["query", "top_k"],
    },
    "cap-rag-search": {
        "tool": "rag_search",
        "description": (
            "Search the user's own imported materials or document library. Use it when "
            "the user asks to find information in their imported or uploaded materials. "
            "It does not apply to public-web search or social-platform search.\n\n中文："
            "在用户自己导入的材料（个人文档库、已上传的资料）里检索内容。当用户要在“我导入的资料／"
            "我的资料库”中查找某个问题时适用。搜索公开网络、或搜索社交平台，都不属于它。"),
        "params": {
            "query": {"type": "string", "required": True, "description": _SEARCH_QUERY_DESC},
            "domain": {"type": "string", "required": False, "description": _RAG_DOMAIN_DESC},
            "top_k": {"type": "integer", "required": False,
                      "description": "optional result count; only when the sentence states one"},
        },
        "model_slots": ["query", "domain", "top_k"],
    },
    "cap-social-search": {
        "tool": "search_social",
        "description": (
            "Search social platforms such as Reddit for discussions, recommendations, "
            "and user opinions. Use it when the user asks what people or online "
            "communities are saying on a social platform. It does not apply to general "
            "public-web search or searches over the user's own materials.\n\n中文：在 "
            "Reddit 等社交平台上搜索大家的讨论、推荐与评价。当用户点名社交平台，或要“社区／论坛里"
            "大家怎么说、网友评价”时适用。一般网页搜索、搜索用户自己的资料，都不属于它。"),
        "params": {
            "query": {"type": "string", "required": True, "description": _SEARCH_QUERY_DESC},
            "platform": {"type": "string", "required": False,
                         "description": "optional platform (reddit / x / zhihu / auto); only when the sentence names one"},
            # A real backend parameter (the search_social tool still accepts it),
            # but TEMPORARILY NOT model-owned: the frozen corpus has zero subreddit
            # positives, so it is deliberately left out of `model_slots` below and
            # never appears in the rendered slot block / function face (matching the
            # production handler's slot_plan). Do not re-add until the data covers it.
            "subreddit": {"type": "string", "required": False,
                          "description": "optional reddit-only scope; only when the sentence names one"},
            "limit": {"type": "integer", "required": False,
                      "description": "optional max results per platform; only when the sentence states one"},
        },
        "model_slots": ["query", "platform", "limit"],
    },
    "cap-translate": {
        "tool": "translate",
        "description": (
            "Translate provided text from one language into another. Use it when the "
            "user explicitly asks to translate text into a different language. It does "
            "not apply to summarizing, explaining, or rewriting text in the same "
            "language.\n\n中文：把给定文本从一种语言翻译成另一种语言。当用户要求“翻译成某种语言”"
            "时适用，附件或引用中的文字同样可以翻译。要求总结、解释或同语言改写的，都不属于它。"),
        "params": {
            "text": {"type": "string", "required": True,
                     "description": "the text to translate, copied from the sentence"},
            "target_language": {"type": "string", "required": False,
                                "description": "optional target language name (e.g. \"English\", "
                                               "\"Chinese\"); omitted -> executor default (English)"},
        },
        "model_slots": ["text", "target_language"],
    },
    "cap-add-term": {
        "tool": "add_term",
        "description": (
            "Add a word, term, or expression to a named vocabulary domain. Use it when "
            "the user asks to add or save a term to their vocabulary, glossary, or "
            "vocabulary list. It does not apply to creating files or folders or editing "
            "existing file content.\n\n中文：把一个词、术语或表达收进用户词汇库（生词本、术语表、"
            "词库里的某个命名域）。当用户要求将某个词、术语或表达加入/收录到词汇库的指定命名域时适用。"
            "它只新增一条词汇记录；新建文件或文件夹、修改已有文件的内容，都不属于它。"),
        "params": {
            "term": {"type": "string", "required": True, "max_len": 120,
                     "description": "the word/term to add, quotes removed"},
            "domain": {"type": "string", "required": True, "max_len": 60,
                       "description": "the vocabulary domain to add it into"},
        },
        "model_slots": ["term", "domain"],
    },
    "cap-create-folder": {
        "tool": "create_folder",
        "description": (
            "Create a new folder with an explicit name in the workspace. Use it when the "
            "user asks to create or make a folder or directory. It does not apply to "
            "creating or editing files, renaming existing folders, or adding terms to a "
            "vocabulary domain.\n\n中文：新建一个名称明确的文件夹（目录、子文件夹）。当用户要求"
            "“新建／创建文件夹”时适用。新建文档或文件、给已有文件夹改名、把词收进词汇库，都不属于它。"),
        "params": {
            "name": {"type": "string", "required": True, "max_len": 120,
                     "description": "the folder name, quotes removed"},
        },
        "model_slots": ["name"],
    },
}

# Slots whose VALUE must be taken from a human-readable surface field, never a slug
# or an opaque id (the business layer resolves the name). Only ``domain`` qualifies:
# PE v2 stored a slug (``domain_id_candidate``) in ``expected_arguments`` and the
# readable form in ``domain_surface``. ``platform`` / ``target_language`` already
# equal their canonical tool-facing value in ``expected_arguments``, so no remap.
SURFACE_FIELD: dict[str, str] = {"domain": "domain_surface"}

# Integer-typed model slots (type-checked during the audit).
INT_SLOTS: frozenset[str] = frozenset({"top_k", "limit"})


def model_slots(capability_id: str) -> list[str]:
    """The slots the extractor model may emit for ``capability_id``."""
    return list(SCHEMA[capability_id]["model_slots"])


def required_model_slots(capability_id: str) -> list[str]:
    """The model-owned slots that are ``required`` in the registry schema."""
    params = SCHEMA[capability_id]["params"]
    return [s for s in SCHEMA[capability_id]["model_slots"] if params[s].get("required")]


def model_schema(capability_id: str) -> dict:
    """The JSON-schema ``function.parameters`` object: ONLY the model-owned slots
    (the Phase-3b face narrowing), with ``required`` recomputed over the survivors."""
    props: dict[str, dict] = {}
    required: list[str] = []
    params = SCHEMA[capability_id]["params"]
    for slot in SCHEMA[capability_id]["model_slots"]:
        spec = params[slot]
        prop: dict = {"type": spec["type"]}
        if spec.get("description"):
            prop["description"] = spec["description"]
        if spec.get("max_len") is not None:
            prop["maxLength"] = int(spec["max_len"])
        props[slot] = prop
        if spec.get("required"):
            required.append(slot)
    return {"type": "object", "properties": props, "required": required}


def function_description(capability_id: str) -> str:
    return SCHEMA[capability_id]["description"]


def tool_name(capability_id: str) -> str:
    return SCHEMA[capability_id]["tool"]

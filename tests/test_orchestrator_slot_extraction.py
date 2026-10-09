"""Unified slot acquisition — the orchestrator WIRING (``_acquisition_hop``).

Scheme A pins how a registered handler's ``slot_plan()`` drives the SHARED
extractor inside the handler branch, WITHOUT the caller ever inferring "needs a
model" from ``slot not in draft``:

* an authorized plan runs the extractor ONCE, over the query bundle, with the
  capability's Prompt; valid, authorized values are folded into the draft;
* a STATED count is a RULE-obtained value (``top_k`` / ``limit`` extracted by the
  shared DET rule in ``acquire()``), so it is never authorized for the model; an
  unstated count stays the tool's default;
* an EMPTY / INVALID / UNAVAILABLE reply leaves the handler draft (the verbatim
  query + DET values) intact — the original request is never lost, and the SAME
  Binder remains the final gate (a required slot no one filled still lands
  MISSING -> the Agent);
* a handler WITHOUT ``slot_plan`` (the asset-anchored handlers: vision / read /
  pdf) calls no model — the pure-deterministic flow is byte-identical to before;
* the six MODEL-ASSISTED handlers (search trio + add-term / create-folder /
  translate) run the same seam: a DET-sufficient draft calls no model, a DET gap
  is authorized by the plan, and a DET-obtained value is never overwritten.

The extractor here is a MOCK seam: the real Docker Qwen call is exercised by the
integration test at the bottom, which SKIPS when the endpoint is not deployed
(the funnel tests never require a live model).
"""
from __future__ import annotations

import types
from urllib.parse import urlparse

import pytest

from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
)
from core.application.chat.intent_funnel.argument_acquisition.prompts import (
    ADD_TERM_PROMPT,
    CREATE_FOLDER_PROMPT,
    SEARCH_QUERY_PROMPT,
    TRANSLATE_PROMPT,
)
from core.application.chat.intent_funnel.cap_handler import roster
from core.application.chat.intent_funnel.contract import REASON_BIND_MISSING
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings


# ── entry builders (mirror the live schema shapes) ───────────────────────────────


def _entry(cid: str, tool: str, params: dict) -> CapabilityEntry:
    q = "搜索知识库"
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description="d",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=q,
                                      language=derive_language(q)),),
        parameters=params,
    )


def _web_entry() -> CapabilityEntry:
    return _entry("cap-web-search", "web_search", {
        "query": {"type": "string", "required": True, "max_len": 200,
                  "description": "query"},
        "top_k": {"type": "integer", "required": False, "description": "count"},
    })


def _rag_entry(*, domain_required: bool = False) -> CapabilityEntry:
    return _entry("cap-rag-search", "rag_search", {
        "query": {"type": "string", "required": True, "description": "query"},
        "top_k": {"type": "integer", "required": False, "description": "count"},
        "domain": {"type": "string", "required": domain_required, "description": "d"},
    })


def _social_entry() -> CapabilityEntry:
    return _entry("cap-social-search", "search_social", {
        "query": {"type": "string", "required": True, "description": "query"},
        "platform": {"type": "string", "required": False, "description": "platform"},
        "subreddit": {"type": "string", "required": False, "description": "scope"},
        "limit": {"type": "integer", "required": False, "description": "count"},
    })


def _add_term_entry() -> CapabilityEntry:
    return _entry("cap-add-term", "add_term", {
        "term": {"type": "string", "required": True, "description": "word"},
        "domain": {"type": "string", "required": True, "description": "domain"},
    })


def _create_folder_entry() -> CapabilityEntry:
    return _entry("cap-create-folder", "create_folder", {
        "name": {"type": "string", "required": True, "max_len": 120,
                 "description": "name"},
        "parent_path": {"type": "string", "required": False,
                        "description": "root"},
    })


def _translate_entry() -> CapabilityEntry:
    return _entry("cap-translate", "translate", {
        "text": {"type": "string", "required": True, "description": "text"},
        # no max_len on either side, exactly like the runtime tool schema
        "target_language": {"type": "string", "required": False,
                            "description": "lang"},
    })


def _view(entry: CapabilityEntry) -> RegistryLiveView:
    return RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))


async def _hop(entry: CapabilityEntry, *, query: str, extractor=None):
    trace: dict = {}
    capture: dict = {}
    deps = types.SimpleNamespace(argument_extractor=extractor)
    out = await orch_mod._acquisition_hop(
        TurnRequirements(), deps, entry, query=query, facts=None,
        view=_view(entry), trace=trace, capture=capture)
    return out, trace, capture


def _mock(returned=None, *, boom=False, seen=None):
    async def _extract(*, query, entry, model_slots, bundle, prompt):
        if seen is not None:
            seen.update(query=query, slots=tuple(model_slots),
                        prompt=prompt, source=bundle.source)
        if boom:
            raise ExtractionUnavailable("endpoint down")
        return returned if returned is not None else {}, bundle.source
    return _extract


# ── the model cleans the search topic (query) ────────────────────────────────────


async def test_web_query_is_replaced_by_the_cleaned_topic():
    msg = "帮我搜一下注意力机制的最新进展"
    seen: dict = {}
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "注意力机制的最新进展"}, seen=seen))

    assert out is not None
    assert out.requested_action["args"] == {"query": "注意力机制的最新进展"}
    assert seen["query"] == msg                      # the model saw the raw sentence
    assert seen["slots"] == ("query",)               # no count stated -> query only
    assert seen["prompt"] == SEARCH_QUERY_PROMPT     # per-capability prompt
    assert capture["acquisition"]["extractor"] == "Qwen"
    assert capture["acquisition"]["slot_plan"] == {
        "model_slots": ["query"], "default_slots": ["top_k"]}


async def test_web_stated_count_is_det_extracted_and_certified():
    msg = "search for transformers, top 3"
    seen: dict = {}
    out, trace, capture = await _hop(
        _web_entry(), query=msg,
        extractor=_mock({"query": "transformers", "top_k": 3}, seen=seen))

    # rule 1: the stated count is a DET-obtained value, never authorized for the model.
    assert seen["slots"] == ("query",)
    # top_k comes from the DET rule (not the model); the Binder normalizes to str.
    assert out.requested_action["args"] == {"query": "transformers", "top_k": "3"}


async def test_web_without_count_keeps_the_tool_default_for_top_k():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "注意力机制"}))
    assert out.requested_action["args"] == {"query": "注意力机制"}
    assert "top_k" not in out.requested_action["args"]   # tool default stands


# ── empty / invalid / unavailable / unauthorized: the draft is never lost ─────────


async def test_empty_model_value_keeps_the_verbatim_query():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "   "}))
    assert out.requested_action["args"] == {"query": msg}


async def test_over_length_model_value_is_rejected():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "x" * 300}))
    assert out.requested_action["args"] == {"query": msg}


async def test_model_unavailable_keeps_the_draft_and_still_certifies():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock(boom=True))
    assert out is not None                           # the turn is NOT dropped
    assert out.requested_action["args"] == {"query": msg}
    assert capture["acquisition"]["model_values"] == {}


async def test_unauthorized_slot_from_the_model_is_never_written():
    msg = "帮我搜一下注意力机制"                       # no count -> top_k not authorized
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "注意力机制", "top_k": 9}))
    assert out.requested_action["args"] == {"query": "注意力机制"}
    assert "top_k" not in out.requested_action["args"]


async def test_platform_fallback_value_is_accepted_only_when_none_named():
    msg = "搜索社区里对注意力机制的讨论"                # DET -> auto, model authorized
    out, trace, capture = await _hop(
        _social_entry(), query=msg, extractor=_mock({"query": "注意力机制讨论", "platform": "x"}))
    assert out.requested_action["args"] == {"query": "注意力机制讨论", "platform": "x"}


async def test_det_platform_survives_a_divergent_model_value():
    msg = "在 reddit 上搜一下 transformer"              # DET -> reddit, not authorized
    out, trace, capture = await _hop(
        _social_entry(), query=msg, extractor=_mock({"query": "transformer", "platform": "zhihu"}))
    assert out.requested_action["args"] == {"query": "transformer", "platform": "reddit"}


# ── a required slot nobody filled still fails closed through the SAME Binder ──────


async def test_binder_missing_required_slot_exits_to_agent():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _rag_entry(domain_required=True), query=msg, extractor=_mock({}))
    assert out is None
    assert trace["fallback"] == REASON_BIND_MISSING


# ── a handler without slot_plan calls no model (pure-deterministic flow) ──────────


async def test_handler_without_slot_plan_calls_no_model(monkeypatch):
    class _PlainHandler:
        capability_id = "cap-web-search"

        async def acquire(self, *, query, facts):
            return {"query": str(query).strip()}

    async def _must_not_run(**kw):
        raise AssertionError("a handler without slot_plan must not call the extractor")

    monkeypatch.setitem(roster.HANDLERS, "cap-web-search", _PlainHandler())
    out, trace, capture = await _hop(_web_entry(), query="帮我搜一下", extractor=_must_not_run)
    assert out.requested_action["args"] == {"query": "帮我搜一下"}
    assert "slot_plan" not in capture["acquisition"]     # nothing was planned
    assert "extractor" not in capture["acquisition"]     # no model was called


async def test_handler_with_no_extractor_keeps_the_deterministic_draft():
    # The production default when the extractor seam is absent: the handler draft
    # stands untouched (proves the pre-existing search-handler tests still hold).
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(_web_entry(), query=msg, extractor=None)
    assert out.requested_action["args"] == {"query": msg}
    assert "extractor" not in capture["acquisition"]


# ── add-term: DET keeps what it reliably named; the model fills ONLY the gap ──────


async def test_add_term_det_sufficient_draft_calls_no_model():
    msg = "把 attention 加入 Tec 词库"                 # DET resolves term AND domain

    async def _must_not_run(**kw):
        raise AssertionError("a DET-sufficient add-term draft must not call the model")

    out, trace, capture = await _hop(_add_term_entry(), query=msg,
                                     extractor=_must_not_run)
    assert out.requested_action["args"] == {"term": "attention", "domain": "Tec"}
    assert "slot_plan" not in capture["acquisition"]   # nothing was authorized
    assert "extractor" not in capture["acquisition"]   # no model was called


async def test_add_term_deictic_gap_is_filled_by_the_model_and_det_value_survives():
    msg = "帮我把这个词加入 金融 词汇库"                # deictic term -> DET misses it
    seen: dict = {}
    out, trace, capture = await _hop(
        _add_term_entry(), query=msg,
        extractor=_mock({"term": "entropy", "domain": "HALLUCINATED"}, seen=seen))

    assert seen["slots"] == ("term",)                  # only the unfilled slot asked
    assert seen["prompt"] == ADD_TERM_PROMPT
    args = out.requested_action["args"]
    assert args["term"] == "entropy"                   # the model closed the gap
    assert args["domain"] == "金融 词汇库"              # DET value is truth, never overwritten


# ── create-folder: name only — a model-fabricated path never reaches the draft ────


async def test_create_folder_det_named_folder_calls_no_model():
    msg = "创建一个叫 2026-Q1 报表 的文件夹"

    async def _must_not_run(**kw):
        raise AssertionError("a DET-named folder must not call the model")

    out, trace, capture = await _hop(_create_folder_entry(), query=msg,
                                     extractor=_must_not_run)
    assert out.requested_action["args"] == {"name": "2026-Q1 报表"}
    assert "extractor" not in capture["acquisition"]


async def test_create_folder_generic_sentence_is_named_by_the_model():
    msg = "帮我建个文件夹"                             # DET yields the empty draft
    seen: dict = {}
    out, trace, capture = await _hop(
        _create_folder_entry(), query=msg,
        extractor=_mock({"name": "会议纪要", "parent_path": "/etc"}, seen=seen))

    assert seen["slots"] == ("name",)                  # name ONLY — never a root/path
    assert seen["prompt"] == CREATE_FOLDER_PROMPT
    args = out.requested_action["args"]
    assert args["name"] == "会议纪要"
    assert "parent_path" not in args                   # unauthorized -> dropped


async def test_create_folder_model_still_missing_name_exits_bind_missing():
    msg = "帮我建个文件夹"
    out, trace, capture = await _hop(_create_folder_entry(), query=msg,
                                     extractor=_mock({}))
    assert out is None
    assert trace["fallback"] == REASON_BIND_MISSING


# ── translate: text DET-first; the semantic target_language always rides the plan ──


async def test_translate_det_payload_plus_model_target_language():
    msg = '把 "machine translation" 翻译成中文'         # DET owns the quoted payload
    seen: dict = {}
    out, trace, capture = await _hop(
        _translate_entry(), query=msg,
        extractor=_mock({"target_language": "Chinese", "text": "EVIL"}, seen=seen))

    assert seen["slots"] == ("target_language",)       # text ACQUIRED -> never re-asked
    assert seen["prompt"] == TRANSLATE_PROMPT
    args = out.requested_action["args"]
    assert args["text"] == "machine translation"       # DET verbatim, model echo dropped
    assert args["target_language"] == "Chinese"


async def test_translate_target_language_unstated_omits_the_slot_for_the_default():
    msg = "translate 你好，世界"                       # bare form: DET resolves nothing
    out, trace, capture = await _hop(
        _translate_entry(), query=msg,
        extractor=_mock({"text": "你好，世界"}))        # model OMITS an unstated target

    args = out.requested_action["args"]
    assert args == {"text": "你好，世界"}              # no invented target_language;
    assert "target_language" not in args               # the EXECUTOR owns the EN default


async def test_translate_zh_en_mixed_sentence_extracts_text_and_target():
    msg = "注意力机制的架构用英语怎么表达"              # no delimiter -> the model reads it
    seen: dict = {}
    out, trace, capture = await _hop(
        _translate_entry(), query=msg,
        extractor=_mock({"text": "注意力机制的架构", "target_language": "English"},
                        seen=seen))

    assert set(seen["slots"]) == {"text", "target_language"}
    assert out.requested_action["args"] == {"text": "注意力机制的架构",
                                            "target_language": "English"}


async def test_translate_model_unavailable_keeps_the_det_text():
    msg = '把 "hello world" 翻译成中文'                # DET text; only target pending
    out, trace, capture = await _hop(_translate_entry(), query=msg,
                                     extractor=_mock(boom=True))
    assert out is not None                             # the turn is NOT dropped
    assert out.requested_action["args"] == {"text": "hello world"}
    assert capture["acquisition"]["model_values"] == {}


async def test_translate_no_payload_and_no_model_value_exits_bind_missing():
    msg = "帮我翻译一下"                               # pure instruction: DET {}
    out, trace, capture = await _hop(_translate_entry(), query=msg,
                                     extractor=_mock({}))
    assert out is None
    assert trace["fallback"] == REASON_BIND_MISSING    # required text: nobody filled it


# ── the REAL Docker Qwen call (skips unless the endpoint is deployed) ─────────────


def _extractor_endpoint_reachable() -> bool:
    import socket
    url = str(getattr(settings, "chat_tool_intent_local_url", "") or "").strip()
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _extractor_endpoint_reachable(),
                    reason="local Qwen extractor endpoint not deployed")
async def test_real_qwen_cleans_a_zh_search_topic():
    """REAL model call: a Chinese instruction frame must be stripped to a topic,
    and a stated count must still land top_k. Run only against the live Docker
    service (``chat_tool_intent_local_url``); skipped when it is not deployed."""
    from core.application.chat.intent_funnel.argument_acquisition.extractor import extract

    seen: dict = {}

    async def real(*, query, entry, model_slots, bundle, prompt):
        seen["prompt"] = prompt
        return await extract(query=query, entry=entry, model_slots=model_slots,
                             bundle=bundle, prompt=prompt)

    msg = "帮我搜一下注意力机制的最新进展"
    out, trace, capture = await _hop(_web_entry(), query=msg, extractor=real)
    args = out.requested_action["args"]
    assert args["query"]                                   # a non-empty topic
    assert "帮我搜一下" not in args["query"]                # the frame was stripped
    assert seen["prompt"] == SEARCH_QUERY_PROMPT
    assert capture["acquisition"]["extractor"] == "Qwen"


@pytest.mark.skipif(not _extractor_endpoint_reachable(),
                    reason="local Qwen extractor endpoint not deployed")
async def test_real_qwen_reads_the_translate_target_language_directive():
    """REAL model call: the DET payload stays untouched and the live Qwen resolves
    the semantic ``target_language`` from the Chinese directive."""
    from core.application.chat.intent_funnel.argument_acquisition.extractor import extract

    seen: dict = {}

    async def real(*, query, entry, model_slots, bundle, prompt):
        seen["prompt"] = prompt
        seen["slots"] = tuple(model_slots)
        return await extract(query=query, entry=entry, model_slots=model_slots,
                             bundle=bundle, prompt=prompt)

    msg = '把 "machine translation" 翻译成中文'
    out, trace, capture = await _hop(_translate_entry(), query=msg, extractor=real)
    args = out.requested_action["args"]
    assert seen["slots"] == ("target_language",)           # text ACQUIRED -> never asked
    assert args["text"] == "machine translation"           # DET verbatim survives
    assert args.get("target_language")                     # the model named a target
    assert seen["prompt"] == TRANSLATE_PROMPT

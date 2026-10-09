"""Unified slot acquisition — the PURE contract units (no cascade, no model).

The acquisition contract (docs/phase4-acquisition-contract.md) puts natural-
language understanding on the MODEL. The search trio keeps the turn's sentence
VERBATIM as a lossless fallback the model may clean; the three MODEL-lane
capabilities (add-term / create-folder / translate) run NO extraction rule at
all, so their ``acquire()`` returns the EMPTY ``{}`` and the model owns every
slot. Nothing here re-implements natural-language extraction with regex / quotes
/ fixed phrases — that C-class implementation had no requirement basis and is
removed.

* ``acquire()`` — the handler draft. A search handler returns ``{"query": …}``
  (verbatim, the model's cleaning fallback); the MODEL-lane trio returns ``{}``.
  Only a BLANK query returns ``None`` (no input → the Agent owns the turn).
* ``SlotPlan`` / each handler's ``slot_plan()`` — the EXPLICIT four-state
  disposition (ACQUIRED / PENDING_MODEL / DEFAULTED / MISSING) the orchestrator
  reads INSTEAD of inferring "needs a model" from ``slot not in draft``.
* ``slot_refine.accept_model_values`` — the fold of authorized, valid model
  values into the draft (never an unauthorized overwrite).
* ``prompts.prompt_for`` — the per-capability system prompt (default
  byte-compatible for a capability that registers none).
* the extractor's ``_payload`` — the ``prompt`` override is the system message
  and an absent one is byte-identical to the frozen default.

The MODEL itself is not exercised here; the real invocation lives in
``test_orchestrator_slot_extraction`` (and its real-Qwen integration test).
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.intent_funnel.argument_acquisition import context_bundle
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    SYSTEM_EXTRACT,
    _payload,
)
from core.application.chat.intent_funnel.argument_acquisition.prompts import (
    ADD_TERM_PROMPT,
    CREATE_FOLDER_PROMPT,
    SEARCH_QUERY_PROMPT,
    TRANSLATE_PROMPT,
    prompt_for,
)
from core.application.chat.intent_funnel.argument_acquisition.slot_refine import (
    accept_model_values,
)
from core.application.chat.intent_funnel.cap_handler import (
    AddTermHandler,
    CreateFolderHandler,
    PdfExtractTextHandler,
    PdfTableToTextHandler,
    RagSearchHandler,
    ReadDocumentHandler,
    ReadFileHandler,
    SlotPlan,
    SlotPlanningHandler,
    SocialSearchHandler,
    TranslateHandler,
    VisionHandler,
    WebSearchHandler,
)
from core.application.chat.intent_funnel.registry.entry import CapabilityEntry


# ── acquire(): the handler draft (search trio=verbatim; MODEL-lane trio={}) ──────


@pytest.mark.parametrize("handler_cls", [WebSearchHandler, RagSearchHandler,
                                         SocialSearchHandler])
async def test_search_acquire_returns_the_verbatim_query(handler_cls):
    handler = handler_cls()
    draft = await handler.acquire(query="  帮我搜一下注意力机制  ", facts=None)
    assert draft == {"query": "帮我搜一下注意力机制"}    # only surrounding spaces stripped


@pytest.mark.parametrize("handler_cls", [AddTermHandler, CreateFolderHandler,
                                         TranslateHandler])
async def test_model_lane_acquire_returns_the_empty_draft_on_any_input(handler_cls):
    handler = handler_cls()
    # NO extraction rule runs: a rich sentence yields the SAME ``{}`` as a bare one.
    assert await handler.acquire(query='把 "keystone" 加入我的工程词汇库', facts=None) == {}
    assert await handler.acquire(query="帮我搜一下注意力机制", facts=None) == {}


@pytest.mark.parametrize("handler_cls", [WebSearchHandler, RagSearchHandler,
                                         SocialSearchHandler, AddTermHandler,
                                         CreateFolderHandler, TranslateHandler])
@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_returns_none(handler_cls, blank):
    handler = handler_cls()
    assert await handler.acquire(query=blank, facts=None) is None


# ── SlotPlan + each handler's slot_plan ──────────────────────────────────────────


def test_slot_plan_defaults_are_empty_and_frozen():
    plan = SlotPlan()
    assert plan.model_slots == () and plan.default_slots == ()
    with pytest.raises(Exception):
        plan.model_slots = ("x",)  # frozen dataclass


async def _draft(handler, message):
    return await handler.acquire(query=message, facts=types.SimpleNamespace())


async def test_web_slot_plan_authorizes_query_and_the_optional_count():
    h = WebSearchHandler()
    draft = await _draft(h, "帮我搜一下注意力机制")
    plan = h.slot_plan(query="帮我搜一下注意力机制", facts=None, draft=draft)
    assert plan.model_slots == ("query", "top_k")        # count is model-owned now
    assert plan.default_slots == ("top_k",)              # unstated -> the tool default


async def test_rag_slot_plan_authorizes_query_domain_and_count():
    h = RagSearchHandler()
    draft = await _draft(h, "帮我搜一下注意力机制")
    plan = h.slot_plan(query="帮我搜一下注意力机制", facts=None, draft=draft)
    assert plan.model_slots == ("query", "domain", "top_k")
    assert plan.default_slots == ("top_k",)


async def test_social_slot_plan_authorizes_query_platform_subreddit_and_count():
    h = SocialSearchHandler()
    draft = await _draft(h, "搜索社区里对注意力机制的讨论")
    plan = h.slot_plan(query="搜索社区里对注意力机制的讨论", facts=None, draft=draft)
    assert plan.model_slots == ("query", "platform", "subreddit", "limit")
    assert plan.default_slots == ("limit",)              # no platform default emitted


async def test_add_term_slot_plan_authorizes_term_and_domain():
    h = AddTermHandler()
    draft = await _draft(h, "帮我加个词到词库里")
    plan = h.slot_plan(query="帮我加个词到词库里", facts=None, draft=draft)
    assert set(plan.model_slots) == {"term", "domain"} and plan.default_slots == ()


async def test_create_folder_slot_plan_authorizes_name_only():
    h = CreateFolderHandler()
    draft = await _draft(h, "帮我建个文件夹")
    plan = h.slot_plan(query="帮我建个文件夹", facts=None, draft=draft)
    assert plan.model_slots == ("name",) and plan.default_slots == ()


async def test_translate_slot_plan_authorizes_text_and_target_language():
    h = TranslateHandler()
    draft = await _draft(h, "帮我翻译一下")
    plan = h.slot_plan(query="帮我翻译一下", facts=None, draft=draft)
    assert set(plan.model_slots) == {"text", "target_language"}
    assert plan.default_slots == ()          # the English default is the EXECUTOR's


def test_slot_planning_handlers_opt_in_pure_det_handlers_do_not():
    # the six capabilities whose slots are semantic -> the model lane
    for h in (WebSearchHandler(), RagSearchHandler(), SocialSearchHandler(),
              AddTermHandler(), CreateFolderHandler(), TranslateHandler()):
        assert isinstance(h, SlotPlanningHandler)
    # asset-anchored handlers carry no slot_plan -> the orchestrator calls no model.
    for h in (VisionHandler(), ReadDocumentHandler(), ReadFileHandler(),
              PdfExtractTextHandler(), PdfTableToTextHandler()):
        assert not isinstance(h, SlotPlanningHandler)


# ── slot_refine.accept_model_values ──────────────────────────────────────────────


_PARAMS = {
    "query": {"type": "string", "required": True, "max_len": 200},
    "top_k": {"type": "integer", "required": False},
    "platform": {"type": "string", "required": False, "enum": ["reddit", "x", "zhihu", "auto"]},
}


def test_accept_folds_authorized_valid_values():
    out = accept_model_values({"query": "raw sentence"}, {"query": "clean topic", "top_k": 3},
                              model_slots=("query", "top_k"), parameters=_PARAMS)
    assert out == {"query": "clean topic", "top_k": 3}   # integer coerced


def test_accept_never_writes_an_unauthorized_slot():
    out = accept_model_values({"query": "raw", "platform": "reddit"},
                              {"query": "clean", "platform": "zhihu"},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "clean", "platform": "reddit"}  # existing value kept


def test_accept_drops_unknown_slot():
    out = accept_model_values({"query": "raw"}, {"query": "clean", "bogus": "x"},
                              model_slots=("query", "bogus"), parameters=_PARAMS)
    assert out == {"query": "clean"}                     # never invent a slot


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"])
def test_accept_rejects_empty_value_keeps_draft(raw):
    out = accept_model_values({"query": "raw sentence"}, {"query": raw},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "raw sentence"}


def test_accept_rejects_type_invalid_integer():
    out = accept_model_values({"query": "raw"}, {"top_k": "three"},
                              model_slots=("query", "top_k"), parameters=_PARAMS)
    assert "top_k" not in out


def test_accept_rejects_over_length_string():
    out = accept_model_values({"query": "raw"}, {"query": "x" * 201},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "raw"}


def test_accept_rejects_out_of_enum():
    out = accept_model_values({"platform": "auto"}, {"platform": "facebook"},
                              model_slots=("platform",), parameters=_PARAMS)
    assert out == {"platform": "auto"}


def test_accept_does_not_mutate_the_input_draft():
    draft = {"query": "raw"}
    accept_model_values(draft, {"query": "clean"}, model_slots=("query",),
                        parameters=_PARAMS)
    assert draft == {"query": "raw"}


# ── prompts.prompt_for ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("cid", ["cap-web-search", "cap-rag-search", "cap-social-search"])
def test_prompt_for_search_caps_is_the_search_prompt(cid):
    assert prompt_for(cid) == SEARCH_QUERY_PROMPT


@pytest.mark.parametrize("cid,expected", [
    ("cap-add-term", ADD_TERM_PROMPT),
    ("cap-create-folder", CREATE_FOLDER_PROMPT),
    ("cap-translate", TRANSLATE_PROMPT),
])
def test_prompt_for_the_new_model_assisted_caps(cid, expected):
    assert prompt_for(cid) == expected


@pytest.mark.parametrize("cid", ["cap-vision", "cap-read-file", "", None])
def test_prompt_for_unregistered_cap_is_the_frozen_default(cid):
    assert prompt_for(cid) == SYSTEM_EXTRACT


# ── extractor._payload: the prompt override ──────────────────────────────────────


def _entry() -> CapabilityEntry:
    return CapabilityEntry(
        capability_id="cap-web-search", tool_binding="web_search",
        parameters={"query": {"type": "string", "required": True, "description": "q"}},
    )


def test_payload_uses_provided_prompt_as_the_system_message():
    bundle = context_bundle.build("帮我搜一下注意力机制")
    payload = _payload(_entry(), ("query",), bundle, "CUSTOM SYSTEM")
    assert payload["messages"][0] == {"role": "system", "content": "CUSTOM SYSTEM"}


def test_payload_without_prompt_is_byte_compatible_with_the_default():
    bundle = context_bundle.build("帮我搜一下注意力机制")
    payload = _payload(_entry(), ("query",), bundle)
    assert payload["messages"][0] == {"role": "system", "content": SYSTEM_EXTRACT}

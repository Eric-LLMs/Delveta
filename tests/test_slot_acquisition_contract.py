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
* ``prompts.prompt_for`` — the shared INTERFACE prompt (no per-capability rule).
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
from core.application.chat.intent_funnel.argument_acquisition.prompts import prompt_for
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


async def test_web_slot_plan_authorizes_query_only():
    h = WebSearchHandler()
    draft = await _draft(h, "帮我搜一下注意力机制")
    plan = h.slot_plan(query="帮我搜一下注意力机制", facts=None, draft=draft)
    assert plan.model_slots == ("query",)                # count is NOT a model slot
    assert plan.default_slots == ()                      # the tool owns the default (5)


async def test_rag_slot_plan_authorizes_query_and_domain():
    h = RagSearchHandler()
    draft = await _draft(h, "帮我搜一下注意力机制")
    plan = h.slot_plan(query="帮我搜一下注意力机制", facts=None, draft=draft)
    assert plan.model_slots == ("query", "domain")       # count is NOT a model slot
    assert plan.default_slots == ()


async def test_social_slot_plan_authorizes_query_and_platform():
    h = SocialSearchHandler()
    draft = await _draft(h, "搜索社区里对注意力机制的讨论")
    plan = h.slot_plan(query="搜索社区里对注意力机制的讨论", facts=None, draft=draft)
    # subreddit is never a model slot; limit is a tool default (5), not a slot.
    assert plan.model_slots == ("query", "platform")
    assert plan.default_slots == ()


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


def test_accept_folds_authorized_values_unchanged():
    # The model's value is folded in EXACTLY as produced — no int coercion, no
    # strip. ("3" stays the string the model emitted.)
    out = accept_model_values({"query": "raw sentence"}, {"query": "clean topic", "top_k": "3"},
                              model_slots=("query", "top_k"), parameters=_PARAMS,
                              fallback_slots=("query",))
    assert out == {"query": "clean topic", "top_k": "3"}


def test_accept_never_writes_an_unauthorized_slot():
    out = accept_model_values({"query": "raw", "platform": "reddit"},
                              {"query": "clean", "platform": "zhihu"},
                              model_slots=("query",), parameters=_PARAMS,
                              fallback_slots=("query",))
    assert out == {"query": "clean", "platform": "reddit"}  # existing value kept


def test_accept_drops_unknown_slot():
    out = accept_model_values({"query": "raw"}, {"query": "clean", "bogus": "x"},
                              model_slots=("query", "bogus"), parameters=_PARAMS,
                              fallback_slots=("query",))
    assert out == {"query": "clean"}                     # never invent a slot


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"])
def test_accept_blank_value_keeps_the_draft(raw):
    # A blank model value is "no value produced" (a PRESENCE check, not a rewrite):
    # the draft's own deterministic fallback stands.
    out = accept_model_values({"query": "raw sentence"}, {"query": raw},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "raw sentence"}


def test_accept_writes_a_non_blank_value_verbatim():
    # A value the model DID produce is carried EXACTLY as emitted (no strip).
    out = accept_model_values({"query": "raw sentence"}, {"query": " padded "},
                              model_slots=("query",), parameters=_PARAMS,
                              fallback_slots=("query",))
    assert out == {"query": " padded "}


def test_accept_does_not_coerce_a_type_invalid_integer():
    out = accept_model_values({"query": "raw"}, {"top_k": "three"},
                              model_slots=("query", "top_k"), parameters=_PARAMS)
    assert out["top_k"] == "three"                       # verbatim, not dropped


def test_accept_does_not_filter_over_length_or_enum():
    out = accept_model_values({"query": "raw", "platform": "auto"},
                              {"query": "x" * 201, "platform": "facebook"},
                              model_slots=("query", "platform"), parameters=_PARAMS,
                              fallback_slots=("query", "platform"))
    assert out == {"query": "x" * 201, "platform": "facebook"}


def test_accept_does_not_mutate_the_input_draft():
    draft = {"query": "raw"}
    accept_model_values(draft, {"query": "clean"}, model_slots=("query",),
                        parameters=_PARAMS)
    assert draft == {"query": "raw"}


def test_validation_report_flags_without_changing_values():
    from core.application.chat.intent_funnel.argument_acquisition.slot_refine import (
        validation_report,
    )
    report = validation_report(
        {"query": "x" * 201, "top_k": "three", "platform": "facebook", "stray": "v"},
        model_slots=("query", "top_k", "platform"), parameters=_PARAMS)
    assert report == {"query": "max_len", "top_k": "type",
                      "platform": "enum", "stray": "unauthorized"}


# ── prompts.prompt_for ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("cid", ["cap-web-search", "cap-rag-search", "cap-social-search",
                                 "cap-add-term", "cap-create-folder", "cap-translate",
                                 "cap-vision", "", None])
def test_prompt_for_every_capability_is_the_shared_interface_prompt(cid):
    # No per-capability rule prompt any more: TRAINING and PRODUCTION carry the
    # SAME interface prompt (the extractor default) for every capability.
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

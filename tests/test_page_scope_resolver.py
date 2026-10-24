"""Page-scope resolver — the ONLINE-LLM reading scope for the file capabilities.

Pins the contract that makes the resolver safe:

* the reply is a CLOSED four-field JSON object (target / page_scope / pages /
  reason) — an identity field (``asset_id``), a ``status`` the system derives,
  or any other unknown key is rejected outright;
* ``all`` omits ``pages``, ``single`` is one 1-based number, ``range`` is
  validated by the tools' own strict parser and canonicalised;
* an unknown target/scope is AMBIGUOUS, a contract violation is INVALID, a
  failed/timed-out/malformed call is UNAVAILABLE — three distinct outcomes that
  are never collapsed into "read the whole document";
* ``conversation_context`` is a settled target even when ``page_scope`` is
  ``unknown`` (a conversation has no page axis), and its body is the sanctioned
  recent user turns — never model-generated;
* the resolver never renders ``asset_id`` or file content into the prompt and
  never lets a model value replace the turn fact's document identity;
* every page-addressed handler consumes the verdict alone: a non-ok verdict is
  a TERMINAL failure for ``read_document`` / ``pdf_extract_text`` (never a
  guessed page, never the whole document), an Agent hand-off for the table
  capability, and an omitted ``page`` for ``vision`` unless the request names
  several pages.

The transport tests pin the actual request shape (``stream=False``,
``enable_thinking=False``, JSON mode, bounded tokens) against a fake client — no
network call, no real key.
"""
from __future__ import annotations

import types

import pytest
from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.cap_handler import (
    CATEGORY_SCOPE_UNSETTLED,
    AcquisitionFailure,
    PdfExtractTextHandler,
    ReadDocumentHandler,
    VisionHandler,
)
from core.application.chat.intent_funnel.contract import TurnFacts
from core.infrastructure import llm as llm_mod

# ── reply parsing: the closed, strict contract ─────────────────────────────────


def test_all_scope_omits_pages():
    out = psr.parse_reply({"target": "document", "page_scope": "all",
                           "pages": None, "reason": "whole doc"})

    assert out.ok and out.is_document
    assert out.page_scope == psr.SCOPE_ALL
    assert out.pages is None


def test_single_scope_keeps_one_page():
    out = psr.parse_reply({"target": "document", "page_scope": "single",
                           "pages": "5", "reason": "page 5"})

    assert out.ok and out.pages == "5"


def test_range_scope_is_canonicalised_by_the_tool_parser():
    out = psr.parse_reply({"target": "document", "page_scope": "range",
                           "pages": "3-5", "reason": "pages 3 to 5"})

    assert out.ok and out.pages == "3-5"


def test_disjoint_range_stays_a_sorted_page_list():
    out = psr.parse_reply({"target": "document", "page_scope": "range",
                           "pages": "5,1,3", "reason": "three pages"})

    assert out.ok and out.pages == "1,3,5"


def test_all_scope_rejects_a_page_value():
    out = psr.parse_reply({"target": "document", "page_scope": "all",
                           "pages": "3", "reason": "contradiction"})

    assert out.status == psr.STATUS_INVALID
    assert out.pages is None


@pytest.mark.parametrize("raw", ["five", "3-", "3.5", "", None])
def test_single_scope_rejects_a_non_integer(raw):
    out = psr.parse_reply({"target": "document", "page_scope": "single",
                           "pages": raw, "reason": ""})

    assert out.status == psr.STATUS_INVALID


def test_single_scope_rejects_zero():
    out = psr.parse_reply({"target": "document", "page_scope": "single",
                           "pages": "0", "reason": ""})

    assert out.status == psr.STATUS_INVALID


def test_single_scope_rejects_a_page_past_the_document_end():
    out = psr.parse_reply({"target": "document", "page_scope": "single",
                           "pages": "9", "reason": ""}, page_count=5)

    assert out.status == psr.STATUS_INVALID


def test_range_scope_rejects_a_page_past_the_document_end():
    out = psr.parse_reply({"target": "document", "page_scope": "range",
                           "pages": "3-9", "reason": ""}, page_count=5)

    assert out.status == psr.STATUS_INVALID


def test_range_scope_rejects_a_span_over_the_per_call_cap():
    out = psr.parse_reply({"target": "document", "page_scope": "range",
                           "pages": "1-17", "reason": ""})

    assert out.status == psr.STATUS_INVALID


def test_range_scope_rejects_reversed_bounds():
    out = psr.parse_reply({"target": "document", "page_scope": "range",
                           "pages": "5-2", "reason": ""})

    assert out.status == psr.STATUS_INVALID


@pytest.mark.parametrize("field,value", [
    ("asset_id", "asset-from-model"),
    ("file", "secret.pdf"),
    ("path", "/tmp/secret.pdf"),
    ("status", "ok"),
])
def test_reply_may_not_carry_identity_status_or_unknown_fields(field, value):
    out = psr.parse_reply({"target": "document", "page_scope": "all",
                           "pages": None, "reason": "", field: value})

    assert out.status == psr.STATUS_INVALID
    assert field in out.reason


def test_unknown_target_is_ambiguous_not_all():
    out = psr.parse_reply({"target": "unknown", "page_scope": "all",
                           "pages": None, "reason": "cannot tell"})

    assert out.status == psr.STATUS_AMBIGUOUS
    assert out.pages is None


def test_unknown_scope_is_ambiguous_not_all():
    out = psr.parse_reply({"target": "document", "page_scope": "unknown",
                           "pages": None, "reason": "cannot tell"})

    assert out.status == psr.STATUS_AMBIGUOUS
    assert out.pages is None


def test_missing_scope_field_is_invalid():
    out = psr.parse_reply({"target": "document", "pages": None,
                           "reason": "incomplete"})

    assert out.status == psr.STATUS_INVALID


def test_non_object_reply_is_invalid():
    assert psr.parse_reply(["not", "an", "object"]).status == psr.STATUS_INVALID  # type: ignore[arg-type]


def test_conversation_target_returns_the_sanctioned_user_turns():
    out = psr.parse_reply(
        {"target": "conversation_context", "page_scope": "all",
         "pages": None, "reason": "about the chat"},
        user_turns=("what did we discuss?",))

    assert out.ok
    assert out.target == psr.TARGET_CONVERSATION
    assert out.context_content == ("what did we discuss?",)
    assert out.pages is None


def test_conversation_target_survives_an_unknown_page_scope():
    # A conversation has no page axis: ``page_scope=unknown`` must not discard
    # the target the model DID identify.
    out = psr.parse_reply(
        {"target": "conversation_context", "page_scope": "unknown",
         "pages": None, "reason": "about the chat"},
        user_turns=("summarise what we said",))

    assert out.ok
    assert out.target == psr.TARGET_CONVERSATION
    assert out.context_content == ("summarise what we said",)
    assert out.pages is None


# ── the bounded online call ────────────────────────────────────────────────────


def _facts(**over):
    base = {
        "has_viewer": True, "viewer_asset_id": "asset-123",
        "viewer_current_page": 4, "viewer_kind": "pdf",
        "viewer_name": "survey.pdf", "viewer_page_from": 4,
        "viewer_page_to": 4,
    }
    base.update(over)
    return TurnFacts(**base)


class _FakeLLM:
    def __init__(self, reply=None, *, error=None):
        self.reply = reply
        self.error = error
        self.seen: dict = {}

    async def complete_json_once(self, prompt, system_prompt, **kwargs):
        self.seen.update(prompt=prompt, system_prompt=system_prompt, **kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.reply or {})


async def test_resolve_passes_the_exact_bounded_request():
    llm = _FakeLLM({"target": "document", "page_scope": "single",
                    "pages": "4", "reason": "this page"})

    out = await psr.resolve(llm=llm, query="summarise this page", facts=_facts(),
                            capability="cap-read-document", asset_id="asset-123")

    assert out.ok and out.pages == "4"
    assert llm.seen["disable_thinking"] is True
    assert llm.seen["temperature"] == 0.0
    assert llm.seen["timeout"] == psr.DEFAULT_TIMEOUT_SECONDS
    assert llm.seen["max_tokens"] == psr.DEFAULT_MAX_TOKENS
    assert isinstance(llm.seen["usage_out"], dict)


async def test_resolve_never_renders_the_asset_id_or_file_content():
    llm = _FakeLLM({"target": "document", "page_scope": "all",
                    "pages": None, "reason": "whole doc"})

    await psr.resolve(llm=llm, query="summarise it", facts=_facts(),
                      capability="cap-read-document", asset_id="asset-123")

    assert "asset-123" not in llm.seen["prompt"]
    # Only the metadata the decision needs reaches the model.
    assert "survey.pdf" in llm.seen["prompt"]
    assert "current page: 4" in llm.seen["prompt"]


async def test_resolve_records_provider_usage():
    llm = _FakeLLM({"target": "document", "page_scope": "all",
                    "pages": None, "reason": "whole doc"})
    original = llm.complete_json_once

    async def _with_usage(prompt, system_prompt, **kwargs):
        kwargs["usage_out"].update(prompt_tokens=12, completion_tokens=3)
        return await original(prompt, system_prompt, **kwargs)

    llm.complete_json_once = _with_usage  # type: ignore[method-assign]

    out = await psr.resolve(llm=llm, query="summarise it", facts=_facts())

    assert dict(out.usage) == {"prompt_tokens": 12, "completion_tokens": 3}


async def test_transport_failure_is_unavailable_not_all():
    llm = _FakeLLM(error=TimeoutError("gateway timeout"))

    out = await psr.resolve(llm=llm, query="summarise pages 3-5",
                            facts=_facts())

    assert out.status == psr.STATUS_UNAVAILABLE
    assert not out.ok
    assert out.pages is None
    assert "TimeoutError" in out.reason


async def test_malformed_json_is_unavailable_not_all():
    llm = _FakeLLM(error=ValueError("model did not return valid JSON"))

    out = await psr.resolve(llm=llm, query="summarise pages 3-5",
                            facts=_facts())

    assert out.status == psr.STATUS_UNAVAILABLE
    assert out.pages is None


# ── complete_json_once: the actual wire parameters ─────────────────────────────


class _Completions:
    def __init__(self, content: str):
        self.content = content
        self.seen: dict = {}

    async def create(self, **kwargs):
        self.seen.update(kwargs)
        message = types.SimpleNamespace(content=self.content)
        usage = types.SimpleNamespace(prompt_tokens=7, completion_tokens=2)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                     usage=usage)


class _Client:
    def __init__(self, completions, base_url="http://test/v1"):
        self.chat = types.SimpleNamespace(completions=completions)
        self.base_url = base_url
        self.api_key = "test-key"


def _llm_with(content: str, monkeypatch):
    completions = _Completions(content)
    monkeypatch.setattr(llm_mod, "AsyncOpenAI",
                        lambda **kwargs: _Client(completions))
    client = llm_mod.OpenAILLM(api_key="test-key", base_url="http://test/v1",
                               model="test-model")
    return client, completions


async def test_complete_json_once_is_non_streaming_json_mode_thinking_off(
        monkeypatch):
    llm, completions = _llm_with('{"ok": true}', monkeypatch)

    parsed = await llm.complete_json_once("prompt", "system", temperature=0.0,
                                          max_tokens=64, timeout=3.0,
                                          disable_thinking=True)

    assert parsed == {"ok": True}
    assert completions.seen["stream"] is False
    assert completions.seen["response_format"] == {"type": "json_object"}
    assert completions.seen["extra_body"] == {"enable_thinking": False}
    assert completions.seen["max_tokens"] == 64
    assert completions.seen["temperature"] == 0.0


async def test_complete_json_once_honours_the_global_thinking_knob(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "llm_disable_thinking", True)
    llm, completions = _llm_with('{"ok": true}', monkeypatch)

    await llm.complete_json_once("prompt", "system")

    assert completions.seen["extra_body"] == {"enable_thinking": False}


async def test_complete_json_once_rejects_a_non_object_reply(monkeypatch):
    llm, _ = _llm_with("[1, 2, 3]", monkeypatch)

    with pytest.raises(TypeError, match="JSON object"):
        await llm.complete_json_once("prompt", "system")


# ── read_document: verdict -> pages, fail closed otherwise ─────────────────────


def _scope(**over) -> psr.PageScope:
    base = {
        "target": psr.TARGET_DOCUMENT, "page_scope": psr.SCOPE_ALL,
        "pages": None, "status": psr.STATUS_OK, "reason": "",
    }
    base.update(over)
    return psr.PageScope(**base)


def _resolver_returning(scope, *, seen=None):
    async def _resolve(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        return scope
    return _resolve


async def test_read_document_all_omits_the_pages_slot():
    draft = await ReadDocumentHandler().acquire(
        query="summarise it", facts=_facts(),
        page_scope_resolver=_resolver_returning(_scope()))

    assert draft == {"asset_id": "asset-123"}


@pytest.mark.parametrize("scope,pages", [
    (psr.SCOPE_SINGLE, "4"),
    (psr.SCOPE_RANGE, "3-5"),
    (psr.SCOPE_RANGE, "1,3,5"),
])
async def test_read_document_maps_single_and_range_to_pages(scope, pages):
    draft = await ReadDocumentHandler().acquire(
        query="summarise", facts=_facts(),
        page_scope_resolver=_resolver_returning(
            _scope(page_scope=scope, pages=pages)))

    assert draft == {"asset_id": "asset-123", "pages": pages}


@pytest.mark.parametrize("result", [
    _scope(status=psr.STATUS_AMBIGUOUS),
    _scope(status=psr.STATUS_INVALID),
    _scope(status=psr.STATUS_UNAVAILABLE),
    _scope(target=psr.TARGET_CONVERSATION),
    _scope(target=psr.TARGET_UNKNOWN),
])
async def test_read_document_never_turns_a_non_document_verdict_into_a_read(result):
    # A document handler has no conversation lane and no rule to lean on: a
    # verdict that does not settle THIS document's scope ends the turn.
    out = await ReadDocumentHandler().acquire(
        query="what did we discuss", facts=_facts(),
        page_scope_resolver=_resolver_returning(result))

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED


async def test_read_document_is_terminal_when_neither_producer_settles():
    out = await ReadDocumentHandler().acquire(
        query="extract some pages", facts=_facts(),
        page_scope_resolver=_resolver_returning(
            _scope(status=psr.STATUS_AMBIGUOUS)))

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED


async def test_read_document_resolver_sees_the_capability_and_asset_but_not_identity_control():
    seen: dict = {}

    draft = await ReadDocumentHandler().acquire(
        query="page 2", facts=_facts(),
        page_scope_resolver=_resolver_returning(_scope(), seen=seen))

    assert draft is not None
    assert seen["capability"] == "cap-read-document"
    assert seen["asset_id"] == "asset-123"
    assert seen["query"] == "page 2"


async def test_read_document_without_an_asset_never_calls_the_resolver():
    async def _boom(**kwargs):
        raise AssertionError("resolver must not run without a document")

    draft = await ReadDocumentHandler().acquire(
        query="summarise it", facts=_facts(viewer_asset_id=""),
        page_scope_resolver=_boom)

    assert draft is None


async def test_read_document_without_a_resolver_is_unsettled_not_whole_document():
    # No semantic judge -> a whole-document read would be a silent fallback that
    # the user never asked for. The turn ends instead.
    out = await ReadDocumentHandler().acquire(
        query="summarise it", facts=_facts(viewer_page_from=3, viewer_page_to=5))

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.scope["status"] == "not_wired"


# ── pdf_extract_text: the same mapping, its own capability id ─────────────────


async def test_pdf_extract_text_maps_a_range_and_reports_its_capability():
    seen: dict = {}

    draft = await PdfExtractTextHandler().acquire(
        query="extract pages 3 to 5", facts=_facts(),
        page_scope_resolver=_resolver_returning(
            _scope(page_scope=psr.SCOPE_RANGE, pages="3-5"), seen=seen))

    assert draft == {"asset_id": "asset-123", "pages": "3-5"}
    assert seen["capability"] == "cap-pdf-extract-text"


async def test_pdf_extract_text_is_terminal_when_the_scope_is_unsettled():
    out = await PdfExtractTextHandler().acquire(
        query="extract some pages", facts=_facts(),
        page_scope_resolver=_resolver_returning(
            _scope(status=psr.STATUS_UNAVAILABLE)))

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True
    assert out.category == CATEGORY_SCOPE_UNSETTLED


async def test_pdf_extract_text_without_a_resolver_is_unsettled():
    out = await PdfExtractTextHandler().acquire(query="extract this",
                                                facts=_facts())

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is True


# ── vision: the page axis of a one-page tool ──────────────────────────────────


async def test_vision_maps_a_single_page_to_the_page_slot():
    draft = await VisionHandler().acquire(
        query="what does page 3 show?", facts=_facts(viewer_kind="pdf"),
        page_scope_resolver=_resolver_returning(
            _scope(page_scope=psr.SCOPE_SINGLE, pages="3")))

    # The funnel's slots are strings; the tool coerces ``page`` to an int.
    assert draft == {"asset_id": "asset-123",
                     "question": "what does page 3 show?", "page": "3"}


async def test_vision_omits_page_for_a_whole_document_verdict():
    # ``page`` omitted -> the tool's documented default (the page open in the
    # viewer); never a guessed page.
    draft = await VisionHandler().acquire(
        query="what is this?", facts=_facts(viewer_kind="pdf"),
        page_scope_resolver=_resolver_returning(_scope()))

    assert draft == {"asset_id": "asset-123", "question": "what is this?"}


async def test_vision_hands_a_multi_page_request_to_the_agent():
    # One vision call renders exactly one page; picking one of several would be
    # a guess.
    out = await VisionHandler().acquire(
        query="compare pages 3 to 5", facts=_facts(viewer_kind="pdf"),
        page_scope_resolver=_resolver_returning(
            _scope(page_scope=psr.SCOPE_RANGE, pages="3-5")))

    assert isinstance(out, AcquisitionFailure)
    assert out.terminal is False
    assert out.category == CATEGORY_SCOPE_UNSETTLED


# ── the online-LLM adapter never leaks a client/key to the handlers ───────────


async def test_for_llm_binds_the_client_behind_the_callable_seam():
    llm = _FakeLLM({"target": "document", "page_scope": "all",
                    "pages": None, "reason": "whole doc"})

    resolver = psr.for_llm(llm)
    out = await resolver(query="summarise it", facts=_facts(),
                         capability="cap-read-document", asset_id="asset-123")

    assert out.ok
    assert "asset-123" not in llm.seen["prompt"]

"""cap_handler — TranslateHandler + its wiring into ``_acquisition_hop``.

Three layers are pinned here:

* the HANDLER contract itself (unit): ``cap-translate``'s single ``text`` slot is
  resolved DETERMINISTICALLY — priority ① a paired quote / code-block span, then
  priority ② a colon that is part of the translation instruction — and the payload
  is copied VERBATIM (outer delimiters stripped, only surrounding whitespace
  trimmed, internal/trailing punctuation kept, CJK never pre-translated). A colon
  that is NOT an instruction delimiter (``https://``, ``localhost:8080``) never
  splits. A bare form, a pure instruction, or an empty sentence fails closed
  (``None``) — the Agent owns the turn.
* the ROSTER wiring: ``handler_for("cap-translate")`` is the registered handler.
* the FUNNEL wiring: the handler short-circuits the generic acquisition chain and
  its draft goes through the SAME ``_certify`` -> Binder -> ``tool_intent``
  handoff; a ``None`` draft exits to the Agent with ``ACQUISITION_MISSING`` and
  the legacy extractor is never invoked.

The gold spans are the ``cap-translate`` cases from the frozen Formal-500
argument benchmark (``logs/_argbench/dataset_formal500.jsonl``), so this suite is
a direct reconciliation against that evidence.
"""
from __future__ import annotations

import logging
import re
import types

import pytest
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    TranslateHandler,
    handler_for,
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    MatchResult,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

MESSAGE = '把 "hello world" 翻译成中文'


# ── Layer 1a: priority ① — paired quote / code-block extraction ──────────────────
# (message, expected payload) — straight/curly double, single, corner, backtick,
# code block, target-language directives dropped, CJK copied not translated.


@pytest.mark.parametrize("message,expected", [
    ('把 "hello world" 翻译成中文', "hello world"),
    ('把 "Hello world" 翻译一下', "Hello world"),
    ('translate "The quick brown fox jumps over the lazy dog"',
     "The quick brown fox jumps over the lazy dog"),
    ('帮我翻译 "gradient descent"', "gradient descent"),
    ('translate "batch normalization" into Chinese', "batch normalization"),
    ('把 "machine translation" 翻译成中文', "machine translation"),
    ('把 "过拟合" 翻成英文', "过拟合"),
    ('帮我把 "latency budget" 翻译成日语', "latency budget"),
    ('translate "机器学习" into English', "机器学习"),
    ('translate "checkpointing" to Chinese', "checkpointing"),
    ('translate "你好，请问洗手间在哪里？" into French', "你好，请问洗手间在哪里？"),
    ('translate "overfitting" into German', "overfitting"),
    ('translate "The quick brown fox" into Spanish', "The quick brown fox"),
    ('translate "数据管道" to English', "数据管道"),
    ('translate "asynchronous" into Korean', "asynchronous"),
])
async def test_paired_double_quote_payload(message, expected):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {"text": expected}


@pytest.mark.parametrize("message,expected", [
    ("翻译 “分布式系统” 这个词", "分布式系统"),      # curly double quotes
    ("translate 'The weather is nice.'", "The weather is nice."),  # straight single
    ("翻译 「机器学习」 成英文", "机器学习"),          # corner brackets
    ("翻译 『深度学习』 一下", "深度学习"),            # white corner brackets
    ("translate `gradient_descent` into Chinese", "gradient_descent"),  # backtick
    ("翻译 ```loss_function```", "loss_function"),     # code block
])
async def test_paired_other_delimiter_payload(message, expected):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {"text": expected}


async def test_paired_quote_span_wins_over_a_later_colon():
    # priority ① precedes ②: a quoted span anywhere is the delimiter of record.
    handler = TranslateHandler()
    draft = await handler.acquire(query='翻译 "hello": 一段说明', facts=None)
    assert draft == {"text": "hello"}


async def test_empty_quoted_span_is_not_a_payload():
    # an empty pair supplies nothing -> fail closed, not a blank ``text``.
    handler = TranslateHandler()
    assert await handler.acquire(query='翻译 ""', facts=None) is None


# ── Layer 1b: priority ② — instruction-context colon ─────────────────────────────


@pytest.mark.parametrize("message,expected", [
    ("翻译这句话：Stay hungry, stay foolish", "Stay hungry, stay foolish"),
    ("翻译这段：Machine learning is a subset of AI.", "Machine learning is a subset of AI."),
    ("translate this sentence: Attention is all you need.", "Attention is all you need."),
    ("翻译一下这段：分布式系统很难调试。", "分布式系统很难调试。"),
    ("翻译：To be or not to be, that is the question.", "To be or not to be, that is the question."),
    ("translate this: 好记性不如烂笔头。", "好记性不如烂笔头。"),
    ("translate the sentence: All models are wrong, but some are useful.",
     "All models are wrong, but some are useful."),
    ("translate this phrase: garbage in, garbage out", "garbage in, garbage out"),
])
async def test_instruction_colon_payload(message, expected):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {"text": expected}


async def test_colon_payload_keeps_an_inner_colon():
    handler = TranslateHandler()
    assert (await handler.acquire(query="翻译：他说：你好", facts=None)) == {"text": "他说：你好"}


# ── Layer 1c: colon guards — URLs / ports are never split ────────────────────────


@pytest.mark.parametrize("message", [
    "翻译 https://example.com/a",
    "translate https://example.com/path:thing",
    "翻译 localhost:8080 的日志",
    "把这个端口 127.0.0.1:8000 翻译一下",
    "translate http://a.b/c:d into Chinese",
])
async def test_colon_in_url_or_port_is_not_a_delimiter(message):
    # no instruction lead-in prefixes these colons -> nothing is extracted.
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) is None


async def test_colon_without_a_translation_verb_is_not_a_delimiter():
    handler = TranslateHandler()
    assert await handler.acquire(query="注意：这里没有翻译意图", facts=None) is None


# ── Layer 1d: payload fidelity — punctuation, no pre-translation, no shell ───────


@pytest.mark.parametrize("message,expected", [
    ("翻译：好记性不如烂笔头。", "好记性不如烂笔头。"),      # trailing CJK period kept
    ("translate this: Attention is all you need.", "Attention is all you need."),  # trailing dot kept
    ('翻译 "你好，世界" into English', "你好，世界"),         # inner comma kept, CJK untouched
    ("翻译 “a, b; c!”", "a, b; c!"),                        # inner punctuation kept
    ('translate "  padded  "', "padded"),                   # only surrounding whitespace trimmed
])
async def test_payload_is_copied_verbatim(message, expected):
    handler = TranslateHandler()
    assert (await handler.acquire(query=message, facts=None))["text"] == expected


async def test_target_language_directive_is_never_part_of_the_payload():
    handler = TranslateHandler()
    for message, expected in [
        ('把 "machine translation" 翻译成中文', "machine translation"),
        ('translate "batch normalization" into Chinese', "batch normalization"),
        ('帮我把 "latency budget" 翻译成日语', "latency budget"),
    ]:
        assert (await handler.acquire(query=message, facts=None))["text"] == expected


async def test_only_the_text_slot_is_ever_emitted():
    handler = TranslateHandler()
    draft = await handler.acquire(query=MESSAGE, facts=types.SimpleNamespace())
    assert set(draft) == {"text"}          # no target-language / no other slot


# ── Layer 1e: strict fail-closed ─────────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    "帮我翻译一下",              # pure instruction, no payload
    "翻译一下这段",
    "把这句话翻译一下",
    "please translate the following",
    "translate the paragraph above",
    "translate 你好，世界 into English",   # bare form, no delimiter
    "翻译",
    "translate",
])
async def test_instruction_without_a_payload_fails_closed(message):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) is None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = TranslateHandler()
    assert await handler.acquire(query=blank, facts=None) is None


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_translate():
    handler = handler_for("cap-translate")
    assert isinstance(handler, TranslateHandler)
    assert handler.capability_id == "cap-translate"
    assert HANDLERS["cap-translate"] is handler


def test_cap_translate_is_no_longer_on_the_generic_chain():
    # it was an "unwired" example before Phase 3-A; it now owns its acquisition.
    assert handler_for("cap-translate") is not None
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 3: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _translate_entry() -> CapabilityEntry:
    q = "翻译文本"
    return CapabilityEntry(
        capability_id="cap-translate", tool_binding="translate",
        description="translate text into Chinese",
        standard_queries=(QueryRecord(id="cap-translate-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "text": {"type": "string", "required": True,
                     "description": "the text to translate, copied from the sentence"},
        },
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="翻译文本")

    async def fake_select_and_extract(query, candidates, *, entries_by_id,
                                      llm=None, facts=None):
        rec.legacy.append(query)
        raise AssertionError("legacy select_and_extract must never run on the new lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(tool_intent_mod, "select_and_extract", fake_select_and_extract)
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")
    return rec, view


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_translate_turn_certifies_handler_payload(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _translate_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "translate"
    assert act["capability_id"] == "cap-translate"
    assert act["args"] == {"text": "hello world"}      # quoted span, directive dropped
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_translate_turn_without_a_payload_exits_to_agent(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _translate_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我翻译一下"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_ACQUISITION_MISSING
    assert not rec.legacy                              # no Qwen, no legacy extraction


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch, caplog):
    """A forged MODEL value never reaches the draft: the handler owns ``text``."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _translate_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-translate": AcquisitionInputs(
        model_values={"text": "HALLUCINATED-INSTRUCTION-ECHO"})}
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    assert out.requested_action["args"] == {"text": "hello world"}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-translate"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-translate", _NoneHandler())
    entry = _translate_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query=MESSAGE, facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING

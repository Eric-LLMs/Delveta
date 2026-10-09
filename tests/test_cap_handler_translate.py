"""cap_handler — TranslateHandler + its wiring into ``_acquisition_hop``.

Layers pinned here:

* the HANDLER contract (unit): ``cap-translate``'s payload slot ``text`` and the
  optional ``target_language`` are BOTH MODEL-owned natural-language slots. Per
  the acquisition contract the handler runs NO extraction rule — the quote /
  code-block / instruction-colon delimiter pre-parsing had no requirement basis
  and is removed. ``acquire()`` returns the EMPTY ``{}`` draft for ANY non-blank
  sentence and ``None`` only for a blank one; ``slot_plan()`` authorizes the model
  for both slots. The deterministic English default for an absent
  ``target_language`` is the EXECUTOR's constant, not a model output, so a model
  that omits an unnamed target simply leaves the slot absent.
* the FUNNEL wiring: ``handler_for("cap-translate")`` owns the draft and short-
  circuits the generic chain; the empty draft is completed by the shared extractor
  (``deps.argument_extractor``) and only the authorized slots are folded, then the
  SAME ``_certify`` -> Binder -> ``tool_intent`` gate runs. With the extractor seam
  ABSENT the required ``text`` stays unfilled and the Binder exits ``BIND_MISSING``.
* the model-assisted round-trip lives in ``test_orchestrator_slot_extraction``.

The ``cap-translate`` cases of the frozen Formal-500 argument benchmark
(``logs/_argbench/dataset_formal500.jsonl``) are reused as a REPRESENTATIVE
corpus: every one must now yield the EMPTY handler draft. The gold is unchanged
(it recorded only the ``text`` slot; ``target_language`` is a later, confirmed
optional slot).
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path

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
    REASON_BIND_MISSING,
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

GOLD_PATH = "logs/_argbench/dataset_formal500.jsonl"

MESSAGE = '把 "hello world" 翻译成中文'

_GOLD_PRESENT = Path(GOLD_PATH).exists()


def _gold_messages() -> list[str]:
    if not _GOLD_PRESENT:
        return []
    import json

    rows = [
        json.loads(line)
        for line in Path(GOLD_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r["query"] for r in rows if r.get("capability_id") == "cap-translate"]


GOLD_MESSAGES = _gold_messages()


# ── Layer 0: the handler runs NO rule — the empty draft for every gold case ───────


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
@pytest.mark.parametrize("message", GOLD_MESSAGES)
async def test_handler_yields_the_empty_draft_for_every_gold_case(message):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
def test_the_representative_corpus_is_the_30_gold_cases():
    assert len(GOLD_MESSAGES) == 30


# ── Layer 1: the handler contract (empty draft on any input; None on blank) ───────


@pytest.mark.parametrize("message", [
    '把 "hello world" 翻译成中文',                 # quoted — no rule
    "翻译这句话：Stay hungry, stay foolish",        # instruction colon — no rule
    "翻译 https://example.com/a",                  # URL — no rule
    "帮我翻译一下",                                 # pure instruction
    "翻译：好记性不如烂笔头。",
])
async def test_acquire_always_returns_the_empty_draft(message):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = TranslateHandler()
    assert await handler.acquire(query=blank, facts=None) is None


async def test_slot_plan_authorizes_text_and_target_language():
    handler = TranslateHandler()
    msg = "帮我翻译一下"
    draft = await handler.acquire(query=msg, facts=None)
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert set(plan.model_slots) == {"text", "target_language"}
    assert plan.default_slots == ()          # the English default is the EXECUTOR's


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_translate():
    handler = handler_for("cap-translate")
    assert isinstance(handler, TranslateHandler)
    assert handler.capability_id == "cap-translate"
    assert HANDLERS["cap-translate"] is handler


def test_cap_translate_is_no_longer_on_the_generic_chain():
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
            "target_language": {"type": "string", "required": False,
                                "description": "the target language, when the sentence names one"},
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


def _deps(provider, extractor=None):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider, argument_extractor=extractor)


def _mock_extractor(returned):
    async def _extract(*, query, entry, model_slots, bundle, prompt):
        return returned, bundle.source
    return _extract


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_translate_turn_certifies_the_model_payload(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _translate_entry()
    rec, view = _wire(monkeypatch, entry)
    extractor = _mock_extractor({"text": "hello world", "target_language": "Chinese"})
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(None, extractor), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "translate"
    assert act["capability_id"] == "cap-translate"
    assert act["args"] == {"text": "hello world", "target_language": "Chinese"}
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_translate_turn_without_a_payload_exits_bind_missing(monkeypatch, caplog):
    """The empty draft is completed by the extractor; a payload no source filled
    leaves the required ``text`` MISSING, so the SAME Binder gate exits BIND_MISSING."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _translate_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我翻译一下"), _deps(None, _mock_extractor({})), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING
    assert not rec.legacy                              # no legacy extraction


async def test_unnamed_target_language_leaves_the_slot_for_the_executor_default(monkeypatch):
    """The model names no target: only ``text`` is certified; the executor's own
    English default owns the absent optional slot."""
    entry = _translate_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx('把 "hello world" 翻译一下'),
                       _deps(None, _mock_extractor({"text": "hello world"})), req)

    assert out is not req
    args = out.requested_action["args"]
    assert args == {"text": "hello world"}
    assert "target_language" not in args


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch, caplog):
    """A forged MODEL value in the generic chain never reaches the draft."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _translate_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-translate": AcquisitionInputs(
        model_values={"text": "HALLUCINATED-INSTRUCTION-ECHO"})}
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE),
                       _deps(lambda cid: poisoned.get(cid),
                             _mock_extractor({"text": "hello world"})), req)

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

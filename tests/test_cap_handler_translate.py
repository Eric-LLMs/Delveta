"""cap_handler — TranslateHandler + its wiring into ``_acquisition_hop``.

The business contract pinned here:

* ``cap-translate`` runs NO Qwen extraction. The handler sends the user's
  sentence VERBATIM as the Registry-required ``text`` slot; the ``translate``
  tool's own decision call reads the target language and decides what to
  translate (sentence / previous chat / document page / video).
* there is NO ``slot_plan`` and NO ``accepts_page_scope``: the orchestrator's
  RULE_CONTEXT branch stops at the handler draft and never reaches the shared
  extractor.
* a VIDEO viewer defers to the existing client capture lane; the already-decided
  ``text`` rides ``DeferredCapture.args`` and the resume injects only the
  uploaded frame's ``asset_id``.

The ``cap-translate`` cases of the frozen Formal-500 argument benchmark are
reused as a representative corpus: every one must yield the verbatim draft.
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path

import pytest
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
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


# ── Layer 0: the handler copies the sentence for every gold case ────────────────


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
@pytest.mark.parametrize("message", GOLD_MESSAGES)
async def test_handler_copies_the_sentence_for_every_gold_case(message):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {"text": message}


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
def test_the_representative_corpus_is_the_30_gold_cases():
    assert len(GOLD_MESSAGES) == 30


# ── Layer 1: the handler contract ───────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    '把 "hello world" 翻译成中文',
    "翻译这句话：Stay hungry, stay foolish",
    "翻译 https://example.com/a",
    "帮我翻译一下",
    "翻译：好记性不如烂笔头。",
])
async def test_acquire_returns_the_verbatim_sentence(message):
    handler = TranslateHandler()
    assert await handler.acquire(query=message, facts=None) == {"text": message}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = TranslateHandler()
    assert await handler.acquire(query=blank, facts=None) is None


def test_no_slot_plan_and_no_page_scope():
    """Neither seam exists: the RULE_CONTEXT branch can never reach the extractor."""
    handler = TranslateHandler()
    assert not hasattr(handler, "slot_plan")
    assert not getattr(handler, "accepts_page_scope", False)


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


def _ctx(message: str, *, viewer=None):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
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
                     "description": "the user's request sentence"},
            "target_language": {"type": "string", "required": False,
                                "description": "the target language, when named"},
            "asset_id": {"type": "string", "required": False,
                         "description": "captured video-frame asset id"},
        },
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


class _NoSelect:
    """Selector guard: a HIT-lane test fails loudly if selection is re-entered."""

    def __init__(self, rec: "_Rec") -> None:
        self.rec = rec

    async def select(self, query, candidates, *, entries_by_id, facts=None):
        self.rec.legacy.append(query)
        raise AssertionError("the selector must never run on a HIT lane")


class _ExplodingExtractor:
    """Any call proves Qwen extraction was re-entered — the test must fail."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        raise AssertionError("cap-translate must never call the shared extractor")


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="翻译文本")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: _NoSelect(rec))
    return rec, view


def _deps(provider, extractor=None):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider, argument_extractor=extractor)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_translate_turn_certifies_the_verbatim_sentence(monkeypatch):
    entry = _translate_entry()
    rec, view = _wire(monkeypatch, entry)
    extractor = _ExplodingExtractor()
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(None, extractor), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "translate"
    assert act["capability_id"] == "cap-translate"
    assert act["args"] == {"text": MESSAGE}
    assert act["funnel_stage"] == "tool_intent"
    assert act["funnel_registry_version"] == view.fingerprint
    assert extractor.calls == 0
    assert not rec.legacy


async def test_an_absent_extractor_changes_nothing(monkeypatch):
    entry = _translate_entry()
    _wire(monkeypatch, entry)
    out = await _route(_ctx(MESSAGE), _deps(None, None), TurnRequirements())
    assert out is not None
    assert out.requested_action["args"] == {"text": MESSAGE}


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """A handler with no legal draft (``None``) exits to the Agent — fail-closed."""
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


# ── Layer 4: the video deferred-capture lane ────────────────────────────────────


def _facts(**overrides):
    from core.application.chat.intent_funnel.contract import TurnFacts

    base = dict(has_viewer=True, viewer_kind="video", viewer_name="lecture.mp4",
                viewer_t_ms=12_500, viewer_duration_ms=600_000,
                viewer_provenance="local")
    base.update(overrides)
    return TurnFacts(**base)


def test_video_turn_defers_to_the_client_frame():
    handler = TranslateHandler()
    capture = handler.capture_for(query="这段视频英文怎么说", facts=_facts())
    assert capture is not None
    assert capture.kind == "frame"
    assert capture.name == "lecture.mp4"
    assert capture.t_ms == 12_500
    assert capture.args == {"text": "这段视频英文怎么说"}
    assert "asset_id" not in capture.args        # the resume injects the upload
    assert capture.request()["question"] == ""


def test_an_explicit_attachment_wins_over_the_capture():
    handler = TranslateHandler()
    assert handler.capture_for(
        query="翻译这份资料", facts=_facts(attachment_asset_id="a1")) is None


def test_a_cloud_video_still_defers_to_the_client_frame():
    """A video viewer's OWN asset is the video file, never a readable picture:
    it must not short-circuit the capture the way an attachment does."""
    handler = TranslateHandler()
    capture = handler.capture_for(
        query="这段视频英文怎么说",
        facts=_facts(viewer_asset_id="video-asset-9", viewer_provenance="cloud"))
    assert capture is not None
    assert capture.args == {"text": "这段视频英文怎么说"}
    assert "asset_id" not in capture.args


@pytest.mark.parametrize("kind", ["pdf", "office", "text", "markdown", "image"])
def test_non_video_viewers_never_ask_for_a_capture(kind):
    handler = TranslateHandler()
    assert handler.capture_for(query="翻译这一页", facts=_facts(viewer_kind=kind)) is None


def test_no_viewer_never_asks_for_a_capture():
    handler = TranslateHandler()
    assert handler.capture_for(query="翻译这句话", facts=_facts(has_viewer=False)) is None

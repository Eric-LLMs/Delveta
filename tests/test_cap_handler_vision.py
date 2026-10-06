"""cap_handler — VisionHandler + its wiring into ``_acquisition_hop``.

Two layers are pinned here:

* the HANDLER contract itself (unit): ``cap-vision`` resolves ``asset_id`` ONLY
  from :class:`TurnFacts` (precedence ``attachment_asset_id`` -> ``path_asset_id``
  -> ``viewer_asset_id``, i.e. the Binder's own order), copies the sentence
  VERBATIM as ``question``, and FAILS CLOSED (``None``) when no fact carries an
  asset id — no guessed id, no fabricated path, no empty string;
* the WIRING: the registered handler short-circuits the generic acquisition chain
  (provider / path_router / Qwen) and its draft goes through the SAME ``_certify``
  -> Binder -> ``tool_intent`` handoff.

The funnel test drives the REAL ``funnel.route`` -> orchestrator cascade with a
faked Matcher HIT (mirroring ``test_phase4_acquisition_step2``).
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
    VisionHandler,
    handler_for,
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    MatchResult,
    TurnFacts,
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

ASSET = "0ea50a94-f4f7-45eb-bbaa-43966d6af575"
MESSAGE = "这张图里的报错信息是什么意思"


# ── Layer 1: the VisionHandler contract (unit) ───────────────────────────────────


async def test_vision_asset_id_prefers_attachment_then_path_then_viewer():
    handler = VisionHandler()
    all_three = TurnFacts(attachment_asset_id="A", path_asset_id="P", viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=all_three))["asset_id"] == "A"
    path_viewer = TurnFacts(path_asset_id="P", viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=path_viewer))["asset_id"] == "P"
    viewer_only = TurnFacts(viewer_asset_id="V")
    assert (await handler.acquire(query=MESSAGE, facts=viewer_only))["asset_id"] == "V"


@pytest.mark.parametrize("facts", [
    None,
    TurnFacts(),                                   # nothing in context
    TurnFacts(has_attachment=True),                # flag set but no id -> still None
])
async def test_vision_without_an_asset_id_fails_closed(facts):
    handler = VisionHandler()
    assert await handler.acquire(query=MESSAGE, facts=facts) is None


async def test_vision_question_is_the_verbatim_sentence():
    handler = VisionHandler()
    draft = await handler.acquire(query=f"  {MESSAGE}  ", facts=TurnFacts(viewer_asset_id=ASSET))
    assert draft == {"asset_id": ASSET, "question": MESSAGE}


async def test_vision_omits_question_when_the_message_is_empty():
    # an image with no typed message -> just the asset; the tool runs its default
    # analysis prompt (question is an optional slot).
    handler = VisionHandler()
    draft = await handler.acquire(query="   ", facts=TurnFacts(viewer_asset_id=ASSET))
    assert draft == {"asset_id": ASSET}


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_vision():
    handler = handler_for("cap-vision")
    assert isinstance(handler, VisionHandler)
    assert handler.capability_id == "cap-vision"
    assert HANDLERS["cap-vision"] is handler


def test_roster_leaves_unwired_capabilities_on_the_generic_chain():
    # capabilities with no handler stay on the generic acquisition chain.
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 3: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str, *, open_asset_id: str = ""):
    """One turn's context. ``open_asset_id`` simulates the UI showing an image:
    it rides ``body.viewer.asset_id``, which ``TurnFacts.of`` lifts to
    ``viewer_asset_id``. ``path_asset_id`` / ``owned_asset_id`` stay empty so the
    test isolates the viewer source."""
    viewer = (types.SimpleNamespace(asset_id=open_asset_id, page=None, selections=[])
              if open_asset_id else None)
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _vision_entry() -> CapabilityEntry:
    q = "分析这张图片"
    return CapabilityEntry(
        capability_id="cap-vision", tool_binding="vision",
        description="analyze an attached image",
        standard_queries=(QueryRecord(id="cap-vision-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "asset_id": {"type": "string", "max_len": 64, "required": True,
                         "description": "the image asset id"},
            "question": {"type": "string", "required": False,
                         "description": "optional question"},
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
                           matched_literal="分析这张图片")

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


async def test_vision_turn_with_an_image_certifies_asset_and_question(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _vision_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE, open_asset_id=ASSET), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "vision"
    assert act["capability_id"] == "cap-vision"
    assert act["args"] == {"asset_id": ASSET, "question": MESSAGE}
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_vision_turn_without_an_image_exits_to_agent(monkeypatch, caplog):
    """No image in context -> the handler returns ``None`` -> the turn goes to
    the Agent with ``ACQUISITION_MISSING`` (fail-closed)."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _vision_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    # non-deictic message so no guardrail veto masks the handler's own decision
    out = await _route(_ctx("分析一下图片内容"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_ACQUISITION_MISSING
    assert not rec.legacy                              # no Qwen, no legacy extraction


async def test_handler_bypasses_a_populated_generic_chain(monkeypatch, caplog):
    """Even with generic provider inputs that WOULD have produced a different
    draft, the registered handler owns the result — proof the branch precedes
    the generic chain."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _vision_entry()
    rec, view = _wire(monkeypatch, entry)
    poisoned = {"cap-vision": AcquisitionInputs(
        model_values={"asset_id": "HALLUCINATED-UUID", "question": "HALLUCINATED Q"})}
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE, open_asset_id=ASSET),
                       _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    assert out.requested_action["args"] == {"asset_id": ASSET, "question": MESSAGE}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-vision"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-vision", _NoneHandler())
    entry = _vision_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query=MESSAGE, facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING

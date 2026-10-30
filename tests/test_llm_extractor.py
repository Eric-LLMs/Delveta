"""Unit surface for the POST-FUNNEL LLM fallback.

Architecture boundary pinned here:

1. the funnel itself runs ONLY the local extractor (Qwen); a certified turn
   clears the handoff, so ``resume_with_llm`` never calls the LLM (zero extra
   model calls on the normal path);
2. when Qwen failed / timed out / left a required slot empty, the funnel exits
   with a handoff record on ``ChatTurnContext`` (selected capability, original
   query, settled args, strategy, model slots);
3. the chat layer calls ``resume_with_llm`` AFTER the funnel returned — outside
   its 5s budget — and the request's OWN LLM (``deps.llm``) re-extracts the FULL
   parameter set;
4. the result goes through the SAME certification gate (``_certify`` → Binder);
   a complete set certifies, anything else leaves the turn with the Agent;
5. the LLM client is the caller's: no model/base_url/api_key override is passed,
   so the request's channel, connection pool and billing context apply.

No network: the LLM is a recording double and the Registry view is a fake.
"""
from __future__ import annotations

import types

import pytest
from core.application.chat.intent_funnel import orchestrator
from core.application.chat.intent_funnel.argument_acquisition import context_bundle
from core.application.chat.intent_funnel.argument_acquisition import llm_extractor as llm_ex
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
)
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    RegistryLiveView,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

SOURCE_QUERY = context_bundle.SOURCE_QUERY
SOURCE_TURNS = context_bundle.SOURCE_CONVERSATION_5_USER_TURNS

PARAMS = {
    "name": {"type": "string", "required": True, "description": "folder name"},
    "color": {"type": "string", "required": False, "description": "optional label colour"},
}


def _entry(params: dict | None = None, capability_id="cap-create-folder",
           tool_binding="create_folder"):
    return types.SimpleNamespace(
        capability_id=capability_id,
        tool_binding=tool_binding,
        description="Create a folder in the user's drive.",
        parameters=params or PARAMS,
    )


def _live_entry():
    return CapabilityEntry(
        capability_id="cap-create-folder", tool_binding="create_folder",
        description="Create a folder in the user's drive.",
        parameters=PARAMS,
    )


def _bundle(query: str = "make a folder called Notes", *, history=(),
            source=SOURCE_QUERY):
    return context_bundle.build(query, history=history, source=source)


def _ctx(*, pending=None, message="make a folder called Notes", history=()):
    ctx = types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id="", history=list(history),
    )
    ctx.intent_funnel_pending = pending
    ctx.intent_funnel_failure = None
    return ctx


def _pending(**over):
    record = {
        "capability_id": "cap-create-folder",
        "tool_binding": "create_folder",
        "query": "make a folder called Notes",
        "args": {},
        "arg_sources": {},
        "strategy": "QUERY_TO_EXTRACTOR",
        "model_slots": ["name"],
        "bundle_source": SOURCE_QUERY,
    }
    record.update(over)
    return record


class _FakeLLM:
    """Recording stand-in for the request-scoped ``LLM``."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def complete(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.reply


def _patch_view(monkeypatch, entries=None):
    view = RegistryLiveView(
        fingerprint="fp-test",
        entries=tuple(entries if entries is not None else [_live_entry()]),
    )

    async def fake_active(**kw):
        return view

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    return view


# ── the LLM extraction itself ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_carries_full_definition_query_and_settled_values():
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name",), bundle=_bundle(), values={"color": "red"},
    )

    prompt, _kwargs = llm.calls[0]
    # FULL registry definition — the fallback must understand the tool contract.
    assert "name (string, required" in prompt
    assert "color (string, optional" in prompt
    assert "<- fill this" in prompt
    assert "color: 'red'" in prompt  # what the funnel already settled
    assert "<user_sentence>make a folder called Notes</user_sentence>" in prompt
    assert "cap-create-folder" in prompt


@pytest.mark.asyncio
async def test_prompt_carries_only_the_users_own_turns():
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')
    bundle = _bundle(
        "call it Notes",
        history=[{"role": "user", "content": "I keep my work in folders"},
                 {"role": "assistant", "content": "ignored"},
                 {"role": "user", "content": "make a folder"}],
        source=SOURCE_TURNS,
    )

    await llm_ex.extract(llm=llm, query="call it Notes", entry=_entry(),
                         model_slots=("name",), bundle=bundle)

    prompt, _kwargs = llm.calls[0]
    assert "- I keep my work in folders" in prompt
    assert "- make a folder" in prompt
    assert "ignored" not in prompt


@pytest.mark.asyncio
async def test_llm_call_reuses_the_caller_channel():
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    await llm_ex.extract(llm=llm, query="make a folder", entry=_entry(),
                         model_slots=("name",), bundle=_bundle())

    _prompt, kwargs = llm.calls[0]
    # No client construction / channel override: only the per-call guard.
    assert set(kwargs) == {"system_prompt", "timeout"}
    assert kwargs["timeout"] == settings.llm_timeout_seconds
    assert "model" not in kwargs and "base_url" not in kwargs and "api_key" not in kwargs


@pytest.mark.asyncio
async def test_capability_prompt_rides_with_the_strict_contract():
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    await llm_ex.extract(llm=llm, query="make a folder", entry=_entry(),
                         model_slots=("name",), bundle=_bundle(),
                         prompt="CAPABILITY PROMPT")

    _prompt, kwargs = llm.calls[0]
    assert kwargs["system_prompt"].startswith(llm_ex.SYSTEM_LLM_EXTRACT)
    assert "CAPABILITY PROMPT" in kwargs["system_prompt"]


@pytest.mark.asyncio
async def test_incomplete_reply_is_returned_not_raised():
    # The CALLER's Binder is the gate; the extractor never fabricates a value.
    llm = _FakeLLM(reply='{"arguments": {"color": "red"}}')

    values, source = await llm_ex.extract(
        llm=llm, query="make a folder", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle())

    assert values == {"color": "red"}
    assert source == SOURCE_QUERY


@pytest.mark.asyncio
async def test_format_error_is_unavailable():
    llm = _FakeLLM(reply="here you go: name is Notes")

    with pytest.raises(ExtractionUnavailable):
        await llm_ex.extract(llm=llm, query="make a folder", entry=_entry(),
                             model_slots=("name",), bundle=_bundle())


@pytest.mark.asyncio
async def test_channel_error_is_unavailable_with_the_original_reason():
    llm = _FakeLLM(error=RuntimeError("channel down"))

    with pytest.raises(ExtractionUnavailable) as excinfo:
        await llm_ex.extract(llm=llm, query="make a folder", entry=_entry(),
                             model_slots=("name",), bundle=_bundle())
    assert "channel down" in str(excinfo.value)


@pytest.mark.asyncio
async def test_no_client_is_unavailable():
    with pytest.raises(ExtractionUnavailable):
        await llm_ex.extract(llm=None, query="make a folder", entry=_entry(),
                             model_slots=("name",), bundle=_bundle())


# ── the post-funnel seam ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_certified_turn_never_calls_the_llm(monkeypatch):
    """Qwen settled the turn -> the funnel cleared the handoff -> no LLM call."""
    _patch_view(monkeypatch)
    llm = _FakeLLM(reply='{"arguments": {"name": "WRONG"}}')

    out = await orchestrator.resume_with_llm(
        _ctx(pending=None), deps=types.SimpleNamespace(llm=llm),
        requirements=TurnRequirements())

    assert out is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_llm_success_certifies_through_the_same_binder(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')
    ctx = _ctx(pending=_pending(), message="make a folder called Notes")

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is not None
    action = out.requested_action
    assert action["capability_id"] == "cap-create-folder"
    assert action["tool"] == "create_folder"
    assert action["args"] == {"name": "Notes"}
    assert action["funnel_stage"] == "tool_intent"  # the same certification gate
    # The handoff is consumed: no second fallback, no stale Agent context.
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None


@pytest.mark.asyncio
async def test_llm_reply_is_merged_over_the_settled_args(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')
    ctx = _ctx(pending=_pending(args={"color": "red"},
                                arg_sources={"color": "handler"}))

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is not None
    assert out.requested_action["args"] == {"color": "red", "name": "Notes"}


@pytest.mark.asyncio
async def test_an_incomplete_llm_reply_hands_the_turn_to_the_agent(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(reply='{"arguments": {"color": "red"}}')
    ctx = _ctx(pending=_pending(), message="make a folder")

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    failure = ctx.intent_funnel_failure
    assert failure is not None
    assert failure["reason"] == "BIND_MISSING"
    assert failure["capability_id"] == "cap-create-folder"
    assert "name" in failure["missing"]


@pytest.mark.asyncio
async def test_an_llm_failure_keeps_the_funnel_handoff_for_the_agent(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(error=RuntimeError("channel down"))
    ctx = _ctx(pending=_pending(), message="make a folder")

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    failure = ctx.intent_funnel_failure
    assert failure is not None
    assert failure["reason"] == "ACQUISITION_MODEL_PENDING"
    assert failure["capability_id"] == "cap-create-folder"
    assert failure["tool_binding"] == "create_folder"
    assert failure["query"] == "make a folder called Notes"


@pytest.mark.asyncio
async def test_an_unadmitted_capability_is_not_resumed(monkeypatch):
    """The table no longer honors the pick -> the Agent decides (no LLM call)."""
    _patch_view(monkeypatch, entries=[])
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    out = await orchestrator.resume_with_llm(
        _ctx(pending=_pending()), deps=types.SimpleNamespace(llm=llm),
        requirements=TurnRequirements())

    assert out is None
    assert llm.calls == []

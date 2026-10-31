"""Unit surface for the POST-FUNNEL verify+extract call.

Architecture boundary pinned here:

1. the funnel itself runs ONLY the local extractor (Qwen); a certified turn
   clears the handoff, so ``resume_with_llm`` never calls the LLM (zero extra
   model calls on the normal path);
2. when Qwen failed / timed out / left a required slot empty, the funnel exits
   with a handoff record on ``ChatTurnContext`` (selected capability, original
   query, settled args, strategy, model slots);
3. the chat layer calls ``resume_with_llm`` AFTER the funnel returned — outside
   its 5s budget — and the request's OWN LLM (``deps.llm``) makes EXACTLY ONE
   call that both verifies the capability and extracts its arguments;
4. MATCH goes through the SAME certification gate (``_certify`` -> Binder);
   REJECT and ERROR hand the untouched original turn to the Agent with no
   capability trace and no routing note;
5. the LLM client is the caller's: no model/base_url/api_key override is passed,
   so the request's channel, connection pool and billing context apply.

No network: the LLM is a recording double and the Registry view is a fake.
"""
from __future__ import annotations

import types

import pytest
from agent.llm.llm_errors import LLMTemporaryError
from api.intent_funnel_context import (
    intent_funnel_reference_section,
    render_intent_funnel_context,
)
from core.application.chat.intent_funnel import orchestrator
from core.application.chat.intent_funnel.argument_acquisition import (
    capability_verify_extract as cve,
)
from core.application.chat.intent_funnel.argument_acquisition import history_window
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    RegistryLiveView,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

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


def _window(*history):
    return history_window.build(history)


def _match(args=None, reason=cve.REASON_CAPABILITY_MATCH):
    return {"decision": "MATCH", "reason_code": reason,
            "arguments": {} if args is None else args}


def _reject(reason=cve.REASON_FOLLOWUP_NOT_ACTION, args=None):
    return {"decision": "REJECT", "reason_code": reason,
            "arguments": {} if args is None else args}


def _ctx(*, pending=None, message="make a folder called Notes", history=(),
         failure=None, agent_context=None):
    ctx = types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id="", history=list(history),
    )
    ctx.intent_funnel_pending = pending
    ctx.intent_funnel_failure = failure
    ctx.agent_context = dict(agent_context or {})
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
        "bundle_source": "query",
    }
    record.update(over)
    return record


class _FakeLLM:
    """Recording stand-in for the request-scoped ``LLM``."""

    def __init__(self, reply=None, error: Exception | None = None):
        self.reply = _match() if reply is None else reply
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def complete_json_once(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.reply


class _CompleteOnlyLLM:
    """A pre-upgrade client: it must never be used for the verdict."""

    def __init__(self):
        self.calls = 0

    async def complete(self, prompt: str, **kwargs):  # pragma: no cover - asserted
        self.calls += 1
        return "{}"


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


def _wrapped(exc_type, cause):
    """The instance ``raise exc_type(...) from cause`` would produce."""
    exc = exc_type("wrapped")
    exc.__cause__ = cause
    return exc


# ── the one call itself ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_carries_full_definition_query_and_settled_values():
    llm = _FakeLLM(_match({"name": "Notes"}))

    await cve.verify_and_extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        window=_window(), values={"color": "red"},
    )

    prompt, _kwargs = llm.calls[0]
    assert "name (string, required" in prompt
    assert "color (string, optional" in prompt
    assert "color: 'red'" in prompt  # what the funnel already settled
    assert "<user_message>make a folder called Notes</user_message>" in prompt
    assert "cap-create-folder" in prompt
    assert "create_folder" in prompt


@pytest.mark.asyncio
async def test_prompt_carries_the_conversation_window():
    llm = _FakeLLM(_match({"name": "Notes"}))
    window = _window(
        {"role": "user", "content": "I keep my work in folders"},
        {"role": "assistant", "content": "Which name should the folder use?"},
    )

    await cve.verify_and_extract(llm=llm, query="call it Notes", entry=_entry(),
                                 window=window)

    prompt, _kwargs = llm.calls[0]
    assert "user: I keep my work in folders" in prompt
    assert "assistant: Which name should the folder use?" in prompt


@pytest.mark.asyncio
async def test_the_call_reuses_the_caller_channel_and_the_verify_guardrail():
    llm = _FakeLLM(_match({"name": "Notes"}))

    await cve.verify_and_extract(llm=llm, query="make a folder", entry=_entry(),
                                 window=_window())

    _prompt, kwargs = llm.calls[0]
    assert set(kwargs) == {"system_prompt", "timeout", "max_tokens",
                           "disable_thinking", "usage_out"}
    assert kwargs["timeout"] == settings.chat_capability_verify_timeout_seconds
    assert kwargs["disable_thinking"] is True
    assert "model" not in kwargs and "base_url" not in kwargs and "api_key" not in kwargs


@pytest.mark.asyncio
async def test_exactly_one_call_per_fallback():
    llm = _FakeLLM(_match({"name": "Notes"}))
    await cve.verify_and_extract(llm=llm, query="make a folder", entry=_entry(),
                                 window=_window())
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_a_complete_only_client_is_never_used():
    llm = _CompleteOnlyLLM()
    result = await cve.verify_and_extract(
        llm=llm, query="make a folder", entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_UNEXPECTED
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_an_empty_or_missing_client_is_an_error():
    for llm in (None, _CompleteOnlyLLM()):
        result = await cve.verify_and_extract(
            llm=llm, query="make a folder", entry=_entry(), window=_window())
        assert result.ok is False
        assert result.error_code == cve.ERROR_UNEXPECTED


# ── MATCH / REJECT verdicts ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [cve.REASON_CAPABILITY_MATCH,
                                    cve.REASON_PARAM_SUPPLEMENT])
async def test_both_match_reason_codes_map_to_match(reason):
    llm = _FakeLLM(_match({"name": "Notes"}, reason=reason))
    result = await cve.verify_and_extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        window=_window())
    assert result.ok is True
    assert result.decision == cve.DECISION_MATCH
    assert result.reason_code == reason
    assert result.arguments == {"name": "Notes"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [cve.REASON_FOLLOWUP_NOT_ACTION,
                                    cve.REASON_OTHER_INTENT])
async def test_both_reject_reason_codes_map_to_reject(reason):
    llm = _FakeLLM(_reject(reason))
    result = await cve.verify_and_extract(
        llm=llm, query="总结一下", entry=_entry(), window=_window())
    assert result.ok is True
    assert result.decision == cve.DECISION_REJECT
    assert result.reason_code == reason
    assert result.arguments == {}


@pytest.mark.asyncio
async def test_match_may_omit_a_required_slot():
    """Capability match and argument completeness are independent."""
    llm = _FakeLLM(_match({}))
    result = await cve.verify_and_extract(
        llm=llm, query="创建一个文件夹", entry=_entry(), window=_window())
    assert result.ok is True
    assert result.decision == cve.DECISION_MATCH
    assert result.arguments == {}


# ── schema discipline ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    {"decision": "MATCH", "reason_code": "FOLLOWUP_NOT_ACTION", "arguments": {}},
    {"decision": "MATCH", "reason_code": "OTHER_INTENT", "arguments": {}},
    {"decision": "REJECT", "reason_code": "CAPABILITY_MATCH", "arguments": {}},
    {"decision": "REJECT", "reason_code": "PARAM_SUPPLEMENT", "arguments": {}},
])
async def test_conflicting_decision_and_reason_are_schema_errors(reply):
    result = await cve.verify_and_extract(
        llm=_FakeLLM(reply), query="make a folder", entry=_entry(),
        window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_SCHEMA


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    {"decision": "MAYBE", "reason_code": "CAPABILITY_MATCH", "arguments": {}},
    {"decision": "MATCH", "reason_code": "WHATEVER", "arguments": {}},
    {"reason_code": "CAPABILITY_MATCH", "arguments": {}},
    {"decision": "MATCH", "arguments": {}},
    {"decision": "MATCH", "reason_code": "CAPABILITY_MATCH"},
])
async def test_unknown_enums_and_missing_fields_are_schema_errors(reply):
    result = await cve.verify_and_extract(
        llm=_FakeLLM(reply), query="make a folder", entry=_entry(),
        window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_SCHEMA


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [
    "not-an-object",
    {"name": 7},                      # string slot, wrong type
    {"unknown_slot": "x"},            # not a declared parameter
    {"name": None},                   # null is not a string
])
async def test_argument_structure_is_validated(arguments):
    result = await cve.verify_and_extract(
        llm=_FakeLLM(_match(arguments)), query="make a folder", entry=_entry(),
        window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_SCHEMA


@pytest.mark.asyncio
async def test_reject_must_not_carry_arguments():
    result = await cve.verify_and_extract(
        llm=_FakeLLM(_reject(args={"name": "x"})), query="总结一下",
        entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_SCHEMA


@pytest.mark.asyncio
async def test_enum_values_are_enforced():
    params = {"mode": {"type": "string", "required": True, "enum": ["fast", "slow"]}}
    ok = await cve.verify_and_extract(
        llm=_FakeLLM(_match({"mode": "fast"})), query="run fast",
        entry=_entry(params), window=_window())
    assert ok.ok is True
    bad = await cve.verify_and_extract(
        llm=_FakeLLM(_match({"mode": "other"})), query="run fast",
        entry=_entry(params), window=_window())
    assert bad.ok is False
    assert bad.error_code == cve.ERROR_SCHEMA


# ── technical error taxonomy ─────────────────────────────────────────────────


class _StatusError(Exception):
    status_code = 502


class _TimeoutError(Exception):
    pass


_TimeoutError.__name__ = "APITimeoutError"


@pytest.mark.asyncio
async def test_timeout_maps_to_timeout():
    result = await cve.verify_and_extract(
        llm=_FakeLLM(error=TimeoutError()), query="make a folder",
        entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_TIMEOUT


@pytest.mark.asyncio
async def test_a_wrapped_provider_timeout_maps_to_timeout():
    exc = _wrapped(LLMTemporaryError, _TimeoutError())
    result = await cve.verify_and_extract(
        llm=_FakeLLM(error=exc), query="make a folder", entry=_entry(),
        window=_window())
    assert result.error_code == cve.ERROR_TIMEOUT


@pytest.mark.asyncio
async def test_http_status_error_maps_to_http_error():
    result = await cve.verify_and_extract(
        llm=_FakeLLM(error=_StatusError()), query="make a folder",
        entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_HTTP


@pytest.mark.asyncio
async def test_invalid_json_maps_to_invalid_json():
    result = await cve.verify_and_extract(
        llm=_FakeLLM(error=ValueError("model did not return valid JSON")),
        query="make a folder", entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_INVALID_JSON


@pytest.mark.asyncio
async def test_non_object_reply_maps_to_schema_error():
    result = await cve.verify_and_extract(
        llm=_FakeLLM(error=TypeError("model did not return a JSON object")),
        query="make a folder", entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_SCHEMA


@pytest.mark.asyncio
async def test_an_unexpected_error_maps_to_unexpected():
    result = await cve.verify_and_extract(
        llm=_FakeLLM(error=RuntimeError("boom")), query="make a folder",
        entry=_entry(), window=_window())
    assert result.ok is False
    assert result.error_code == cve.ERROR_UNEXPECTED


# ── deterministic history window ─────────────────────────────────────────────


def test_user_turns_are_verbatim_and_assistant_is_abbreviated():
    long_answer = "A" * 1000
    window = history_window.build([
        {"role": "user", "content": "  keep  this  verbatim  "},
        {"role": "assistant", "content": long_answer},
    ])
    assert "user: keep  this  verbatim" in window.text
    assert "…[" in window.text and "characters omitted]…" in window.text
    assert window.text.endswith("A" * history_window.ASSISTANT_TAIL_CHARS)
    assert window.truncated is True


def test_the_window_takes_user_turns_not_mechanically_the_last_two_messages():
    history = [{"role": "user", "content": f"u{i}"} for i in range(8)]
    window = history_window.build(history)
    assert window.user_turns == history_window.MAX_USER_TURNS
    assert "u3" in window.text and "u2" not in window.text
    assert window.truncated is True


def test_the_budget_drops_the_oldest_turns_first():
    history = [{"role": "user", "content": "old " + "x" * 900},
               {"role": "user", "content": "new"}]
    window = history_window.build(history, budget=200)
    assert "new" in window.text
    assert "old" not in window.text
    assert window.truncated is True


def test_a_short_history_is_not_truncated():
    window = history_window.build([
        {"role": "user", "content": "make a folder"},
        {"role": "assistant", "content": "What name?"},
    ])
    assert window.truncated is False


# ── the post-funnel seam ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_certified_turn_never_calls_the_llm(monkeypatch):
    """Qwen settled the turn -> the funnel cleared the handoff -> no LLM call."""
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({"name": "WRONG"}))

    out = await orchestrator.resume_with_llm(
        _ctx(pending=None), deps=types.SimpleNamespace(llm=llm),
        requirements=TurnRequirements())

    assert out is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_match_certifies_through_the_same_binder(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({"name": "Notes"}))
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
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None
    assert "intent_funnel" not in ctx.agent_context


@pytest.mark.asyncio
async def test_match_merges_the_settled_args(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({"name": "Notes"}))
    ctx = _ctx(pending=_pending(args={"color": "red"},
                                arg_sources={"color": "handler"}))

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is not None
    assert out.requested_action["args"] == {"color": "red", "name": "Notes"}


@pytest.mark.asyncio
async def test_match_with_missing_args_keeps_the_capability_note(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({}))
    ctx = _ctx(pending=_pending(), message="make a folder")

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    failure = ctx.intent_funnel_failure
    assert failure["reason"] == "BIND_MISSING"
    assert failure["capability_id"] == "cap-create-folder"
    assert "name" in failure["missing"]
    note = render_intent_funnel_context(ctx.agent_context["intent_funnel"])
    assert note.startswith("## Intent Funnel routing note")
    assert "cap-create-folder" in note


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    _reject(cve.REASON_FOLLOWUP_NOT_ACTION),
    _reject(cve.REASON_OTHER_INTENT),
])
async def test_reject_hands_over_with_no_capability_trace(monkeypatch, reply):
    _patch_view(monkeypatch)
    llm = _FakeLLM(reply)
    stale = {"capability_id": "cap-create-folder", "reason": "ACQUISITION_MODEL_PENDING"}
    ctx = _ctx(pending=_pending(), failure=stale,
               agent_context={"intent_funnel": stale, "viewer": {"x": 1}})

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None
    assert "intent_funnel" not in ctx.agent_context
    assert ctx.agent_context["viewer"] == {"x": 1}  # other context preserved
    turn = types.SimpleNamespace(context=ctx.agent_context)
    assert await intent_funnel_reference_section({"turn": turn}) == ""


@pytest.mark.asyncio
async def test_error_hands_over_with_no_capability_trace(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(error=TimeoutError())
    stale = {"capability_id": "cap-create-folder", "reason": "ACQUISITION_MODEL_PENDING"}
    ctx = _ctx(pending=_pending(), failure=stale,
               agent_context={"intent_funnel": stale})

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None
    assert "intent_funnel" not in ctx.agent_context
    assert ctx.capability_verify_decision == cve.ERROR_TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [_reject(), None])
async def test_reject_and_error_never_reach_the_binder(monkeypatch, reply):
    _patch_view(monkeypatch)
    if reply is None:
        llm = _FakeLLM(error=RuntimeError("down"))
    else:
        llm = _FakeLLM(reply)

    def _explode(*a, **kw):  # pragma: no cover - asserted by not being called
        raise AssertionError("the Binder must not run")

    monkeypatch.setattr(orchestrator, "_certify", _explode)
    out = await orchestrator.resume_with_llm(
        _ctx(pending=_pending()), deps=types.SimpleNamespace(llm=llm,
                                                            session_factory=None),
        requirements=TurnRequirements())
    assert out is None


@pytest.mark.asyncio
async def test_no_llm_client_clears_the_capability_context(monkeypatch):
    _patch_view(monkeypatch)
    stale = {"capability_id": "cap-create-folder", "reason": "ACQUISITION_MODEL_PENDING"}
    ctx = _ctx(pending=_pending(), failure=stale,
               agent_context={"intent_funnel": stale})

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=None), requirements=TurnRequirements())

    assert out is None
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None
    assert "intent_funnel" not in ctx.agent_context


@pytest.mark.asyncio
async def test_history_reaches_the_verify_prompt(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({"name": "Notes"}))
    ctx = _ctx(pending=_pending(), message="call it Notes", history=[
        {"role": "user", "content": "make a folder"},
        {"role": "assistant", "content": "Which name should it use?"},
    ])

    await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    prompt, _kwargs = llm.calls[0]
    assert "make a folder" in prompt
    assert "Which name should it use?" in prompt
    assert "<user_message>call it Notes</user_message>" in prompt


@pytest.mark.asyncio
async def test_an_unadmitted_capability_is_not_resumed(monkeypatch):
    """The table no longer honors the pick -> the Agent decides (no LLM call)."""
    _patch_view(monkeypatch, entries=[])
    llm = _FakeLLM(_match({"name": "Notes"}))

    out = await orchestrator.resume_with_llm(
        _ctx(pending=_pending()), deps=types.SimpleNamespace(llm=llm),
        requirements=TurnRequirements())

    assert out is None
    assert llm.calls == []


# ── the mandated real-world cases (verbatim user queries) ─────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["做一下扩展", "总结这份文档"])
async def test_a_non_action_message_against_cap_create_folder_is_rejected(
        monkeypatch, query):
    """A nearby Recall hit is not a match: the verdict clears the candidate."""
    _patch_view(monkeypatch)
    llm = _FakeLLM(_reject(cve.REASON_OTHER_INTENT))
    stale = {"capability_id": "cap-create-folder", "reason": "BIND_MISSING"}
    ctx = _ctx(pending=_pending(query=query), message=query, failure=stale,
               agent_context={"intent_funnel": stale})

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    assert llm.calls[0][0].count(query) == 1  # the query reaches the call verbatim
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None
    assert "intent_funnel" not in ctx.agent_context
    assert ctx.capability_verify_decision == cve.DECISION_REJECT


@pytest.mark.asyncio
async def test_create_a_folder_with_no_name_is_match_then_agent(monkeypatch):
    """Correct capability + missing argument is MATCH (not REJECT)."""
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({}))
    ctx = _ctx(pending=_pending(query="创建一个文件夹"), message="创建一个文件夹")

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is None
    assert ctx.intent_funnel_failure["reason"] == "BIND_MISSING"
    assert ctx.intent_funnel_failure["capability_id"] == "cap-create-folder"
    note = render_intent_funnel_context(ctx.agent_context["intent_funnel"])
    assert "cap-create-folder" in note
    assert "name" in note


@pytest.mark.asyncio
async def test_create_a_named_folder_is_match_then_action(monkeypatch):
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({"name": "项目笔记"}))
    ctx = _ctx(pending=_pending(query="创建名为项目笔记的文件夹"),
               message="创建名为项目笔记的文件夹")

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is not None
    assert out.requested_action["args"] == {"name": "项目笔记"}
    assert ctx.intent_funnel_pending is None
    assert ctx.intent_funnel_failure is None
    assert "intent_funnel" not in ctx.agent_context


@pytest.mark.asyncio
async def test_the_name_answer_to_the_assistant_is_a_parameter_supplement(
        monkeypatch):
    """Turn N-1 asked for the name; turn N is only the name -> MATCH."""
    _patch_view(monkeypatch)
    llm = _FakeLLM(_match({"name": "项目笔记"},
                          reason=cve.REASON_PARAM_SUPPLEMENT))
    history = [
        {"role": "user", "content": "创建一个文件夹"},
        {"role": "assistant", "content": "文件夹叫什么名字？请告诉我名称。"},
    ]
    ctx = _ctx(pending=_pending(query="项目笔记"), message="项目笔记",
               history=history)

    out = await orchestrator.resume_with_llm(
        ctx, deps=types.SimpleNamespace(llm=llm, session_factory=None),
        requirements=TurnRequirements())

    assert out is not None
    assert out.requested_action["args"] == {"name": "项目笔记"}
    prompt, _kwargs = llm.calls[0]
    assert "文件夹叫什么名字" in prompt       # the assistant's question
    assert "创建一个文件夹" in prompt          # the earlier user turn
    assert "<user_message>项目笔记</user_message>" in prompt


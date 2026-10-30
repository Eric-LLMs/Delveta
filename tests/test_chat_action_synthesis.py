"""Query capabilities on the ACTION lane synthesize an answer from retrieved material.

The lane dispatches the bound tool directly (no ReAct loop). For query tools the
tool output is MATERIAL — titles, URLs, snippets — so the executor must run exactly
ONE synthesis completion over (question + material) before returning. The tests pin
both sides of that boundary:

  * web_search / rag_search / search_social ⇒ synthesized answer;
  * side-effect tools (create_folder / edit_file / bash) ⇒ deterministic output, no
    model call;
  * tool denial / state-unknown / a failed or empty synthesis ⇒ never a fake success:
    the raw material (or the honest terminal message) is returned unchanged;
  * the synthesis streams: one content frame per model delta, and the joined frames
    are byte-identical to done.answer and the saved assistant row;
  * the synthesis call carries no model/channel override, so the turn's pinned
    channel resolution is the one that applies.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from core.application.chat.execution_plan import ExecutionPlan, PlanKind
from core.application.chat.executors.action import ActionExecutor
from core.application.chat.executors.answer_synthesis import (
    SYNTHESIS_TOOLS,
    _MAX_MATERIAL_CHARS,
    build_prompt,
)
from core.application.chat.executors.base import ChatDeps, TurnRequest
from core.application.chat.intent_funnel.registry.entry import (
    STATUS_ACTIVE,
    CapabilityEntry,
    RegistryLiveView,
)

QUERY_TOOLS = ("web_search", "rag_search", "search_social")
SIDE_EFFECT_TOOLS = ("create_folder", "edit_file", "bash")


def _entry(cid: str, tool: str) -> CapabilityEntry:
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, enabled=True,
        status=STATUS_ACTIVE, intent_kind="action")


_VIEW = RegistryLiveView(
    fingerprint="live1-synthesis-test",
    entries=(
        _entry("cap-web-search", "web_search"),
        _entry("cap-rag-search", "rag_search"),
        _entry("cap-social-search", "search_social"),
        _entry("cap-create-folder", "create_folder"),
        _entry("cap-edit-file", "edit_file"),
        _entry("cap-bash", "bash"),
    ),
)


def _schema(name: str, slot: str) -> dict:
    return {"name": name, "description": "d", "parameters": {
        "type": "object",
        "properties": {slot: {"type": "string", "maxLength": 2000}},
        "required": [slot],
    }}


_FAKE_KERNEL = SimpleNamespace(runtime=SimpleNamespace(schemas=lambda: [
    _schema("web_search", "query"),
    _schema("rag_search", "query"),
    _schema("search_social", "query"),
    _schema("create_folder", "name"),
    _schema("edit_file", "path"),
    _schema("bash", "command"),
]))

_ARGS = {"query": "北京今天的天气怎么样?", "name": "x", "path": "a.md",
         "command": "ls"}
_CAP = {"web_search": "cap-web-search", "rag_search": "cap-rag-search",
        "search_social": "cap-social-search", "create_folder": "cap-create-folder",
        "edit_file": "cap-edit-file", "bash": "cap-bash"}


class _FakeLLM:
    def __init__(self, answer: str = "北京今天晴，25℃（来源：https://weather.example/beijing）",
                 error: Exception | None = None,
                 chunks: list[str] | None = None,
                 fail_after: int | None = None):
        self.answer = answer
        self.error = error
        self.chunks = list(chunks) if chunks is not None else _split(answer)
        # Raise on the Nth delta (0 = before the first one).
        self.fail_after = fail_after
        self.calls: list[dict] = []

    def _record(self, prompt, system_prompt, kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt,
                           "kwargs": kwargs})

    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        self._record(prompt, system_prompt, kwargs)
        if self.error is not None:
            raise self.error
        return self.answer

    async def complete_stream(self, prompt: str, system_prompt: str = "", **kwargs):
        self._record(prompt, system_prompt, kwargs)
        if self.error is not None:
            raise self.error
        for idx, piece in enumerate(self.chunks):
            if self.fail_after is not None and idx >= self.fail_after:
                raise RuntimeError("upstream 500 mid-stream")
            yield piece


def _split(text: str, size: int = 8) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


class _SM:
    def __init__(self):
        self.rows = []

    async def append_message(self, role, content):
        self.rows.append((role, content))


@pytest.fixture(autouse=True)
def _live_registry(monkeypatch):
    async def fake_active(*, session_factory=None):
        return _VIEW

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    # edit_file / bash are hidden from the chat plane by default; this file tests the
    # post-dispatch synthesis boundary, so admit every entry in its own view.
    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.entry._hidden_capability_ids",
        lambda: frozenset())
    return _VIEW


def _req(tool: str, run_tool, llm):
    ctx = SimpleNamespace(
        user_text="北京今天的天气怎么样?", session_memory=_SM(), history=[])
    deps = ChatDeps(
        session_factory=None, queue=None, drive=None, agent=_FAKE_KERNEL, llm=llm,
        embedder=None, viewer=None, new_approval_bridge=None, persist_turn_meta=None,
        log_usage=None, resolve_research=None, run_tool=run_tool,
    )
    plan = ExecutionPlan(kind=PlanKind.ACTION, action={
        "tool": tool, "args": _ARGS, "capability_id": _CAP[tool]})
    return TurnRequest(ctx=ctx, deps=deps, plan=plan)


async def _events_req(req) -> tuple[list[dict], dict]:
    events = []
    async for evt in ActionExecutor().stream(req, progress_sink=lambda e: None):
        events.append(evt)
    done = next((e["data"] for e in events if e["type"] == "done"), {})
    return events, done


async def _events(tool: str, run_tool, llm) -> tuple[list[dict], dict]:
    return await _events_req(_req(tool, run_tool, llm))


async def _answer(tool: str, run_tool, llm) -> str:
    events, _done = await _events(tool, run_tool, llm)
    return "".join(e["data"] for e in events if e["type"] == "content")


def _tool(output: str):
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append((tool, args))
        return {"ok": True, "output": output}

    return run_tool, calls


RAW = ("1. 北京今天晴，最高 25℃\n"
       "   url: https://weather.example/beijing\n"
       "2. 北京空气质量良\n"
       "   url: https://aq.example/beijing")


@pytest.mark.parametrize("tool", QUERY_TOOLS)
async def test_query_tools_return_the_synthesized_answer(tool):
    run_tool, calls = _tool(RAW)
    llm = _FakeLLM()
    answer = await _answer(tool, run_tool, llm)

    assert answer == llm.answer
    assert calls and calls[0][0] == tool
    assert len(calls) == 1
    assert len(llm.calls) == 1


async def test_prompt_carries_question_material_and_source_rules():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM()
    await _answer("web_search", run_tool, llm)

    prompt, system = llm.calls[0]["prompt"], llm.calls[0]["system_prompt"]
    assert "北京今天的天气怎么样?" in prompt
    assert RAW in prompt  # material verbatim, provenance intact
    assert "https://weather.example/beijing" in prompt
    assert "never invent a source" in system
    assert "Preserve the sources" in system
    assert "same language" in system


async def test_synthesis_does_not_override_the_turn_channel():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM()
    await _answer("web_search", run_tool, llm)

    # No model/base_url/api_key kwarg: OpenAILLM resolves the request contextvar.
    assert llm.calls[0]["kwargs"] == {}


@pytest.mark.parametrize("tool", SIDE_EFFECT_TOOLS)
async def test_side_effect_tools_never_call_the_model(tool):
    run_tool, calls = _tool("Done.")
    llm = _FakeLLM()

    assert await _answer(tool, run_tool, llm) == "Done."
    assert len(calls) == 1
    assert llm.calls == []


async def test_failed_synthesis_falls_back_to_the_raw_material():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM(error=RuntimeError("upstream 500"))

    assert await _answer("web_search", run_tool, llm) == RAW


async def test_empty_synthesis_falls_back_to_the_raw_material():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM(answer="   ")

    assert await _answer("web_search", run_tool, llm) == RAW


async def test_missing_llm_seam_falls_back_to_the_raw_material():
    run_tool, _ = _tool(RAW)

    assert await _answer("web_search", run_tool, None) == RAW


async def test_decided_denial_is_terminal_and_never_synthesized():
    llm = _FakeLLM()

    async def run_tool(tool, args, ctx):
        return {"ok": False, "reason": "approval denied"}

    answer = await _answer("web_search", run_tool, llm)
    assert answer.startswith("Could not complete that request:")
    assert "approval denied" in answer
    assert llm.calls == []


async def test_state_unknown_is_honest_and_never_synthesized():
    llm = _FakeLLM()

    async def run_tool(tool, args, ctx):
        raise RuntimeError("socket closed mid-write")

    answer = await _answer("web_search", run_tool, llm)
    assert "could not be confirmed" in answer
    assert llm.calls == []


def test_prompt_truncates_pathological_material_with_an_explicit_marker():
    huge = "x" * (_MAX_MATERIAL_CHARS + 10)
    prompt = build_prompt("q", "web_search", huge)
    assert "[material truncated]" in prompt
    assert prompt.count("x") == _MAX_MATERIAL_CHARS


# ── streaming: the answer arrives as the model writes it ───────────────────────


@pytest.mark.parametrize("tool", QUERY_TOOLS)
async def test_query_answer_streams_as_multiple_content_events(tool):
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM()

    events, _done = await _events(tool, run_tool, llm)

    content = [e["data"] for e in events if e["type"] == "content"]
    assert len(content) > 1  # one frame per model delta, not one whole answer
    assert "".join(content) == llm.answer
    assert len(llm.calls) == 1


async def test_streamed_frames_done_answer_and_saved_row_are_the_same_text():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM()
    req = _req("web_search", run_tool, llm)

    events, done = await _events_req(req)

    streamed = "".join(e["data"] for e in events if e["type"] == "content")
    assert streamed == done["answer"] == llm.answer
    assert done["messages"][-1] == {"role": "assistant", "content": done["answer"]}
    assert req.ctx.session_memory.rows[-1] == ("assistant", done["answer"])
    # The user row is still written before any output.
    assert req.ctx.session_memory.rows[0] == ("user", req.ctx.user_text)


async def test_a_midstream_failure_keeps_the_prefix_and_never_appends_the_material():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM(fail_after=2)

    events, done = await _events("web_search", run_tool, llm)

    content = [e["data"] for e in events if e["type"] == "content"]
    assert content == llm.chunks[:2]
    assert done["answer"] == "".join(llm.chunks[:2])
    assert RAW not in done["answer"]  # the raw material is never spliced on


async def test_a_stream_failing_before_the_first_delta_falls_back_to_the_material():
    run_tool, _ = _tool(RAW)
    llm = _FakeLLM(fail_after=0)

    assert await _answer("web_search", run_tool, llm) == RAW


async def test_a_seam_without_a_streaming_method_answers_in_one_frame():
    run_tool, _ = _tool(RAW)

    class _CompleteOnly:
        answer = "合成后的答案"

        async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
            return self.answer

    events, done = await _events("web_search", run_tool, _CompleteOnly())

    content = [e["data"] for e in events if e["type"] == "content"]
    assert content == [_CompleteOnly.answer]
    assert done["answer"] == _CompleteOnly.answer


def test_registry_set_is_exactly_the_query_tools():
    assert SYNTHESIS_TOOLS == {"web_search", "rag_search", "search_social"}


# ── a tool that WRITES its answer streams through the turn's content sink ──────


_SUMMARY_FRAMES = [
    "# Deep Dive\n\n", "## Executive Summary\n", "The document explains retrieval.\n"]
_SUMMARY_TEXT = "".join(_SUMMARY_FRAMES)


def _summary_req(run_tool):
    """A certified ACTION turn for ``summary_gen`` (not a query tool: no synthesis)."""
    ctx = SimpleNamespace(
        user_text="总结一下", session_memory=_SM(), history=[])
    deps = ChatDeps(
        session_factory=None, queue=None, drive=None,
        agent=SimpleNamespace(runtime=SimpleNamespace(schemas=lambda: [{
            "name": "summary_gen", "description": "d", "parameters": {
                "type": "object",
                "properties": {"context": {"type": "string", "maxLength": 2000}},
                "required": ["context"]},
        }])),
        llm=None, embedder=None, viewer=None, new_approval_bridge=None,
        persist_turn_meta=None, log_usage=None, resolve_research=None,
        run_tool=run_tool,
    )
    plan = ExecutionPlan(kind=PlanKind.ACTION, action={
        "tool": "summary_gen", "args": {"context": "总结一下"},
        "capability_id": "cap-summary"})
    return TurnRequest(ctx=ctx, deps=deps, plan=plan)


@pytest.fixture
def _summary_view(monkeypatch):
    view = RegistryLiveView(
        fingerprint="live1-summary-stream",
        entries=(_entry("cap-summary", "summary_gen"),),
    )

    async def fake_active(*, session_factory=None):
        return view

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    return view


async def _events_with_sink(req) -> tuple[list[dict], list[dict], dict]:
    """(progress-sink events, yielded events, done) — the orchestrator pumps both."""
    sunk: list[dict] = []
    events = []
    async for evt in ActionExecutor().stream(req, progress_sink=sunk.append):
        events.append(evt)
    done = next((e["data"] for e in events if e["type"] == "done"), {})
    return sunk, events, done


async def test_a_writing_tool_streams_its_deltas_before_it_returns(_summary_view):
    from core.infrastructure.request_context import get_request_content_sink

    async def run_tool(tool, args, ctx):
        sink = get_request_content_sink()
        assert sink is not None  # installed for exactly this call
        for piece in _SUMMARY_FRAMES:
            sink(piece)
        return {"ok": True, "output": _SUMMARY_TEXT}

    req = _summary_req(run_tool)
    sunk, events, done = await _events_with_sink(req)

    content = [e["data"] for e in sunk if e["type"] == "content"]
    assert content == _SUMMARY_FRAMES                  # not one finished string
    assert [e["data"] for e in events if e["type"] == "content"] == []
    assert done["answer"] == _SUMMARY_TEXT
    assert req.ctx.session_memory.rows[-1] == ("assistant", _SUMMARY_TEXT)
    # The sink never outlives the tool call.
    assert get_request_content_sink() is None


async def test_a_writing_tool_failing_midway_keeps_the_streamed_prefix(_summary_view):
    async def run_tool(tool, args, ctx):
        from core.infrastructure.request_context import get_request_content_sink

        sink = get_request_content_sink()
        for piece in _SUMMARY_FRAMES[:2]:
            sink(piece)
        raise RuntimeError("upstream 500")

    sunk, _events_seen, done = await _events_with_sink(_summary_req(run_tool))

    assert [e["data"] for e in sunk if e["type"] == "content"] == _SUMMARY_FRAMES[:2]
    assert done["answer"] == "".join(_SUMMARY_FRAMES[:2])


async def test_a_writing_tool_that_returns_one_string_answers_in_one_frame(_summary_view):
    async def run_tool(tool, args, ctx):
        return {"ok": True, "output": _SUMMARY_TEXT}

    events, done = await _events_req(_summary_req(run_tool))

    assert [e["data"] for e in events if e["type"] == "content"] == [_SUMMARY_TEXT]
    assert done["answer"] == _SUMMARY_TEXT

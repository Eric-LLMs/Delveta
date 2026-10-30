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
                 error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    async def complete(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt,
                           "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        return self.answer


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


async def _answer(tool: str, run_tool, llm) -> str:
    events = []
    async for evt in ActionExecutor().stream(_req(tool, run_tool, llm),
                                             progress_sink=lambda e: None):
        events.append(evt)
    content = [e for e in events if e["type"] == "content"]
    assert len(content) == 1
    return content[0]["data"]


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


def test_registry_set_is_exactly_the_query_tools():
    assert SYNTHESIS_TOOLS == {"web_search", "rag_search", "search_social"}

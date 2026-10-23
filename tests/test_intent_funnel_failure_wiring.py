"""Intent Funnel fallback context — the WIRING from the cascade to the kernel.

The structured record is only useful if it actually reaches ``AgentKernel.run``.
This module pins the production chain end to end, with no real model or tool:

    run_nodes (acquisition / Binder abstain)
      -> ctx.intent_funnel_failure + ctx.agent_context["intent_funnel"]
      -> TurnOrchestrator.resolve_plan (AGENT plan, context sunk)
      -> AgentExecutor.run (context=ctx.agent_context)

It also pins the executor's contract-mismatch branch through the SAME hop: the
record published before ``EscalateToAgent`` is what the orchestrator's fallback
call hands the Agent. The error categories stay distinct throughout:

* an acquisition failure is ``ACQUISITION_FAILURE`` (never ``ADMISSION_REJECT``);
* a Binder miss is ``CONTRACT_MISMATCH`` with the missing slots;
* an executor schema rejection is ``CONTRACT_MISMATCH`` at the executor stage;
* a preflight refusal is ``EXECUTION_PREFLIGHT``, not a parameter problem.

No test here executes a tool, calls a model, or loosens an admission rule.
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.execution_plan import ExecutionPlan, PlanKind
from core.application.chat.executors.action import ActionExecutor
from core.application.chat.executors.agent import AgentExecutor
from core.application.chat.executors.base import (
    ChatDeps,
    EscalateToAgent,
    TurnRequest,
)
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel import funnel as funnel_mod
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel.argument_acquisition.page_scope_resolver import (
    STATUS_INVALID,
    TARGET_DOCUMENT,
    PageScope,
)
from core.application.chat.intent_funnel.contract import MATCH_HIT, MatchResult
from core.application.chat.intent_funnel.observability import new_trace
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.turn_orchestrator import TurnOrchestrator
from core.application.chat.understanding import TurnRequirements
from core.config import settings


class _Kernel:
    """Captures the exact kwargs the executor hands the kernel."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, user_text, history, **kwargs):
        self.calls.append({"user_text": user_text, "history": history, **kwargs})
        return types.SimpleNamespace(
            final_answer="agent handled it", messages=[], usage={}, error=None)

    async def run_stream(self, *a, **kw):  # pragma: no cover - not exercised
        raise AssertionError("stream must not run in these cases")


def _deps(kernel, *, page_scope_resolver=None) -> ChatDeps:
    return ChatDeps(
        session_factory=None, queue=None, drive=None, agent=kernel, llm=None,
        embedder=lambda: object(), viewer=None, new_approval_bridge=None,
        persist_turn_meta=None, log_usage=None, resolve_research=None,
        run_tool=None, page_scope_resolver=page_scope_resolver,
    )


def _ctx(message: str, *, kernel_facing: bool = False):
    viewer = types.SimpleNamespace(asset_id="", page=None, selections=[])
    out = types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id="", research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
        agent_context=None, intent_funnel_failure=None,
    )
    if kernel_facing:
        out.user_text = message
        out.session_memory = object()
        out.model = "test-model"
        out.base_url = ""
        out.api_key = ""
        out.disable_thinking = False
    return out


def _entry(cid: str, *, tool: str, parameters: dict | None = None) -> CapabilityEntry:
    q = "打开这个文件"
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool,
        description="test capability",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=q,
                                      language=derive_language(q)),),
        parameters=dict(parameters or {}),
    )


def _wire_hit(monkeypatch, entry: CapabilityEntry) -> None:
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]),
                            entries=(entry,))

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="打开这个文件")

    class _NoSelect:
        async def select(self, query, candidates, *, entries_by_id, facts=None):
            raise AssertionError("the selector must never run on a HIT lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: _NoSelect())


@pytest.fixture(autouse=True)
def _deterministic_cascade(monkeypatch):
    async def no_event(*a, **kw):
        return None

    monkeypatch.setattr(funnel_mod, "_persist_event", no_event)
    monkeypatch.setattr(settings, "chat_funnel_trace_capture", False)


# ── acquisition failure -> plan -> kernel run(context=…) ─────────────────────────


async def test_acquisition_failure_reaches_the_kernel_context(monkeypatch):
    entry = _entry("cap-undeclared", tool="mystery_tool")
    _wire_hit(monkeypatch, entry)
    kernel = _Kernel()
    deps = _deps(kernel)
    ctx = _ctx("帮我处理这个", kernel_facing=True)
    orchestrator = TurnOrchestrator(deps)

    plan = await orchestrator.resolve_plan(ctx, deps)

    # The funnel abstained for an ACQUISITION reason and the plan fell open.
    assert plan.kind is PlanKind.AGENT
    record = ctx.agent_context["intent_funnel"]
    assert record is ctx.intent_funnel_failure or record == ctx.intent_funnel_failure
    assert record["kind"] == "ACQUISITION_FAILURE"
    assert record["reason"] == "ACQUISITION_UNDECLARED"
    assert record["stage"] == "acquisition"
    assert record["capability_id"] == "cap-undeclared"
    assert record["tool_binding"] == "mystery_tool"
    assert record["query"] == "帮我处理这个"
    assert record["execution_state"] == "not_started"
    # It is NOT relabelled as an admission reject.
    assert record["kind"] != "ADMISSION_REJECT"

    await AgentExecutor().run(TurnRequest(ctx=ctx, deps=deps, plan=plan))
    sent = kernel.calls[0]["context"]
    assert sent["intent_funnel"] == record
    assert kernel.calls[0]["user_text"] == "帮我处理这个"  # byte-identical replay


async def test_summary_contract_violation_reaches_the_agent_with_its_evidence(
        monkeypatch):
    """A handler-level acquisition exit rides the SAME kernel-facing record.

    ``cap-summary``'s scope model returned a contract-violating reply. The
    handler must NOT fall back to a page or the document: the Agent receives the
    category, stage, validation reason, resolver verdict, and the viewer's asset
    id and page — as real control flow, not logs.
    """
    entry = _entry("cap-summary", tool="summary_gen", parameters={})
    _wire_hit(monkeypatch, entry)
    kernel = _Kernel()

    async def bad_scope(**kw):
        return PageScope(target=TARGET_DOCUMENT, page_scope="range",
                         pages="9-5", status=STATUS_INVALID,
                         reason="pages must ascend")

    deps = _deps(kernel, page_scope_resolver=bad_scope)
    ctx = _ctx("总结第 5 页", kernel_facing=True)
    ctx.history = [
        {"role": "user", "content": "这是第 5 页"},
        {"role": "assistant", "content": "好的"},
    ]
    ctx.body.viewer = types.SimpleNamespace(
        asset_id="asset-123", page=7, page_from=5, page_to=9, selections=[])
    orchestrator = TurnOrchestrator(deps)

    plan = await orchestrator.resolve_plan(ctx, deps)
    assert plan.kind is PlanKind.AGENT

    record = ctx.agent_context["intent_funnel"]
    assert record["kind"] == "ACQUISITION_FAILURE"
    assert record["capability_id"] == "cap-summary"
    assert record["query"] == "总结第 5 页"
    acquisition = record["acquisition"]
    assert acquisition["category"] == "scope_contract_violation"
    assert acquisition["stage"] == "scope_resolution"
    assert "pages must ascend" in acquisition["detail"]
    assert acquisition["scope"]["status"] == STATUS_INVALID
    assert acquisition["scope"]["pages"] == "9-5"
    assert acquisition["context"]["viewer_asset_id"] == "asset-123"
    assert acquisition["context"]["viewer_current_page"] == 7

    await AgentExecutor().run(TurnRequest(ctx=ctx, deps=deps, plan=plan))
    call = kernel.calls[0]
    assert call["user_text"] == "总结第 5 页"        # the original request
    assert call["history"] == ctx.history           # the conversation context
    sent = call["context"]["intent_funnel"]
    assert sent["acquisition"] == acquisition       # byte-identical at the kernel


# ── a required model slot that the extractor cannot fill -> Agent ───────────────


async def test_summary_missing_output_dir_extraction_reaches_the_agent(
        monkeypatch):
    """An explicitly requested save destination that cannot be extracted is an
    acquisition failure, not a silent chat-only summary.

    ``cap-summary`` authorized ``output_dir`` because the sentence says "保存到",
    but no extractor is wired. The hop exits and the Agent receives the
    category, stage, and the turn context — through the real control flow.
    """
    entry = _entry(
        "cap-summary", tool="summary_gen",
        parameters={"path": {"type": "string", "required": False,
                             "description": "one file"},
                    "output_dir": {"type": "string", "required": False,
                                   "description": "directory"}})
    _wire_hit(monkeypatch, entry)
    kernel = _Kernel()
    deps = _deps(kernel)                      # no argument_extractor
    ctx = _ctx("总结 ./reports/q3.md 并保存到 outputs", kernel_facing=True)
    ctx.body.viewer = types.SimpleNamespace(
        asset_id="", page=3, page_from=None, page_to=None, selections=[])
    orchestrator = TurnOrchestrator(deps)

    plan = await orchestrator.resolve_plan(ctx, deps)
    assert plan.kind is PlanKind.AGENT

    record = ctx.agent_context["intent_funnel"]
    assert record["kind"] == "ACQUISITION_FAILURE"
    assert record["capability_id"] == "cap-summary"
    assert record["query"] == "总结 ./reports/q3.md 并保存到 outputs"
    acquisition = record["acquisition"]
    assert acquisition["category"] == "argument_extraction_failed"
    assert acquisition["stage"] == "argument_extraction"
    assert "output_dir" in acquisition["detail"]
    assert acquisition["context"]["viewer_current_page"] == 3

    await AgentExecutor().run(TurnRequest(ctx=ctx, deps=deps, plan=plan))
    sent = kernel.calls[0]["context"]["intent_funnel"]
    assert sent["acquisition"] == acquisition       # byte-identical at the kernel


# ── Binder contract mismatch -> the same Agent context ───────────────────────────


def test_binder_mismatch_is_published_with_diagnostics():
    entry = _entry(
        "cap-create-folder", tool="create_folder",
        parameters={"name": {"type": "string", "required": True}})
    view = RegistryLiveView(fingerprint="fp-binder", entries=(entry,))
    trace = new_trace()
    failure: dict = {}

    out = orch_mod._certify(
        TurnRequirements(), entry, {}, facts=None, view=view, trace=trace,
        capture=None, failure=failure, strategy="RULE_CONTEXT", arg_sources={})

    assert out is None
    assert trace["fallback"] == "BIND_MISSING"
    assert failure["kind"] == "CONTRACT_MISMATCH"
    assert failure["stage"] == "binder"
    assert failure["reason"] == "BIND_MISSING"
    assert failure["capability_id"] == "cap-create-folder"
    assert failure["missing"] == ["name"]
    assert failure["execution_state"] == "not_started"
    assert failure["detail"] == "binder state: MISSING"


# ── executor contract mismatch → EscalateToAgent → same context at the kernel ────


async def test_executor_contract_mismatch_reaches_the_agent_with_context(monkeypatch):
    entry = _entry(
        "cap-create-folder", tool="create_folder",
        parameters={"name": {"type": "string", "required": True}})
    view = RegistryLiveView(fingerprint="fp-exec", entries=(entry,))

    async def fake_active(**kw):
        return view

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)

    class _Runtime:
        def schemas(self):
            return [{"name": "create_folder", "description": "d", "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "maxLength": 120}},
                "required": ["name"],
            }}]

    kernel = _Kernel()
    kernel.runtime = _Runtime()  # type: ignore[attr-defined]
    deps = _deps(kernel)
    ctx = _ctx("新建文件夹", kernel_facing=True)
    action_plan = ExecutionPlan(kind=PlanKind.ACTION, action={
        "tool": "create_folder", "args": {},
        "capability_id": "cap-create-folder",
    })

    with pytest.raises(EscalateToAgent):
        await ActionExecutor().run(TurnRequest(ctx=ctx, deps=deps, plan=action_plan))

    record = ctx.agent_context["intent_funnel"]
    assert record["kind"] == "CONTRACT_MISMATCH"
    assert record["stage"] == "executor"
    assert record["reason"] == "EXECUTOR_PARAMETER_MISMATCH"
    assert record["missing"] == ["name"]
    assert record["query"] == "新建文件夹"
    assert record["execution_state"] == "not_started"

    # This is exactly the call the orchestrator makes after catching the escalation.
    agent_plan = ExecutionPlan(kind=PlanKind.AGENT)
    await AgentExecutor().run(TurnRequest(ctx=ctx, deps=deps, plan=agent_plan))
    sent = kernel.calls[0]["context"]
    assert sent["intent_funnel"] == record
    assert sent["intent_funnel"]["execution_state"] == "not_started"

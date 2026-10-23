"""ActionExecutor: staging, the final execution authorization, and the
side-effect boundary isolation.

Constraint 1 pinning, BOTH directions:

  pre-execution failure (authorization refusal / schema / seam-not-wired /
      ActionPreflightFailure) ⇒ no seam call; authorization + schema may escalate
      only for provable user-input problems, governance never does;
  post-entry failure (any other exception out of run_tool)
      ⇒ state UNKNOWN ⇒ ONE honest terminal message + normal done, the Agent is
      NEVER entered (a blind replay could duplicate the side effect);
  decided denial (ok:False) ⇒ terminal honest message, no escalation.

The final authorization is read from the LIVE Registry at dispatch time — never
from the action payload, the model's pick, or a routing stamp. A hidden, disabled
or unconfirmable capability is refused fail-closed.

Every executable case asserts the seam call count — the executor must call
run_tool EXACTLY once when it calls it at all.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from core.application.chat.actions import ActionPreflightFailure
from core.application.chat.execution_plan import ExecutionPlan, PlanKind
from core.application.chat.executors.action import ActionExecutor
from core.application.chat.executors.base import ChatDeps, EscalateToAgent, TurnRequest
from core.application.chat.intent_funnel.registry.entry import (
    STATUS_ACTIVE,
    STATUS_DISABLED,
    CapabilityEntry,
    RegistryLiveView,
)
from core.config import settings


def _entry(cid: str, tool: str, **over) -> CapabilityEntry:
    kw = {"capability_id": cid, "tool_binding": tool, "enabled": True,
          "status": STATUS_ACTIVE, "intent_kind": "action"}
    kw.update(over)
    return CapabilityEntry(**kw)


_VIEW = RegistryLiveView(
    fingerprint="live1-testview",
    entries=(
        _entry("cap-create-folder", "create_folder"),
        _entry("cap-summary", "summary_gen"),
        _entry("cap-rm", "rm_rf"),
        _entry("cap-hidden", "create_folder"),
        _entry("cap-disabled", "create_folder", enabled=False, status=STATUS_DISABLED),
    ),
)


@pytest.fixture(autouse=True)
def _live_registry(monkeypatch):
    """The authoritative view every dispatch reads unless a case overrides it."""
    async def fake_active(*, session_factory=None):
        return _VIEW

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    return _VIEW


class _SM:
    def __init__(self):
        self.rows = []

    async def append_message(self, role, content):
        self.rows.append((role, content))


def _req(action, run_tool):
    ctx = SimpleNamespace(
        user_text="新建文件夹\"x\"", session_memory=_SM(), history=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "noted"},
        ],
    )
    deps = ChatDeps(
        session_factory=None, queue=None, drive=None, agent=_FAKE_KERNEL, llm=None,
        embedder=None, viewer=None, new_approval_bridge=None, persist_turn_meta=None,
        log_usage=None, resolve_research=None, run_tool=run_tool,
    )
    plan = ExecutionPlan(kind=PlanKind.ACTION, action=action)
    return TurnRequest(ctx=ctx, deps=deps, plan=plan)


# the roster the executor gates on is deps.agent.runtime.schemas() (ruling
#: the live ToolRuntime is the ONE tool existence/schema truth) —
# the unit world fakes exactly the tools its cases dispatch.
_FAKE_KERNEL = SimpleNamespace(runtime=SimpleNamespace(schemas=lambda: [
    {"name": "create_folder", "description": "d",
     "parameters": {
         "type": "object",
         "properties": {"name": {"type": "string", "maxLength": 120},
                        "parent_path": {"type": "string"}},
         "required": ["name"]}},
    {"name": "summary_gen", "description": "d",
     "parameters": {
         "type": "object",
         "properties": {
             "paths": {"type": "array", "items": {"type": "string"}},
             "path": {"type": "string"},
             "asset_id": {"type": "string"},
             "pages": {"type": "string"},
             "context": {"type": "string"},
             "output_dir": {"type": "string"},
         },
         "required": []}},
]))


GOOD = {"tool": "create_folder", "args": {"name": "x"},
        "capability_id": "cap-create-folder"}


async def _drain(req):
    events = []
    async for evt in ActionExecutor().stream(req, progress_sink=lambda e: None):
        events.append(evt)
    return events


async def test_success_emits_deterministic_confirmation_without_llm():
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append((tool, args))
        return {"ok": True, "output": "Created folder 'x'."}

    req = _req(GOOD, run_tool)
    events = await _drain(req)
    assert calls == [("create_folder", {"name": "x"})]
    assert [e["type"] for e in events] == ["content", "done"]
    assert events[0]["data"] == "Created folder 'x'."
    done = events[1]["data"]
    assert done["answer"] == "Created folder 'x'." and done["usage"] == {} and done["error"] is None
    assert "refusal_reason" not in done
    assert done["messages"][:2] == [
        {"role": "user", "content": "earlier"}, {"role": "assistant", "content": "noted"},
    ]
    assert req.ctx.session_memory.rows == [
        ("user", "新建文件夹\"x\""), ("assistant", "Created folder 'x'."),
    ]


# ── final execution authorization (fail-closed) ──────────────────────────────────


async def test_hidden_capability_selected_by_model_cannot_execute(monkeypatch):
    """A hidden capability may be offered as a card; selecting/routing it never
    grants execution. The routing stamp is deliberately present and valid."""
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", "cap-hidden")
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append(tool)
        return {"ok": True, "output": "must not run"}

    action = {"tool": "create_folder", "args": {"name": "x"},
              "capability_id": "cap-hidden",
              "funnel_registry_version": _VIEW.fingerprint}
    events = await _drain(_req(action, run_tool))
    assert calls == []
    assert events[-1]["data"]["refusal_reason"] == "CAPABILITY_NOT_ALLOWLISTED"
    assert events[-1]["data"]["error"] is None


async def test_directly_constructed_hidden_action_cannot_execute(monkeypatch):
    """Bypassing the funnel does not bypass the execution allowlist."""
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", "cap-hidden")
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append(tool)
        return {"ok": True}

    action = {"tool": "create_folder", "args": {"name": "x"},
              "capability_id": "cap-hidden"}
    outcome = await ActionExecutor()._dispatch(_req(action, run_tool))
    assert outcome.refusal_reason == "CAPABILITY_NOT_ALLOWLISTED"
    assert calls == []


async def test_disabled_capability_after_selection_is_refused():
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append(tool)
        return {"ok": True}

    action = {"tool": "create_folder", "args": {"name": "x"},
              "capability_id": "cap-disabled",
              "funnel_registry_version": _VIEW.fingerprint}
    outcome = await ActionExecutor()._dispatch(_req(action, run_tool))
    assert outcome.refusal_reason == "CAPABILITY_INACTIVE"
    assert calls == []


@pytest.mark.parametrize("mode", ["raises", "empty"])
async def test_registry_unavailable_fails_closed(monkeypatch, mode):
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append(tool)
        return {"ok": True}

    if mode == "raises":
        async def boom(**kw):
            raise RuntimeError("registry db down")
        monkeypatch.setattr(
            "core.application.chat.intent_funnel.registry.active_view", boom)
    else:
        async def empty(**kw):
            return None
        monkeypatch.setattr(
            "core.application.chat.intent_funnel.registry.active_view", empty)

    events = await _drain(_req(GOOD, run_tool))
    assert calls == []
    assert events[-1]["data"]["refusal_reason"] == "REGISTRY_UNAVAILABLE"
    assert "not available" in events[0]["data"].lower()


async def test_unknown_capability_is_refused_without_fallback():
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append(tool)
        return {"ok": True}

    action = {"tool": "create_folder", "args": {"name": "x"},
              "capability_id": "cap-does-not-exist"}
    outcome = await ActionExecutor()._dispatch(_req(action, run_tool))
    assert outcome.refusal_reason == "CAPABILITY_NOT_REGISTERED"
    assert calls == []


async def test_tool_binding_divergence_is_refused():
    action = {"tool": "summary_gen", "args": {},
              "capability_id": "cap-create-folder"}
    outcome = await ActionExecutor()._dispatch(_req(action, lambda *a: None))
    assert outcome.refusal_reason == "TOOL_BINDING_MISMATCH"


async def test_offline_virtual_executor_seam_still_runs():
    """The evaluation chain substitutes only the execution seam. An allowlisted
    capability must still reach that seam exactly once, with no LLM involved."""
    class _VirtualExecutor:
        def __init__(self):
            self.calls = []

        async def __call__(self, tool, args, ctx):
            self.calls.append((tool, args))
            return {"ok": True, "output": f"virtual:{tool}"}

    virtual = _VirtualExecutor()
    events = await _drain(_req(GOOD, virtual))
    assert virtual.calls == [("create_folder", {"name": "x"})]
    assert events[0]["data"] == "virtual:create_folder"
    assert events[-1]["type"] == "done"


# ── pre-execution failures: no seam call ─────────────────────────────────────────


async def test_malformed_action_escalates_and_never_touches_the_seam():
    calls = []

    async def run_tool(*a):
        calls.append(a)
        return {"ok": True}

    # a roster tool with a missing REQUIRED slot is the user-input class: the
    # Agent owns the clarification (the roster itself says the tool exists).
    req = _req({**GOOD, "args": {}}, run_tool)
    with pytest.raises(EscalateToAgent, match="schema"):
        await _drain(req)
    assert calls == []  # pre-execution failure: the seam was NEVER entered
    # The Agent receives structured contract evidence, not only an error string.
    record = req.ctx.agent_context["intent_funnel"]
    assert record["kind"] == "CONTRACT_MISMATCH"
    assert record["reason"] == "EXECUTOR_PARAMETER_MISMATCH"
    assert record["capability_id"] == "cap-create-folder"
    assert record["tool_binding"] == "create_folder"
    assert record["missing"] == ["name"]
    assert record["args"] == {}
    assert record["required"]["name"]["required"] is True
    assert record["execution_state"] == "not_started"
    assert "missing slot: name" in record["original_error"]


# Shadow-A/B #17, revisited: cap-summary now publishes the real summary_gen
# slots (asset_id / path / pages / context / output_dir, none globally
# required), so a certified attachment summary reaches the seam with its
# arguments intact instead of dying at the schema gate.
async def test_certified_summary_asset_action_reaches_the_seam():
    calls = []

    async def run_tool(tool, args, ctx):
        calls.append((tool, args))
        return {"ok": True, "output": "# Summary"}

    ctx = SimpleNamespace(user_text="帮我总结一下这份文档", session_memory=_SM(),
                          history=[])
    deps = ChatDeps(
        session_factory=None, queue=None, drive=None, agent=_FAKE_KERNEL,
        llm=None, embedder=None, viewer=None, new_approval_bridge=None,
        persist_turn_meta=None, log_usage=None, resolve_research=None,
        run_tool=run_tool,
    )
    plan = ExecutionPlan(kind=PlanKind.ACTION,
                         action={"tool": "summary_gen",
                                 "args": {"asset_id": "asset-1", "pages": "5"},
                                 "capability_id": "cap-summary"})
    events = await _drain(TurnRequest(ctx=ctx, deps=deps, plan=plan))

    assert calls == [("summary_gen", {"asset_id": "asset-1", "pages": "5"})]
    assert events[0]["data"] == "# Summary"


async def test_tool_missing_from_live_roster_without_stamp_is_terminal_integrity():
    # Tool existence is a system-integrity fact: an action naming a tool the live
    # roster does not register terminates honestly — the Agent must never re-plan
    # around a missing executable. Without a routing stamp there is no "stale
    # route" to blame, so the message is the integrity one (the historical split).
    calls = []

    async def run_tool(*a):
        calls.append(a)
        return {"ok": True}

    action = {"tool": "rm_rf", "args": {}, "capability_id": "cap-rm"}
    events = await _drain(_req(action, run_tool))
    assert calls == []
    assert [e["type"] for e in events] == ["content", "done"]
    assert events[-1]["data"]["refusal_reason"] == "TOOL_NOT_ON_ROSTER"
    assert "not available" in events[0]["data"].lower()


async def test_missing_tool_runtime_fails_closed():
    calls = []

    async def run_tool(*a):
        calls.append(a)
        return {"ok": True}

    req = _req(GOOD, run_tool)
    req.deps.agent = SimpleNamespace(runtime=None)
    events = await _drain(req)
    assert calls == []
    assert events[-1]["data"]["refusal_reason"] == "TOOL_ROSTER_UNAVAILABLE"


async def test_seam_not_wired_terminates_as_integrity_never_escalates():
    # C2 (frozen boundary 2): a missing seam is an internal wiring fault, not a
    # user-input problem — the Agent must never serve as the recovery channel.
    req = _req(GOOD, None)
    events = await _drain(req)
    assert [e["type"] for e in events] == ["content", "done"]
    assert "not available" in events[0]["data"].lower()


async def test_preflight_failure_escalates_once_side_effect_free():
    hits = []

    async def run_tool(tool, args, ctx):
        hits.append(1)
        raise ActionPreflightFailure("preflight: domain not found")

    req = _req(GOOD, run_tool)
    with pytest.raises(EscalateToAgent, match="preflight"):
        await _drain(req)
    assert len(hits) == 1  # entered once, proven non-executing — escalation is legal
    record = req.ctx.agent_context["intent_funnel"]
    assert record["kind"] == "EXECUTION_PREFLIGHT"
    assert record["reason"] == "EXECUTOR_PREFLIGHT"
    assert record["execution_state"] == "not_started"
    assert "domain not found" in record["original_error"]


async def test_state_unknown_terminates_honestly_without_agent_context():
    hits = []

    async def run_tool(tool, args, ctx):
        hits.append(1)
        raise RuntimeError("connection reset mid-write")

    req = _req(GOOD, run_tool)
    events = await _drain(req)  # NO EscalateToAgent — the Agent must not replay a write
    assert len(hits) == 1       # called exactly once — no duplicate side effect
    assert events[1]["data"]["error"] is None
    # There is no fallback, so no "safe to retry" context may be published.
    assert "intent_funnel" not in (getattr(req.ctx, "agent_context", None) or {})


async def test_state_unknown_terminates_honestly_never_escalates():
    hits = []

    async def run_tool(tool, args, ctx):
        hits.append(1)
        raise RuntimeError("connection reset mid-write")

    req = _req(GOOD, run_tool)
    events = await _drain(req)  # NO EscalateToAgent — the Agent must not replay a write
    assert len(hits) == 1       # called exactly once — no duplicate side effect
    assert [e["type"] for e in events] == ["content", "done"]
    data = events[1]["data"]
    assert data["error"] is None and data["answer"]
    assert "could not be confirmed" in data["answer"].lower()
    # The user row + the honest assistant row are persisted like any answer.
    assert [r[0] for r in req.ctx.session_memory.rows] == ["user", "assistant"]


async def test_binding_integrity_marker_terminates_never_escalates():
    # Stage-2 (routing-layer) C2: resolve_plan stamps a binding_integrity marker;
    # the executor must issue the decided terminal BEFORE the schema gate (such an
    # action carries no args) and never hand the system fault to the Agent.
    hits = []

    async def run_tool(tool, args, ctx):
        hits.append(1)
        return {"ok": True}

    action = {"tool": "create_folder", "args": None,
              "capability_id": "cap-create-folder", "binding_integrity": "no binding"}
    req = _req(action, run_tool)
    events = await _drain(req)
    assert hits == []                                       # seam never entered
    assert [e["type"] for e in events] == ["content", "done"]
    assert events[-1]["data"]["refusal_reason"] == "BINDING_INTEGRITY"
    assert "not available" in events[0]["data"].lower()


async def test_decided_denial_is_terminal_not_an_escalation():
    async def run_tool(tool, args, ctx):
        return {"ok": False, "reason": "approval denied by the user"}

    events = await _drain(_req(GOOD, run_tool))
    assert events[0]["type"] == "content"
    assert "approval denied" in events[0]["data"]
    assert events[1]["type"] == "done" and events[1]["data"]["error"] is None


async def test_run_mirrors_stream_shape():
    async def run_tool(tool, args, ctx):
        return {"ok": True, "output": "Done."}

    req = _req(GOOD, run_tool)
    result = await ActionExecutor().run(req)
    assert result.final_answer == "Done." and result.usage == {} and result.error is None
    assert result.refusal_reason is None
    assert result.messages[-1] == {"role": "assistant", "content": "Done."}
    assert result.messages[-2] == {"role": "user", "content": "新建文件夹\"x\""}

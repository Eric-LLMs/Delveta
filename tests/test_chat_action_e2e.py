"""Host-seam contract for the certified ACTION lane.

The funnel/executor end-to-end path is covered by ``tests/test_funnel_e2e.py``.
What is pinned here is the one thing that must never regress in the HTTP adapter:
the synthetic ``ToolExecution`` handed to ``ToolRuntime.execute`` MUST carry
``agent=`` — the sandbox guard and the ASK listener resolve the tool definition
through ``exec.agent.runtime``. Without it the fast path would silently bypass the
permission/approval funnel the Agent path goes through.
"""
from __future__ import annotations

from types import SimpleNamespace

from api.routers import chat as chat_mod

MSG = 'create a folder named "archive"'


async def test_run_tool_binds_agent_so_the_sandbox_funnel_still_sees_the_tool(monkeypatch):
    captured = {}

    class _RT:
        async def execute(self, execution):
            captured["execution"] = execution
            return SimpleNamespace(is_error=False, value="Created.", error=None)

    agent = SimpleNamespace(runtime=_RT())
    monkeypatch.setattr(chat_mod, "get_agent", lambda: agent)
    ctx = SimpleNamespace(user_text=MSG, agent_context=None)

    out = await chat_mod._run_tool("create_folder", {"name": "archive"}, ctx)
    assert out == {"ok": True, "output": "Created."}
    assert captured["execution"].agent is agent           # guards resolve the tool
    assert captured["execution"].name == "create_folder"
    assert captured["execution"].arguments == {"name": "archive"}

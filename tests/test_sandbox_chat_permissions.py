"""Regression tests: chat turns run their tools with the surface's standing grant.

The chat workbench receives a request the user typed into it and runs the tools that
request selects (the intent funnel's ACTION lane and the Agent lane alike). Its client
only offers a binary Allow/Deny prompt with a 120 s timeout, so an ASK per call merely
interrupts the turn — ``summary_gen``, ``create_folder`` and every other write tool sat
behind it. The chat control plane therefore sinks ``context["tool_permissions"]``
(read/write/network) and the sandbox widens that turn's grant.

This is a grant, not a bypass:

* a turn without the marker (worker / research kernels) keeps the READ default;
* a DENY fence (source policy) is decided before the granted set is consulted;
* unknown permission names are ignored instead of widening the grant.
"""
from __future__ import annotations

import pytest

from agent.engine.context import AgentTurn, bind_turn
from agent.security.sandbox import Sandbox, SandboxDecision
from agent.tools.definition import ToolDefinition, ToolOutput
from agent.tools.tool_permissions import ToolPermission


def _tool(name: str, permission) -> ToolDefinition:
    async def execute(args, exec):  # noqa: ANN202
        return {}

    return ToolDefinition(
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        output=ToolOutput(),
        execute=execute,
        permission=permission,
    )


@pytest.fixture(autouse=True)
def _unbound_turn():
    bind_turn(None)
    yield
    bind_turn(None)


def _chat_turn(**context) -> AgentTurn:
    return AgentTurn(user_msg="hi", context={
        "tool_permissions": ("read", "write", "network"), **context})


def _summary() -> ToolDefinition:
    return _tool("summary_gen", {ToolPermission.READ, ToolPermission.WRITE})


def _create_folder() -> ToolDefinition:
    return _tool("create_folder", {ToolPermission.WRITE})


def _bash() -> ToolDefinition:
    return _tool("bash", {ToolPermission.WRITE, ToolPermission.NETWORK})


def test_chat_tools_run_without_an_ask():
    bind_turn(_chat_turn())
    sandbox = Sandbox()

    assert sandbox.check(_summary(), {}) is SandboxDecision.ALLOW
    assert sandbox.check(_create_folder(), {}) is SandboxDecision.ALLOW
    assert sandbox.check(_bash(), {}) is SandboxDecision.ALLOW


def test_without_the_chat_marker_the_session_stays_read_only():
    bind_turn(AgentTurn(user_msg="hi"))
    sandbox = Sandbox()

    assert sandbox.check(_summary(), {}) is SandboxDecision.ASK
    assert sandbox.check(_bash(), {}) is SandboxDecision.ASK


def test_a_source_policy_deny_beats_the_chat_grant():
    bind_turn(_chat_turn(source_policy="private_only"))
    sandbox = Sandbox()

    assert sandbox.check(_bash(), {}) is SandboxDecision.DENY


def test_unknown_permission_names_do_not_widen_the_grant():
    bind_turn(AgentTurn(user_msg="hi", context={"tool_permissions": ("admin",)}))
    sandbox = Sandbox()

    assert sandbox.check(_create_folder(), {}) is SandboxDecision.ASK

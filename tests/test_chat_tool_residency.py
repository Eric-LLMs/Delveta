"""Resident chat tools: the composition root's list and the gateway visibility contract.

``web_search`` used to be deferred — the model had to call ``tool_search`` before its
schema appeared, and with thinking off it never made that discovery hop at all (a
weather question was answered "I cannot check" with zero tool calls). Residency is the
fix; these tests pin both the list and the mechanism.
"""
from agent.engine.runtime import ToolRuntime
from agent.tools.tool_gateway import ToolGateway, ToolVisibilityPolicy

from api.agent_factory import RESIDENT_CHAT_TOOLS
from api.tools.web_search_tool import register as register_web


class _Ctx:
    def resolve(self, key):
        return None


def _gateway(*, allow):
    runtime = ToolRuntime()
    register_web(runtime, _Ctx(), None)
    gateway = ToolGateway(runtime, policy=ToolVisibilityPolicy())
    for name in allow:
        gateway.policy.allow(name)
    return gateway


def _visible(gateway):
    return {s["function"]["name"]: s for s in gateway.visible_schemas()}


def test_web_search_is_resident():
    assert "web_search" in RESIDENT_CHAT_TOOLS
    # The pre-existing residents stay resident; this change is additive.
    assert {"rag_search", "summary_gen", "mindmap_gen", "slides_gen"} <= set(RESIDENT_CHAT_TOOLS)


def test_resident_web_search_is_visible_from_step_zero_with_its_full_schema():
    schemas = _visible(_gateway(allow=RESIDENT_CHAT_TOOLS))
    assert "web_search" in schemas
    # Full parameters, not the deferred stub's empty shape.
    props = schemas["web_search"]["function"]["parameters"]["properties"]
    assert "query" in props


def test_a_deferred_tool_still_needs_tool_search_to_appear():
    # Control: the same tool WITHOUT residency stays invisible until mounted, so the
    # default discovery flow for every other tool is unchanged.
    gateway = _gateway(allow=())
    assert "web_search" not in _visible(gateway)
    assert gateway.mount("web_search") is True
    assert "web_search" in _visible(gateway)

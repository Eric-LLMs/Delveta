"""Allowlist roster + executor schema gate.

The DIRECT_TOOLS roster is FLOW CONTROL data over the existing tool registry,
never a second capability system. L0's lexical recogniser (``match_direct_tool``)
was deleted — capability recognition belongs to the Intent Funnel's Matcher; what
remains here is the roster contract and the executor's final schema gate.
"""
from __future__ import annotations

import pytest
from core.application.chat.actions import (
    DIRECT_TOOLS,
    ActionSchemaError,
    validate_action,
)


# ── registered-tool admission: every allowlisted tool must EXIST in the tool layer ──

def test_allowlist_is_converged_v1_set():
    assert set(DIRECT_TOOLS) == {"create_folder", "add_term", "pdf_extract_text"}


def test_workflow_backed_capabilities_are_not_allowlisted():
    # doc_mindmap / doc_slides exist as tools, but their PRODUCT paths are the toolkit
    # workflow engines (mindmap_gen / slides_gen). A same-named wrapper does not make
    # the intent a single-shot atomic call — they must stay off the fast path.
    for name in ("doc_mindmap", "doc_slides", "doc_outline"):
        assert name not in DIRECT_TOOLS


@pytest.mark.parametrize("tool", ["create_folder", "add_term", "pdf_extract_text"])
def test_allowlisted_tools_are_registered_agent_tools(tool):
    # The lossless-fallback invariant lives in the apps layer: the same capability a
    # DIRECT_TOOL dispatches must be callable by the LLM through the ToolRuntime
    # (registered by an auto-discovered apps/api/tools/*_tool.py module).
    import pathlib

    registry_dir = pathlib.Path(__file__).resolve().parents[1] / "apps" / "api" / "tools"
    sources = [p.read_text(encoding="utf-8") for p in registry_dir.rglob("*.py")]
    assert any(
        f'name="{tool}"' in src for src in sources
    ), f"{tool} is allowlisted but never registered as an agent tool"


# ── schema gate (executor's final pre-seam check) — roster is the truth ───────────

_ROSTER = {
    "add_term": {"term": {"max_len": 120, "required": True},
                 "domain": {"max_len": 60, "required": True}},
    "create_folder": {"name": {"max_len": 120, "required": True},
                      "parent_path": {"max_len": 0, "required": False}},
}


def test_validate_action_strips_and_bounds():
    out = validate_action("add_term", {"term": " x ", "domain": " y "},
                          tool_schemas=_ROSTER)
    assert out == {"tool": "add_term", "args": {"term": "x", "domain": "y"}}


def test_validate_action_optional_slot_may_be_absent_and_extras_drop():
    out = validate_action("create_folder", {"name": "季度报告"}, tool_schemas=_ROSTER)
    assert out["args"] == {"name": "季度报告"}


def test_validate_action_truth_is_the_roster_not_the_legacy_table():
    # a tool living ONLY in DIRECT_TOOLS (the legacy L0 table) is refused when
    # the runtime roster lacks it; a roster-only tool passes even though
    # DIRECT_TOOLS never heard of it (ruling — single truth).
    assert "pdf_extract_text" in DIRECT_TOOLS
    with pytest.raises(ActionSchemaError):
        validate_action("pdf_extract_text", {"asset_id": "a-1"},
                        tool_schemas=_ROSTER)
    out = validate_action(
        "list_documents", {"query": "x"},
        tool_schemas={"list_documents": {"query": {"max_len": 0, "required": True}}})
    assert out["tool"] == "list_documents"


@pytest.mark.parametrize("tool,args", [
    ("nope", {}),                                        # not on the roster
    ("create_folder", {}),                               # missing required slot
    ("create_folder", {"name": "   "}),                  # blank
    ("create_folder", {"name": 3}),                      # not a str
    ("add_term", {"term": "x" * 121, "domain": "d"}),    # over length
])
def test_validate_action_rejects(tool, args):
    with pytest.raises(ActionSchemaError):
        validate_action(tool, args, tool_schemas=_ROSTER)

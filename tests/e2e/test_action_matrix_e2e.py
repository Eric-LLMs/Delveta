"""The 11-capability × real-dispatch matrix (Phase 4-A / A-1).

Each PASS row drives the capability's canonical phrase through the REAL
``/chat/stream`` control plane and asserts, at the REAL tool-body boundary:

  * anti-fake guard — the runtime recorded EXACTLY one call, of the right tool,
    with exactly the certified args (a turn that "reports success" on a redirected
    no-op seam fails here);
  * zero-LLM — the funnel extraction and the Agent ReAct loop spent no model call
    (``ScriptedPort.single_shot == steps == judged == 0``); a tool's own
    external-LLM double is a SEPARATE object and never touches these counters;
  * permission — WRITE-classified caps surface an ASK; READ caps surface none.

:10: ``cap-social-search`` and :11: ``cap-read-file`` are BLOCKED (a prerequisite
outside A-1 scope), NOT failures — they are ``pytest.mark.skip`` so CI reports
``9 passed, 2 skipped``.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from tests.e2e.harness import (
    CAP_BY_ID,
    build_stack,
    expected_args,
    sse_for,
    turn_fields,
)

MATRIX = [
    pytest.param("cap-create-folder", id="1-create-folder"),
    pytest.param("cap-add-term", id="2-add-term"),
    pytest.param("cap-web-search", id="3-web-search"),
    pytest.param("cap-rag-search", id="4-rag-search"),
    pytest.param("cap-translate", id="5-translate"),
    pytest.param("cap-vision", id="6-vision"),
    pytest.param("cap-read-document", id="7-read-document"),
    pytest.param("cap-pdf-extract-text", id="8-pdf-extract-text"),
    pytest.param("cap-pdf-table-to-text", id="9-pdf-table-to-text"),
    pytest.param(
        "cap-social-search", id="10-social-search",
        marks=pytest.mark.skip(
            reason="BLOCKED: cap-social-search resolves through the PluginManager "
            "search_social mount, not the builtin ToolRuntime roster A-1 drives"),
    ),
    pytest.param(
        "cap-read-file", id="11-read-file",
        marks=pytest.mark.skip(
            reason="BLOCKED: cap-read-file needs a granted workspace + sandbox file "
            "scope that is outside the A-1 in-process scope"),
    ),
]


@pytest.mark.parametrize("cap_id", MATRIX)
async def test_action_matrix(monkeypatch, cap_id):
    cap = CAP_BY_ID[cap_id]
    stack = build_stack(monkeypatch)
    asset_id = uuid4() if cap.needs_asset else None

    res = await sse_for(stack, cap.message, **turn_fields(cap, asset_id=asset_id))

    # ── anti-fake guard: the REAL tool body ran exactly once, with exact args ──
    assert stack.recorder.call_count == 1, stack.recorder.calls
    assert stack.recorder.recorded_tool == cap.tool, stack.recorder.calls
    assert stack.recorder.recorded_args == expected_args(cap, asset_id), stack.recorder.calls
    assert stack.recorder.calls[0]["is_error"] is False, stack.recorder.calls

    # ── zero funnel/Agent model calls; the Agent loop was never entered ──
    assert stack.port.steps == 0
    assert stack.port.single_shot == 0
    assert stack.port.judged == 0

    # ── permission: WRITE caps ASK, READ caps do not ──
    if cap.write:
        assert res.approvals and res.approvals[0]["name"] == cap.tool
    else:
        assert res.approvals == []


# ── Page-scope contract: the SENTENCE states it; the viewer never substitutes ──────
_PAGE_RANGE_MESSAGE = "extract pages 4 to 5 from this pdf"
_PAGE_SINGLE_MESSAGE = "extract page 4 from this pdf"


async def test_sentence_page_range_reaches_the_tool(monkeypatch):
    """A range stated in the sentence reaches the tool as ``"from-to"``."""
    cap = CAP_BY_ID["cap-pdf-extract-text"]
    stack = build_stack(monkeypatch,
                        query_overrides={cap.cap_id: _PAGE_RANGE_MESSAGE})
    asset_id = uuid4()

    await sse_for(stack, _PAGE_RANGE_MESSAGE,
                  **turn_fields(cap, asset_id=asset_id))

    assert stack.recorder.call_count == 1, stack.recorder.calls
    assert stack.recorder.recorded_tool == "pdf_extract_text"
    # The certified ACTION carries the string spec; the tool body parses it into a
    # 1-based page list before handing it to the pdf lib (parse_pages_spec).
    assert stack.recorder.recorded_args == {"asset_id": str(asset_id), "pages": "4-5"}
    assert stack.pdf_seam.extract_calls == [[4, 5]]
    assert stack.port.steps == 0


async def test_sentence_single_page_reaches_the_tool(monkeypatch):
    """A single page stated in the sentence reaches the tool as ``"n"``."""
    cap = CAP_BY_ID["cap-pdf-extract-text"]
    stack = build_stack(monkeypatch,
                        query_overrides={cap.cap_id: _PAGE_SINGLE_MESSAGE})
    asset_id = uuid4()

    await sse_for(stack, _PAGE_SINGLE_MESSAGE,
                  **turn_fields(cap, asset_id=asset_id))

    assert stack.recorder.call_count == 1, stack.recorder.calls
    assert stack.recorder.recorded_args == {"asset_id": str(asset_id), "pages": "4"}
    assert stack.pdf_seam.extract_calls == [[4]]
    assert stack.port.steps == 0


async def test_the_viewer_range_does_not_narrow_a_whole_document_request(monkeypatch):
    """A viewer range is NOT a page statement: the whole document stays whole."""
    cap = CAP_BY_ID["cap-pdf-extract-text"]
    stack = build_stack(monkeypatch)
    asset_id = uuid4()

    await sse_for(stack, cap.message,
                  **turn_fields(cap, asset_id=asset_id, page_from=4, page_to=5))

    assert stack.recorder.call_count == 1, stack.recorder.calls
    assert stack.recorder.recorded_args == {"asset_id": str(asset_id)}
    assert stack.pdf_seam.extract_calls == [None]        # no pages -> whole document
    assert stack.port.steps == 0


async def test_a_half_open_viewer_range_is_not_a_page_scope(monkeypatch):
    """A half-open viewer range carries no page decision: whole document."""
    cap = CAP_BY_ID["cap-pdf-extract-text"]
    stack = build_stack(monkeypatch)
    asset_id = uuid4()

    await sse_for(stack, cap.message,
                  **turn_fields(cap, asset_id=asset_id, page_from=4))

    assert stack.recorder.call_count == 1, stack.recorder.calls
    assert stack.recorder.recorded_args == {"asset_id": str(asset_id)}
    assert stack.pdf_seam.extract_calls == [None]
    assert stack.port.steps == 0

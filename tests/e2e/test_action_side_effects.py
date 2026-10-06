"""Deterministic effect verification at the tool boundary (Phase 4-A / A-1).

WRITE caps: the REAL tool body performed EXACTLY the expected mutation (observed
through the service double behind the REAL ``DriveService`` / ``VocabularyService``
seam) and the sandbox surfaced an ASK. READ caps: the REAL tool body consulted its
outer seam exactly once and surfaced NO ASK.

None of this asserts a persisted database row — persistence is A-2 Live's job.

The final test is the ANTI-FAKE GUARD: redirecting the executor's seam to a no-op
makes the turn *report* success, yet the recorder (a real tool-body channel) shows
zero calls — proving the matrix cannot go green without a real tool body.
"""
from __future__ import annotations

from uuid import uuid4

from api.routers import chat as chat_mod

from tests.e2e.harness import CAP_BY_ID, USER, build_stack, sse_for, turn_fields


# ── WRITE: exact effect + ASK ───────────────────────────────────────────────────────
async def test_write_create_folder_exact_effect(monkeypatch):
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)

    res = await sse_for(stack, cap.message)

    assert stack.spy.folders_created == [(str(USER), "gamma")]
    assert stack.recorder.call_count == 1
    assert stack.recorder.recorded_value  # the tool body returned a confirmation
    assert res.approvals and res.approvals[0]["name"] == "create_folder"
    assert stack.port.steps == 0


async def test_write_add_term_exact_effect(monkeypatch):
    cap = CAP_BY_ID["cap-add-term"]
    stack = build_stack(monkeypatch)

    res = await sse_for(stack, cap.message)

    assert len(stack.spy.terms_added) == 1
    domain_id, word, user_id, _definition = stack.spy.terms_added[0]
    assert word == "quantum"
    assert user_id == str(USER)
    assert domain_id == "d0"  # the resolved "physics terms" domain
    assert res.approvals and res.approvals[0]["name"] == "add_term"


# ── READ: outer-seam effect counted, no ASK ─────────────────────────────────────────
async def test_read_web_search_effect(monkeypatch):
    cap = CAP_BY_ID["cap-web-search"]
    stack = build_stack(monkeypatch)

    res = await sse_for(stack, cap.message)

    assert stack.spy.web_queries == [cap.message]
    assert stack.recorder.call_count == 1
    assert stack.recorder.recorded_value[0]["title"] == "t"
    assert res.approvals == []
    assert stack.port.steps == 0


async def test_read_rag_search_effect(monkeypatch):
    cap = CAP_BY_ID["cap-rag-search"]
    stack = build_stack(monkeypatch)

    res = await sse_for(stack, cap.message)

    assert len(stack.retrieval.calls) == 1
    assert stack.retrieval.calls[0][0] == cap.message
    assert stack.recorder.recorded_value == [{"text": "learning material chunk"}]
    assert res.approvals == []
    assert stack.port.steps == 0


async def test_read_translate_effect(monkeypatch):
    cap = CAP_BY_ID["cap-translate"]
    stack = build_stack(monkeypatch)

    res = await sse_for(stack, cap.message)

    # The tool's OWN model call goes through the decoupled ToolLLM — proving the
    # "zero funnel/Agent LLM" counters and a real tool-internal LLM call coexist.
    assert len(stack.tool_llm.calls) == 1
    assert stack.tool_llm.calls[0][0] == "hello world"
    assert stack.port.steps == 0
    assert stack.port.single_shot == 0
    assert stack.port.judged == 0
    assert res.approvals == []


async def test_read_vision_effect(monkeypatch):
    cap = CAP_BY_ID["cap-vision"]
    stack = build_stack(monkeypatch)
    asset_id = uuid4()

    res = await sse_for(stack, cap.message, **turn_fields(cap, asset_id=asset_id))

    assert len(stack.vision_seam.calls) == 1
    assert "describe this image" in stack.vision_seam.calls[0]["prompt"]
    assert stack.recorder.recorded_value == "[vision analysis]"
    assert res.approvals == []


async def test_read_document_effect(monkeypatch):
    cap = CAP_BY_ID["cap-read-document"]
    stack = build_stack(monkeypatch)
    asset_id = uuid4()

    res = await sse_for(stack, cap.message, **turn_fields(cap, asset_id=asset_id))

    assert stack.read_doc_seam.calls == ["report.pdf"]
    assert stack.recorder.recorded_value == "[document text]"
    assert res.approvals == []


async def test_read_pdf_extract_effect(monkeypatch):
    cap = CAP_BY_ID["cap-pdf-extract-text"]
    stack = build_stack(monkeypatch)
    asset_id = uuid4()

    res = await sse_for(stack, cap.message, **turn_fields(cap, asset_id=asset_id))

    assert stack.pdf_seam.extract_calls == [None]  # whole document
    assert stack.recorder.recorded_value == "[pdf text]"
    assert res.approvals == []


async def test_read_pdf_table_effect(monkeypatch):
    cap = CAP_BY_ID["cap-pdf-table-to-text"]
    stack = build_stack(monkeypatch)
    asset_id = uuid4()

    res = await sse_for(stack, cap.message, **turn_fields(cap, asset_id=asset_id))

    assert stack.pdf_seam.table_calls == [None]
    assert len(stack.pdf_seam.transcribe_calls) == 1
    assert stack.recorder.recorded_value == "[table text]"
    assert res.approvals == []


# ── ANTI-FAKE GUARD (E-1): a redirected seam must NOT pass as a real effect ──────────
async def test_anti_fake_guard_detects_redirected_seam(monkeypatch):
    """Redirect the executor's seam to a no-op: the turn reports success, the
    recorder stays empty — the assertion surface the matrix relies on is real."""
    cap = CAP_BY_ID["cap-create-folder"]
    stack = build_stack(monkeypatch)

    async def _fake_run_tool(tool, args, ctx):
        return {"ok": True, "output": "faked"}

    monkeypatch.setattr(chat_mod, "_run_tool", _fake_run_tool)

    res = await sse_for(stack, cap.message)

    assert res.answer == "faked"           # the turn LOOKS like a success ...
    assert stack.recorder.call_count == 0  # ... but no real tool body ran
    assert stack.spy.folders_created == []

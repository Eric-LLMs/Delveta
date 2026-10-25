"""Content-capability E2E — ``cap-translate`` / ``cap-summary`` real-turn material.

These two capabilities do NOT run the shared Qwen extractor. The funnel certifies
a deterministic ACTION (translate: the verbatim sentence; summary: the ``auto``
marker + the Binder's ``asset_id``), and the TOOL then assembles its own turn
material and makes its own decision call(s). This suite drives the REAL
``/chat/stream`` control plane into the REAL tool bodies and asserts, per
scenario, on the material each model call actually saw:

  * translate — exactly ONE decision call over the sentence / previous chat /
    the open document page / the captured video frame (never more);
  * summary — the FIRST call sees only query + previous message + the open
    document's identity (never its body); a second call runs ONLY when the
    decision needs material it does not have, and then sends ONLY that material;
  * save intent — decided in the first call, carried through unchanged; a file
    is written only when the user asked to save, and only at a path the user
    literally wrote;
  * zero Qwen — the slot extractor seam records nothing for either capability.

The only fakes are the outer-world seams ``build_stack`` already owns (Registry
read, LLM port, drive/storage, PDF bytes). Everything from the HTTP route down
to the tool body is production code.
"""
from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from api.tools import document_content as document_content_mod
from core.infrastructure.request_context import set_request_user

from tests.e2e.harness import Cap, ToolLLM, build_stack, sse_for
from tests.e2e.real_chain import USER, TurnResult

# ── canned replies (the tool-internal model, never the funnel/Agent port) ──────────
TRANSLATE_REPLY = {
    "source": "document",
    "target_language": "Chinese",
    "translation": "你好世界",
}

SUMMARY_GENERATION_REPLY = {
    "title": "Summary",
    "executive_summary": "A short summary.",
    "key_points": [],
    "sections": [],
    "qa": [],
}


def _first(**overrides) -> dict:
    """One ``run_auto_summary`` first-call reply."""
    base = {
        "target": "none", "ready": False, "summary": "", "asset_id": "",
        "pages": "", "save_to_file": False, "save_path": "", "reason": "x",
    }
    base.update(overrides)
    return base


def _summary_cap(message: str) -> Cap:
    return Cap("cap-summary", "summary_gen", message, write=True)


def _tail(*rows: tuple[str, str]) -> dict:
    """A v2 client body: no SQL, the client's own tail is the history."""
    return {
        "context_state": {"summary": None, "has_pending_mutations": False},
        "tail": [{"role": role, "content": content} for role, content in rows],
    }


def _viewer_doc(asset_id: str, *, page: int = 3) -> dict:
    return {"kind": "pdf", "name": "report.pdf", "asset_id": asset_id,
            "follow": True, "page": page}


def _viewer_video(asset_id: str) -> dict:
    return {
        "kind": "video", "name": "lecture.mp4", "asset_id": asset_id,
        "follow": True, "t_ms": 12_500, "duration_ms": 600_000,
        "cues": [
            {"start_ms": 8_000, "end_ms": 11_000, "text": "Earlier line."},
            {"start_ms": 11_000, "end_ms": 14_000, "text": "The line at the playhead."},
            {"start_ms": 14_000, "end_ms": 17_000, "text": "A later line."},
        ],
    }


def _pdf_pages(monkeypatch, text: str):
    """Local page extraction seam (real bytes never reach PyMuPDF here)."""
    seen: list[tuple] = []

    def _extract(data, name, pages):
        seen.append(tuple(pages))
        return text

    monkeypatch.setattr(document_content_mod, "_extract_pdf_pages", _extract)
    return seen


# ── deferred-capture driver (the real pause/resume channel) ───────────────────────
class _QueueStream(httpx.AsyncByteStream):
    """SSE body fed frame-by-frame from the app's own ``send``."""

    def __init__(self, queue: asyncio.Queue, task: asyncio.Task) -> None:
        self._queue = queue
        self._task = task

    async def __aiter__(self):
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # A client that stops reading has disconnected: tear the app task down
            # instead of leaking it, exactly like a dropped SSE connection.
            if not self._task.done():
                self._task.cancel()


class _StreamingASGITransport(httpx.AsyncBaseTransport):
    """ASGI transport that DELIVERS frames while the app is still running.

    ``httpx.ASGITransport`` awaits the whole app call and hands back one joined
    body, so a turn that PAUSES for the client (``capture-request`` ->
    ``POST /chat/captures/{ticket}`` -> resume) can never be driven through it:
    the frame arrives only after the pause already timed out. This transport
    wires the app's ``send`` to a queue so each SSE chunk is readable as it is
    produced. It is a transport seam only — app, routes, orchestrator and
    handlers are the production ones.
    """

    def __init__(self, app) -> None:
        self._app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        started = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()
        head = {"status": 500, "headers": []}
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": request.method,
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "root_path": "",
            "headers": [(k.lower(), v) for k, v in request.headers.raw],
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 47111),
        }
        received = False
        never = asyncio.Event()

        async def receive():
            nonlocal received
            if received:
                await never.wait()  # this driver never disconnects mid-turn
            received = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                head["status"] = message["status"]
                head["headers"] = [
                    (k.decode("latin-1"), v.decode("latin-1"))
                    for k, v in message.get("headers", [])
                ]
                started.set()
            elif message["type"] == "http.response.body":
                await queue.put(message.get("body", b""))
                if not message.get("more_body", False):
                    await queue.put(None)

        async def run():
            try:
                await self._app(scope, receive, send)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 — surface instead of hanging
                await queue.put(exc)
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        await started.wait()
        return httpx.Response(
            head["status"], headers=head["headers"],
            stream=_QueueStream(queue, task),
        )


async def _stream_with_capture(stack, message: str, *, captures: list[dict],
                              user=USER, **fields) -> TurnResult:
    """Drive /chat/stream through its ``capture-request`` pause.

    ``captures`` is consumed one submission per request frame, exactly like the PC
    client would answer. Everything else (frame parsing, ticket routing) matches
    ``real_chain.sse``.
    """
    set_request_user(user)
    body = {"message": message, "session_id": str(uuid4()), **fields}
    events, requests, done = [], [], None
    pending = list(captures)
    transport = _StreamingASGITransport(stack.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                 timeout=30) as client:
        async with client.stream("POST", "/chat/stream", json=body) as r:
            assert r.status_code == 200
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                evt = json.loads(line[5:].strip())
                events.append(evt)
                if evt.get("type") == "capture-request":
                    requests.append(evt["data"])
                    payload = pending.pop(0) if pending else {
                        "status": "failed", "reason": "no capture scripted"}
                    res = await client.post(
                        f"/chat/captures/{evt['data']['ticket_id']}", json=payload)
                    assert res.status_code == 200, res.text
                elif evt.get("type") == "done":
                    done = evt.get("data")
                    break
    return TurnResult(
        events=events, types=[e.get("type") for e in events], done=done,
        approvals=[e["data"] for e in events if e.get("type") == "approval-request"],
        answer=(done or {}).get("answer"), wall_ms=0.0,
    )


# ══ cap-translate ═════════════════════════════════════════════════════════════════
async def test_translate_sentence_is_one_call_with_only_the_sentence(monkeypatch):
    message = 'translate "hello world" into Chinese'
    stack = build_stack(monkeypatch, query_overrides={"cap-translate": message})

    res = await sse_for(stack, message)

    assert len(stack.tool_llm.translate_calls) == 1
    call = stack.tool_llm.translate_calls[0]
    assert message in call["prompt"]
    assert "<previous_chat>" not in call["prompt"]
    assert "<document" not in call["prompt"]
    assert "<video_subtitles>" not in call["prompt"]
    assert call["images"] == []
    # zero Qwen / zero funnel+Agent LLM; one real tool body call
    assert stack.slot_seam.calls == []
    assert (stack.port.single_shot, stack.port.steps, stack.port.judged) == (0, 0, 0)
    assert stack.recorder.call_count == 1
    assert stack.recorder.recorded_tool == "translate"
    assert stack.recorder.recorded_args == {"text": message}
    assert res.answer == "Translation (Chinese):\n你好世界"


async def test_translate_document_page_sends_the_page_and_previous_two_rows(monkeypatch):
    message = "translate the paragraph on this page into Chinese"
    asset_id = str(uuid4())
    stack = build_stack(monkeypatch, query_overrides={"cap-translate": message})
    pages_seen = _pdf_pages(monkeypatch, "[[Page 3]]\nThe quick brown fox jumps.")
    body = _tail(
        ("user", "First question"),
        ("assistant", "First answer"),
        ("user", "Third question"),
        ("assistant", "Third answer"),
    )

    await sse_for(stack, message, viewer=_viewer_doc(asset_id),
                  **body)

    assert len(stack.tool_llm.translate_calls) == 1
    call = stack.tool_llm.translate_calls[0]
    prompt = call["prompt"]
    # the document identity + the LOCALLY extracted page text
    assert f'<document name="report.pdf" asset_id="{asset_id}" (page 3)>' in prompt
    assert "The quick brown fox jumps." in prompt
    assert pages_seen == [(3,)]
    # exactly the previous TWO chat rows ride along
    assert "<previous_chat>" in prompt
    assert "Third question" in prompt and "Third answer" in prompt
    assert "First question" not in prompt and "First answer" not in prompt
    assert call["images"] == []
    assert stack.slot_seam.calls == []
    assert stack.recorder.recorded_args == {"text": message, "asset_id": asset_id}


async def test_translate_previous_chat_target_reads_the_last_two_rows(monkeypatch):
    message = "translate my previous message into English"
    stack = build_stack(monkeypatch, query_overrides={"cap-translate": message})

    await sse_for(stack, message, **_tail(
        ("user", "First question"), ("assistant", "First answer"),
        ("user", "Third question"), ("assistant", "Third answer")))

    assert len(stack.tool_llm.translate_calls) == 1
    prompt = stack.tool_llm.translate_calls[0]["prompt"]
    assert "<previous_chat>" in prompt
    assert "Third question" in prompt and "Third answer" in prompt
    assert "First question" not in prompt
    assert "<document" not in prompt and "<video_subtitles>" not in prompt


async def test_translate_video_pauses_for_the_frame_then_makes_one_call(monkeypatch):
    message = "翻译这段视频里说的话"
    video_id, frame_id = str(uuid4()), str(uuid4())
    stack = build_stack(monkeypatch, query_overrides={"cap-translate": message})

    res = await _stream_with_capture(
        stack, message,
        viewer=_viewer_video(video_id),
        captures=[{"status": "ok", "asset_id": frame_id, "name": "lecture.mp4",
                   "t_ms": 12_500,
                   "text": "client-prepared captions around 00:12"}],
        **_tail(("user", "First question"), ("assistant", "First answer"),
                ("user", "Second question"), ("assistant", "Second answer")),
    )

    # the funnel paused exactly once and asked for a FRAME (never an asset id)
    assert stack.slot_seam.calls == []          # no Qwen anywhere on the lane
    assert len([t for t in res.types if t == "capture-request"]) == 1
    # the paused plan was resumed, not re-funnelled: ONE decision call
    assert len(stack.tool_llm.translate_calls) == 1
    call = stack.tool_llm.translate_calls[0]
    prompt = call["prompt"]
    assert "<video_subtitles>" in prompt
    assert "The line at the playhead." in prompt           # the client's cue window
    assert "client-prepared captions around 00:12" in prompt  # the capture text
    assert "<video_frame>" in prompt
    assert "<document" not in prompt
    assert "Second question" in prompt and "First question" not in prompt
    assert len(call["images"]) == 1
    assert call["images"][0].startswith("data:image/png;base64,")
    # the only asset id the tool saw is the CLIENT's uploaded frame
    assert stack.recorder.recorded_args == {"text": message, "asset_id": frame_id}
    assert len(res.approvals) == 0


# ══ cap-summary ══════════════════════════════════════════════════════════════════
async def test_summary_ready_first_call_answers_without_a_second_call(monkeypatch, tmp_path):
    message = "summarize what we just discussed"
    tool_llm = ToolLLM(summary_first=_first(
        target="previous_message", ready=True,
        summary="They agreed on the release date."))
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    res = await sse_for(stack, message, **_tail(
        ("user", "When do we ship?"), ("assistant", "We ship on Friday.")))

    assert len(tool_llm.once_calls) == 1
    assert tool_llm.generation_calls == []                 # ONE call, no second
    first = tool_llm.once_calls[0]["prompt"]
    assert f"<user_request>\n{message}\n</user_request>" in first
    assert '<previous_message role="assistant">' in first
    assert "We ship on Friday." in first
    assert "<open_document/>" in first
    assert "<chat_rows>2</chat_rows>" in first
    assert "They agreed on the release date." in res.answer
    # chat window only: nothing was written
    assert list(tmp_path.rglob("*.md")) == []
    assert stack.slot_seam.calls == []
    assert (stack.port.single_shot, stack.port.steps, stack.port.judged) == (0, 0, 0)
    assert stack.recorder.recorded_args == {"auto": "1"}
    # summary_gen is pre-approved on the chat surface: no Allow/Deny frame at all
    assert res.approvals == []


async def test_summary_ready_with_explicit_save_writes_a_real_file(monkeypatch, tmp_path):
    message = "summarize my previous message and save it to reports/q3.md"
    tool_llm = ToolLLM(summary_first=_first(
        target="previous_message", ready=True, summary="The release plan, condensed.",
        save_to_file=True, save_path="reports/q3.md"))
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    res = await sse_for(stack, message, **_tail(
        ("user", "Ship it?"), ("assistant", "Yes, on Friday.")))

    assert len(tool_llm.once_calls) == 1
    assert tool_llm.generation_calls == []
    written = list((tmp_path / "reports").glob("q3_*.md"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8") == "The release plan, condensed."
    # a saved turn answers with the file it wrote (the body lives in the file,
    # the same contract the toolkit's explicit output_dir lane has)
    assert written[0].name in res.answer
    assert str(written[0]) in res.answer
    assert "The release plan, condensed." not in res.answer
    # the save cue path is a DESTINATION, not an input: the handler stayed AUTO
    assert stack.recorder.recorded_args == {"auto": "1"}
    assert stack.slot_seam.calls == []


async def test_summary_document_page_range_second_call_sends_only_the_document(
        monkeypatch, tmp_path):
    message = "summarize pages 4 to 5 of this document"
    asset_id = str(uuid4())
    tool_llm = ToolLLM(summary_first=_first(
        target="document", asset_id=asset_id, pages="4-5"))
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))
    pages_seen = _pdf_pages(monkeypatch, "[[Page 4]]\nAlpha\n\n[[Page 5]]\nBeta")

    await sse_for(stack, message, viewer=_viewer_doc(asset_id, page=3),
                  **_tail(("user", "First question"), ("assistant", "First answer")))

    assert len(tool_llm.once_calls) == 1
    first = tool_llm.once_calls[0]["prompt"]
    # the first call sees the document's IDENTITY only, never its body
    assert f'<open_document asset_id="{asset_id}" name="report.pdf" kind="pdf" page="3"/>' in first
    assert "Alpha" not in first and "Beta" not in first
    # exactly ONE second call, on the requested pages only
    assert len(tool_llm.generation_calls) == 1
    generated = tool_llm.generation_calls[0]["prompt"]
    assert pages_seen == [(4, 5)]
    assert "Current summary request:" in generated
    assert "Alpha" in generated and "Beta" in generated
    assert "First question" not in generated and "First answer" not in generated
    assert "The line at the playhead" not in generated
    assert list(tmp_path.rglob("*.md")) == []      # no save intent -> chat only
    assert stack.slot_seam.calls == []


async def test_summary_session_second_call_sends_only_the_transcript(
        monkeypatch, tmp_path):
    message = "summarize our whole conversation"
    tool_llm = ToolLLM(summary_first=_first(target="session"))
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    await sse_for(stack, message, **_tail(
        ("user", "First question"), ("assistant", "First answer"),
        ("user", "Second question"), ("assistant", "Second answer")))

    assert len(tool_llm.once_calls) == 1
    assert len(tool_llm.generation_calls) == 1
    generated = tool_llm.generation_calls[0]["prompt"]
    assert "**user:** First question" in generated
    assert "**assistant:** Second answer" in generated
    assert "<document" not in generated
    assert list(tmp_path.rglob("*.md")) == []
    assert stack.slot_seam.calls == []


async def test_summary_previous_message_second_call_sends_that_message_only(
        monkeypatch, tmp_path):
    message = "summarize the last message"
    tool_llm = ToolLLM(summary_first=_first(target="previous_message"))
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    await sse_for(stack, message, **_tail(
        ("user", "Older question"), ("assistant", "The message to summarize.")))

    assert len(tool_llm.generation_calls) == 1
    generated = tool_llm.generation_calls[0]["prompt"]
    assert "The message to summarize." in generated
    assert "Older question" not in generated
    assert stack.slot_seam.calls == []


async def test_summary_first_call_failure_falls_back_to_the_previous_message(
        monkeypatch, tmp_path):
    message = "summarize this"
    tool_llm = ToolLLM(once_raises=True)
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    res = await sse_for(stack, message, **_tail(
        ("user", "Question"), ("assistant", "The previous answer.")))

    assert len(tool_llm.once_calls) == 1          # the failed decision call
    assert len(tool_llm.generation_calls) == 1    # the ladder reasoned locally
    generated = tool_llm.generation_calls[0]["prompt"]
    assert "The previous answer." in generated
    assert "Question" not in generated
    assert res.answer                       # a real summary, not an escalation
    assert stack.port.steps == 0
    assert stack.slot_seam.calls == []


async def test_summary_with_nothing_to_summarize_fails_honestly(
        monkeypatch, tmp_path):
    message = "summarize this"
    tool_llm = ToolLLM(once_raises=True)
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    res = await sse_for(stack, message)     # no chat rows, no document open

    # the tool body failed honestly (no invented summary, no file, no second call)
    assert stack.recorder.call_count == 1
    assert stack.recorder.calls[0]["is_error"] is True
    assert "nothing to summarize" in stack.recorder.calls[0]["error"]
    assert tool_llm.generation_calls == []
    assert list(tmp_path.rglob("*.md")) == []
    assert stack.slot_seam.calls == []
    # the failure escalates to the Agent lane (the established preflight path)
    assert stack.port.steps == 1
    assert "nothing to summarize" not in (res.answer or "")


async def test_summary_invented_save_path_is_dropped(monkeypatch, tmp_path):
    message = "summarize my previous message and save it"
    tool_llm = ToolLLM(summary_first=_first(
        target="previous_message", ready=True, summary="Condensed.",
        save_to_file=True, save_path="/tmp/evil.md"))
    stack = build_stack(monkeypatch, workspace=tmp_path, tool_llm=tool_llm,
                        extra_caps=(_summary_cap(message),))

    res = await sse_for(stack, message, **_tail(
        ("user", "Ship it?"), ("assistant", "Yes.")))

    # the invented path was never used; the explicit save landed in the default
    assert not (tmp_path / "tmp").exists()
    assert not (tmp_path / "evil.md").exists()
    written = list((tmp_path / ".output" / "summary").glob("summary_*.md"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8") == "Condensed."
    assert "/tmp/evil.md" not in (res.answer or "")


@pytest.mark.parametrize("message", ["summarize this", "translate this sentence"])
async def test_the_content_capabilities_never_reach_the_qwen_seam(monkeypatch, message):
    """Belt and braces: on a plain turn both capabilities stay Qwen-free."""
    is_summary = message == "summarize this"
    tool_llm = ToolLLM(summary_first=_first(target="none", reason="nothing"))
    stack = build_stack(
        monkeypatch, tool_llm=tool_llm,
        query_overrides={} if is_summary else {"cap-translate": message},
        extra_caps=(_summary_cap(message),) if is_summary else ())

    await sse_for(stack, message)

    assert stack.slot_seam.calls == []

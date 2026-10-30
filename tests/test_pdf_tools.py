"""``pdf_table_to_text``: the table image carries the user's OWN request.

Local data preparation (PyMuPDF table detection + compositing) is unchanged. The one
vision call now receives the turn's sentence — so "解释一下这个表格" is answered with an
explanation and "只提取表格中的数据" with the exact table — instead of a fixed
transcription-only order. On a live chat turn the shared ``describe_image`` entry streams
every delta to the content sink and returns exactly that text (no second synthesis pass);
a headless caller keeps the transcription order and the non-streaming call.
"""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from agent import Context
from agent.engine.context import AgentTurn
from agent.engine.decisions import ToolExecution
from agent.engine.runtime import ToolRuntime
from core.infrastructure import pdf as pdf_lib
from core.infrastructure import vision_caption as vc
from core.infrastructure.request_context import (
    request_user,
    reset_request_content_sink,
    set_request_content_sink,
)
from core.infrastructure.storage import LocalStorage, object_key

from api.tools import pdf_tools
from tests._drive_fakes import make_drive

_STRIP = b"\x89PNG\r\n\x1a\nTABLE-STRIP"


@asynccontextmanager
async def _turn(user_msg: str):
    """Bind an AgentTurn the way ``_run_tool`` does for the ACTION lane."""
    from agent.engine.context import _TURN_CTX

    token = _TURN_CTX.set(AgentTurn(user_msg=user_msg))
    try:
        yield
    finally:
        _TURN_CTX.reset(token)


@pytest.fixture
def sink():
    frames: list[str] = []
    token = set_request_content_sink(frames.append)
    try:
        yield frames
    finally:
        reset_request_content_sink(token)


async def _stack(monkeypatch, tmp_path, llm=None):
    """A runtime whose drive asset resolves to stored bytes and whose table render is local."""
    storage = LocalStorage(tmp_path)
    data = b"%PDF-1.7 fake"
    sha = hashlib.sha256(data).hexdigest()
    await storage.put(object_key(sha), data)
    drive = make_drive(tmp_path)
    uid = uuid4()
    asset = await drive.assets.create(uid, "audit.pdf", object_sha256=sha, scope="drive")
    request_user.set(uid)

    monkeypatch.setattr(pdf_tools, "DriveService", lambda sf: drive)
    monkeypatch.setattr(pdf_lib, "render_tables_png",
                        lambda raw, pages=None: _STRIP)

    ctx = Context()
    ctx.provide("storage", storage)
    ctx.provide("session_factory", object)
    runtime = ToolRuntime()
    pdf_tools.register(runtime, ctx, llm=llm)
    return runtime, str(asset.id)


async def _call(runtime, asset_id, **args):
    return await runtime.execute(
        ToolExecution("c1", "pdf_table_to_text", {"asset_id": asset_id, **args}))


def _capture_describe(monkeypatch, answer="ANSWER"):
    calls: list[dict] = []

    async def _describe(data, mime="", *, llm, session_factory, prompt=None, user_id=None,
                        role_id=None):
        calls.append({"data": data, "mime": mime, "prompt": prompt})
        return answer

    monkeypatch.setattr(pdf_tools, "describe_image", _describe)
    return calls


async def test_explanation_request_reaches_the_model_verbatim(monkeypatch, tmp_path):
    runtime, asset_id = await _stack(monkeypatch, tmp_path)
    calls = _capture_describe(monkeypatch, answer="这张表比较了四条主张的证据强度…")

    async with _turn("解释一下这个表格"):
        res = await _call(runtime, asset_id, pages="3")

    assert res.is_error is False
    assert str(res.value) == "这张表比较了四条主张的证据强度…"   # model's answer IS the reply
    assert len(calls) == 1                                       # exactly ONE vision call
    assert calls[0]["data"] == _STRIP and calls[0]["mime"] == "image/png"
    prompt = calls[0]["prompt"]
    assert "解释一下这个表格" in prompt                          # the user's own sentence
    assert "output only the transcribed text" not in prompt      # never pinned to transcription


async def test_extraction_request_also_carries_the_sentence(monkeypatch, tmp_path):
    runtime, asset_id = await _stack(monkeypatch, tmp_path)
    calls = _capture_describe(monkeypatch, answer="| k1 | … |")

    async with _turn("只提取表格中的所有数据"):
        res = await _call(runtime, asset_id)

    assert res.is_error is False and str(res.value) == "| k1 | … |"
    assert "只提取表格中的所有数据" in calls[0]["prompt"]
    assert "逐字保留" in calls[0]["prompt"]                      # exactness stated, not enforced as "transcribe only"


async def test_no_turn_keeps_the_transcription_order(monkeypatch, tmp_path):
    runtime, asset_id = await _stack(monkeypatch, tmp_path)
    calls = _capture_describe(monkeypatch, answer="table text")

    res = await _call(runtime, asset_id)                          # no AgentTurn bound (headless)

    assert str(res.value) == "table text"
    assert calls[0]["prompt"] == pdf_tools._TRANSCRIBE_PROMPT


async def test_live_turn_streams_every_delta_and_returns_the_same_text(
        monkeypatch, tmp_path, sink):
    async def _chain(session_factory, *, user_id=None, role_id=None):
        return [("https://v.example/v1", "key", "vision-model")]

    monkeypatch.setattr(vc, "resolve_vision_channels", _chain)

    class _StreamLLM:
        def __init__(self):
            self.stream_calls = 0
            self.chat_calls = 0

        async def chat_stream(self, messages, *, model=None, base_url=None, api_key=None):
            self.stream_calls += 1
            for delta in ("这张表", "有四条主张。"):
                yield {"type": "thinking", "data": "hidden"}
                yield {"type": "content", "data": delta}

        async def chat(self, messages, *, model=None, base_url=None, api_key=None):
            self.chat_calls += 1
            return {"content": "one shot", "tool_calls": [], "usage": {}}

    llm = _StreamLLM()
    runtime, asset_id = await _stack(monkeypatch, tmp_path, llm=llm)

    async with _turn("解释一下这个表格"):
        res = await _call(runtime, asset_id)

    assert str(res.value) == "这张表有四条主张。"
    assert sink == ["这张表", "有四条主张。"]                     # deltas, in order
    assert str(res.value) == "".join(sink)                       # tool result == streamed text
    assert llm.stream_calls == 1 and llm.chat_calls == 0         # streamed, never one-shot

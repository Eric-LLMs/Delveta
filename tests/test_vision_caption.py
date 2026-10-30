"""Vision plumbing: the streaming lane of :func:`describe_image`.

A live chat turn installs a content sink (the ACTION executor does it around the
tool call); the shared vision seam then streams each delta to the user and returns
the SAME accumulated text as the tool result. Headless callers (ingest captioning,
non-streaming /chat) have no sink and keep the one-shot ``llm.chat`` contract.

The fail-over rule is the non-streaming one (a channel that errors or answers with
no text moves to the next candidate) with a single exception: text that already
reached the user is final, so a later fault keeps that prefix and never replays a
second, different answer.
"""
from __future__ import annotations

import pytest
from core.infrastructure import vision_caption as vc

_CHAIN = [("https://a.example/v1", "key-a", "model-a"),
          ("https://b.example/v1", "key-b", "model-b")]


@pytest.fixture
def content_sink():
    from core.infrastructure.request_context import (
        reset_request_content_sink,
        set_request_content_sink,
    )

    frames: list[str] = []
    token = set_request_content_sink(frames.append)
    try:
        yield frames
    finally:
        reset_request_content_sink(token)


def _patch_chain(monkeypatch):
    async def _chain(session_factory, *, user_id=None, role_id=None):
        return list(_CHAIN)

    monkeypatch.setattr(vc, "resolve_vision_channels", _chain)


class _StreamLLM:
    """Replays scripted channels; each channel is a list of deltas or an exception."""

    def __init__(self, channels):
        self.channels = channels
        self.stream_calls: list[str] = []
        self.chat_calls: list[str] = []

    async def chat_stream(self, messages, *, model=None, base_url=None, api_key=None):
        self.stream_calls.append(model)
        script = self.channels[len(self.stream_calls) - 1]
        if isinstance(script, Exception):
            raise script
        for delta in script:
            yield {"type": "thinking", "data": "hidden"}
            yield {"type": "content", "data": delta}
        yield {"type": "tool_calls", "data": []}

    async def chat(self, messages, *, model=None, base_url=None, api_key=None):
        self.chat_calls.append(model)
        return {"content": "one shot", "tool_calls": [], "usage": {}}


class _ChatOnlyLLM:
    """No streaming surface at all: the ingest/worker shape."""

    def __init__(self):
        self.chat_calls = 0

    async def chat(self, messages, *, model=None, base_url=None, api_key=None):
        self.chat_calls += 1
        return {"content": "one shot", "tool_calls": [], "usage": {}}


async def test_live_turn_streams_every_delta_and_returns_the_same_text(
        monkeypatch, content_sink):
    _patch_chain(monkeypatch)
    llm = _StreamLLM([["这张", "图是", "流程图。"]])
    text = await vc.describe_image(b"\x89PNG\r\n", "image/png", llm=llm,
                                   session_factory=None)
    assert content_sink == ["这张", "图是", "流程图。"]   # deltas, in order
    assert text == "".join(content_sink)               # tool result == streamed text
    assert llm.chat_calls == []                        # never the one-shot call


async def test_headless_call_stays_non_streaming(monkeypatch):
    _patch_chain(monkeypatch)
    llm = _ChatOnlyLLM()                               # no sink either (worker/ingest)
    text = await vc.describe_image(b"\x89PNG\r\n", "image/png", llm=llm,
                                   session_factory=None)
    assert text == "one shot"
    assert llm.chat_calls == 1


async def test_failure_before_the_first_delta_tries_the_next_channel(
        monkeypatch, content_sink):
    _patch_chain(monkeypatch)
    llm = _StreamLLM([RuntimeError("upstream 500"), ["第二", "个通道"]])
    text = await vc.describe_image(b"\x89PNG\r\n", "image/png", llm=llm,
                                   session_factory=None)
    assert text == "第二个通道"
    assert content_sink == ["第二", "个通道"]
    assert llm.stream_calls == ["model-a", "model-b"]


async def test_empty_channel_falls_through_to_the_next_one(
        monkeypatch, content_sink):
    _patch_chain(monkeypatch)
    llm = _StreamLLM([[], ["兜底", "答案"]])
    text = await vc.describe_image(b"\x89PNG\r\n", "image/png", llm=llm,
                                   session_factory=None)
    assert text == "兜底答案"
    assert llm.stream_calls == ["model-a", "model-b"]


async def test_failure_after_a_delta_keeps_the_prefix_and_never_fails_over(
        monkeypatch, content_sink):
    class _HalfThenBoom(_StreamLLM):
        async def chat_stream(self, messages, *, model=None, base_url=None, api_key=None):
            self.stream_calls.append(model)
            if len(self.stream_calls) > 1:  # a fail-over would replay the answer
                yield {"type": "content", "data": "不该出现"}
                return
            yield {"type": "content", "data": "已经显示"}
            raise RuntimeError("stream died mid-answer")

    _patch_chain(monkeypatch)
    llm = _HalfThenBoom([[]])
    text = await vc.describe_image(b"\x89PNG\r\n", "image/png", llm=llm,
                                   session_factory=None)
    assert text == "已经显示"
    assert content_sink == ["已经显示"]                  # nothing replayed
    assert llm.stream_calls == ["model-a"]


async def test_all_channels_silent_raises_vision_unsupported(
        monkeypatch, content_sink):
    _patch_chain(monkeypatch)
    llm = _StreamLLM([[], RuntimeError("boom")])
    with pytest.raises(vc.VisionUnsupported):
        await vc.describe_image(b"\x89PNG\r\n", "image/png", llm=llm,
                                session_factory=None)
    assert content_sink == []

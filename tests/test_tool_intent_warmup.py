"""Unit surface for the tool-intent warm-up (O2) in ``apps/api/main.py``.

The sidecar's health check proves the model is registered, not loaded. The
lifespan therefore starts ONE background request with the existing endpoint and
model tag, using a loading-specific timeout, and never blocks or fails startup.

No network: ``httpx.MockTransport`` stands in for the sidecar.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from apps.api import main as api_main
from core.config import settings


def _patch_transport(monkeypatch, handler, seen: list | None = None):
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(api_main.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_warmup_skips_when_no_endpoint_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "")
    calls: list = []
    _patch_transport(monkeypatch, lambda request: calls.append(request))

    await api_main._warm_tool_intent_model()

    assert calls == []


@pytest.mark.asyncio
async def test_warmup_posts_the_configured_model(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "http://warm.test/v1")
    monkeypatch.setattr(settings, "chat_tool_intent_local_model", "pe-test:tag")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    _patch_transport(monkeypatch, handler)

    await api_main._warm_tool_intent_model()

    assert len(seen) == 1
    assert seen[0].url.path.endswith("/chat/completions")
    assert b'"model":"pe-test:tag"' in seen[0].content


@pytest.mark.asyncio
async def test_warmup_failure_is_logged_not_raised(monkeypatch, caplog):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "http://warm.test/v1")
    monkeypatch.setattr(settings, "chat_tool_intent_local_model", "pe-test:tag")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="not ready")

    _patch_transport(monkeypatch, handler)

    await api_main._warm_tool_intent_model()  # must not raise

    assert any("warmup failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_warmup_is_scheduled_at_most_once(monkeypatch):
    started: list[int] = []
    gate = asyncio.Event()

    async def fake_warm() -> None:
        started.append(1)
        await gate.wait()

    monkeypatch.setattr(api_main, "_warmup_task", None)
    monkeypatch.setattr(api_main, "_warm_tool_intent_model", fake_warm)

    api_main._schedule_tool_intent_warmup()
    api_main._schedule_tool_intent_warmup()
    task = api_main._warmup_task
    await asyncio.sleep(0)

    assert started == [1]
    gate.set()
    await task

"""Startup warm-up of the tool-intent extractor (the O2 surface, now owned by
``apps/api/local_model_readiness.py``).

The sidecar's health check proves the model is registered, not loaded, so the
API warms it with a REAL native ``/api/generate`` call that also asks for
permanent residency. Failures are logged into the readiness record and never
abort startup.

No network: ``httpx.MockTransport`` stands behind the manager's ``_client``.
"""
from __future__ import annotations

import json
import logging

import httpx
import pytest
from api import local_model_readiness as lmr
from core.config import settings


def _clear_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_base_url", "")
    monkeypatch.setattr(settings, "chat_cap_router_url", "")
    monkeypatch.setattr(settings, "tts_base_url", "")
    monkeypatch.setattr(settings, "stt_base_url", "")


def _patch_client(monkeypatch, handler) -> None:
    real = httpx.AsyncClient

    def factory(timeout):
        return real(timeout=timeout, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(lmr, "_client", factory)


@pytest.mark.asyncio
async def test_warmup_skips_when_no_endpoint_is_configured(monkeypatch):
    _clear_endpoints(monkeypatch)
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "")
    calls: list = []
    _patch_client(monkeypatch, lambda request: calls.append(request))

    await lmr.LocalModelReadiness().warm_all()

    assert calls == []


@pytest.mark.asyncio
async def test_warmup_posts_the_configured_model_for_residency(monkeypatch):
    _clear_endpoints(monkeypatch)
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "http://warm.test/v1")
    monkeypatch.setattr(settings, "chat_tool_intent_local_model", "pe-test:tag")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{
                "name": "pe-test:tag", "size": 1, "size_vram": 0,
                "expires_at": "never"}]})
        return httpx.Response(200, json={"response": "", "done": True})

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    generate = [r for r in seen if r.url.path == "/api/generate"]
    assert len(generate) == 1
    assert generate[0].url.host == "warm.test"
    body = json.loads(generate[0].content)
    assert body["model"] == "pe-test:tag"
    assert body["keep_alive"] == -1
    assert manager.snapshot()["models"]["tool_intent"]["state"] == lmr.READY


@pytest.mark.asyncio
async def test_warmup_failure_is_logged_not_raised(monkeypatch, caplog):
    _clear_endpoints(monkeypatch)
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "http://warm.test/v1")
    monkeypatch.setattr(settings, "chat_tool_intent_local_model", "pe-test:tag")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(500, text="not ready")

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    with caplog.at_level(logging.INFO):
        await manager.warm_all()  # must not raise

    assert manager.snapshot()["models"]["tool_intent"]["state"] == lmr.FAILED
    assert any("local-model not-ready name=tool_intent" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_warmup_generates_exactly_once_per_run(monkeypatch):
    _clear_endpoints(monkeypatch)
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "http://warm.test/v1")
    ps_calls = {"n": 0}
    generates: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            ps_calls["n"] += 1
            return httpx.Response(200, json={"models": [{"name": "delveta-pe:q5km"}]})
        generates.append(request)
        return httpx.Response(200, json={"response": "", "done": True})

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()
    await manager.warm_all()

    assert len(generates) == 2  # one warm-up per run, never a poll loop of them

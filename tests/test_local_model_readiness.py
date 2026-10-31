"""Startup load, real-inference warm-up, residency and Ready for local models.

No network: ``httpx.MockTransport`` stands behind the module's ``_client`` seam.
Every probe here is a REAL wire shape (the embedding ``/embed`` body, the
production ``/v1/systemone`` choice question, Ollama's native
``/api/generate`` + ``/api/ps``) — the tests fail if a probe degrades into a
plain health check.
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from api import local_model_readiness as lmr
from core.config import settings

_EMBED_VECTOR = [1.0] + [0.0] * 1023


def _configure(monkeypatch, *, embedding="http://embed.test", cap_router="",
               tool_intent="", tts="", stt="") -> None:
    monkeypatch.setattr(settings, "embedding_base_url", embedding)
    monkeypatch.setattr(settings, "chat_cap_router_url", cap_router)
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", tool_intent)
    monkeypatch.setattr(settings, "chat_tool_intent_local_model", "delveta-pe:q5km")
    monkeypatch.setattr(settings, "tts_base_url", tts)
    monkeypatch.setattr(settings, "stt_base_url", stt)


def _patch_client(monkeypatch, handler) -> None:
    real = httpx.AsyncClient

    def factory(timeout):
        return real(timeout=timeout, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(lmr, "_client", factory)


def _embedding_handler(requests: list, *, vector=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/embed":
            return httpx.Response(200, json=[vector or _EMBED_VECTOR])
        return httpx.Response(404, json={"detail": "no such route"})

    return handler


@pytest.mark.asyncio
async def test_embedding_warmup_runs_a_real_embedding_and_is_ready(monkeypatch, caplog):
    _configure(monkeypatch, tts="", stt="")
    seen: list[httpx.Request] = []
    _patch_client(monkeypatch, _embedding_handler(seen))

    manager = lmr.LocalModelReadiness()
    with caplog.at_level(logging.INFO):
        await manager.warm_all()

    embed_calls = [r for r in seen if r.url.path == "/embed"]
    assert len(embed_calls) == 1
    payload = json.loads(embed_calls[0].content)
    assert payload["inputs"] == [lmr._EMBED_PROBE_TEXT]
    assert payload["normalize"] is True

    state = manager.snapshot()["models"]["embedding"]
    assert state["state"] == lmr.READY
    assert state["load_ms"] is not None
    assert state["warmup_ms"] is not None
    assert manager.ready is True

    messages = [r.getMessage() for r in caplog.records]
    assert any("local-model model-loaded name=embedding" in m for m in messages)
    assert any("local-model warmup-completed name=embedding" in m for m in messages)
    assert any("local-model ready name=embedding" in m for m in messages)


@pytest.mark.asyncio
async def test_embedding_with_the_wrong_dimension_is_not_ready(monkeypatch, caplog):
    _configure(monkeypatch)
    _patch_client(monkeypatch, _embedding_handler([], vector=[0.5] * 512))

    manager = lmr.LocalModelReadiness()
    with caplog.at_level(logging.INFO):
        await manager.warm_all()

    state = manager.snapshot()["models"]["embedding"]
    assert state["state"] == lmr.FAILED
    assert "expected 1024" in state["error"]
    assert manager.ready is False
    assert any("local-model not-ready name=embedding" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_cap_router_asks_the_production_choice_wire(monkeypatch):
    _configure(monkeypatch, embedding="", cap_router="http://cap.test")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok",
                                             "model": "/model/current",
                                             "device": "cpu"})
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json={"answers": {
                "capability": {"type": "choice", "choice": "cap-warmup",
                               "probabilities": {"cap-warmup": 0.9}}},
            })
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    posts = [r for r in seen if r.url.path == "/v1/systemone"]
    assert len(posts) == 1
    payload = json.loads(posts[0].content)
    assert payload["state"] == "warmup"
    question = payload["questions"]["capability"]
    assert question["type"] == "choice"
    assert "cap-warmup" in question["criteria"]
    assert "REJECT" in question["criteria"]

    state = manager.snapshot()["models"]["cap_router"]
    assert state["state"] == lmr.READY
    assert state["detail"] == "model=/model/current"


@pytest.mark.asyncio
async def test_cap_router_off_card_choice_is_not_ready(monkeypatch):
    _configure(monkeypatch, embedding="", cap_router="http://cap.test")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"answers": {
            "capability": {"type": "choice", "choice": "ghost"}}})

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    state = manager.snapshot()["models"]["cap_router"]
    assert state["state"] == lmr.FAILED
    assert "off-card" in state["error"]


@pytest.mark.asyncio
async def test_cap_router_http_200_while_loading_is_not_ready(monkeypatch):
    _configure(monkeypatch, embedding="", cap_router="http://cap.test")
    monkeypatch.setitem(lmr._WARM_TIMEOUTS, "cap_router", 0.05)
    _patch_client(monkeypatch,
                  lambda request: httpx.Response(200, json={"status": "loading"}))

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    state = manager.snapshot()["models"]["cap_router"]
    assert state["state"] == lmr.FAILED
    assert "did not finish loading" in state["error"]


@pytest.mark.asyncio
async def test_tool_intent_generates_with_keep_alive_and_confirms_residency(
        monkeypatch, caplog):
    _configure(monkeypatch, embedding="", tool_intent="http://ollama.test/v1")
    posts: list[dict] = []
    ps_phase = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            ps_phase["n"] += 1
            if ps_phase["n"] == 1:
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"models": [{
                "name": "delveta-pe:q5km", "model": "delveta-pe:q5km",
                "size": 987654321, "size_vram": 0,
                "expires_at": "2318-01-01T00:00:00Z"}]})
        if request.url.path == "/api/generate":
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"response": "", "done": True})
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    with caplog.at_level(logging.INFO):
        await manager.warm_all()

    assert len(posts) == 1
    assert posts[0]["model"] == "delveta-pe:q5km"
    assert posts[0]["keep_alive"] == -1
    assert posts[0]["stream"] is False
    assert posts[0]["options"] == {"num_predict": 1}

    state = manager.snapshot()["models"]["tool_intent"]
    assert state["state"] == lmr.READY
    assert any("local-model residency-confirmed name=tool_intent" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_tool_intent_not_resident_after_warmup_is_not_ready(monkeypatch):
    _configure(monkeypatch, embedding="", tool_intent="http://ollama.test/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"response": "", "done": True})

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    state = manager.snapshot()["models"]["tool_intent"]
    assert state["state"] == lmr.FAILED
    assert "not resident" in state["error"]


@pytest.mark.asyncio
async def test_a_required_failure_makes_the_whole_snapshot_not_ready(monkeypatch):
    _configure(monkeypatch)
    _patch_client(monkeypatch,
                  lambda request: httpx.Response(500, text="not ready"))

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    snapshot = manager.snapshot()
    assert snapshot["status"] == "not_ready"
    assert snapshot["models"]["embedding"]["state"] == lmr.FAILED
    assert "500" in snapshot["models"]["embedding"]["error"]


@pytest.mark.asyncio
async def test_optional_models_never_gate_readiness(monkeypatch):
    _configure(monkeypatch, embedding="http://embed.test",
               tts="http://tts.test/v1", stt="http://stt.test/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "embed.test":
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(200, json=[_EMBED_VECTOR])
        return httpx.Response(503, text="down")

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    snapshot = manager.snapshot()
    assert snapshot["models"]["tts"]["kind"] == lmr.OPTIONAL
    assert snapshot["models"]["tts"]["state"] == lmr.FAILED
    assert snapshot["models"]["stt"]["state"] == lmr.FAILED
    assert snapshot["status"] == "ready"
    assert manager.ready is True


@pytest.mark.asyncio
async def test_a_reachable_optional_model_is_not_rewarmed_every_tick(monkeypatch,
                                                                     caplog):
    _configure(monkeypatch, embedding="", tts="http://tts.test/v1")
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"status": "healthy"})

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    with caplog.at_level(logging.INFO):
        await manager.warm_all()
        await manager._recheck_all()

    # One health probe per pass; the optional model is not run through the
    # load/warm sequence again once it is reachable.
    assert paths.count("/health") == 2
    assert sum("local-model reachable name=tts" in r.getMessage()
               for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_unconfigured_models_are_reported_and_never_probed(monkeypatch):
    _configure(monkeypatch, embedding="", cap_router="", tool_intent="",
               tts="", stt="")
    seen: list[httpx.Request] = []
    _patch_client(monkeypatch, lambda request: seen.append(request))

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()

    assert seen == []
    snapshot = manager.snapshot()
    assert snapshot["status"] == "ready"
    assert all(m["state"] == lmr.NOT_CONFIGURED
               for m in snapshot["models"].values())


@pytest.mark.asyncio
async def test_a_restarted_model_is_reloaded_and_rewarmed(monkeypatch):
    _configure(monkeypatch, embedding="http://embed.test")
    healthy = {"value": True}
    embed_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not healthy["value"]:
            return httpx.Response(503, text="sidecar restarting")
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        embed_calls.append(1)
        return httpx.Response(200, json=[_EMBED_VECTOR])

    _patch_client(monkeypatch, handler)

    manager = lmr.LocalModelReadiness()
    await manager.warm_all()
    assert manager.snapshot()["models"]["embedding"]["state"] == lmr.READY
    assert len(embed_calls) == 1

    healthy["value"] = False
    await manager._recheck_all()
    assert manager.snapshot()["models"]["embedding"]["state"] == lmr.FAILED
    assert manager.ready is False

    healthy["value"] = True
    await manager._recheck_all()
    # Recovery runs the FULL sequence again: the probe re-embeds, it does not
    # promote the model on health alone.
    assert manager.snapshot()["models"]["embedding"]["state"] == lmr.READY
    assert len(embed_calls) == 2


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_cancels_the_only_task(monkeypatch):
    _configure(monkeypatch, embedding="", cap_router="", tool_intent="",
               tts="", stt="")
    manager = lmr.LocalModelReadiness(interval=3600)
    calls: list[int] = []
    started = asyncio.Event()
    never = asyncio.Event()

    async def fake_warm_all() -> None:
        calls.append(1)
        started.set()
        await never.wait()

    manager.warm_all = fake_warm_all

    manager.start()
    manager.start()
    await asyncio.wait_for(started.wait(), timeout=1)

    assert calls == [1]
    task = manager._task
    assert task is not None

    await manager.stop()

    assert manager._task is None
    assert task.done() or task.cancelled()


@pytest.mark.asyncio
async def test_health_endpoints_separate_liveness_from_model_readiness(monkeypatch):
    from api import main as api_main

    monkeypatch.setattr(api_main, "_readiness", None)
    response = await api_main.health_ready()
    assert response.status_code == 503
    assert json.loads(response.body)["reason"] == "readiness_not_started"

    assert await api_main.health() == {"status": "ok"}
    assert await api_main.health_live() == {"status": "alive"}

    _configure(monkeypatch, embedding="", cap_router="", tool_intent="",
               tts="", stt="")
    manager = lmr.LocalModelReadiness()
    await manager.warm_all()
    monkeypatch.setattr(api_main, "_readiness", manager)

    ready = await api_main.health_ready()
    assert ready.status_code == 200
    body = json.loads(ready.body)
    assert body["status"] == "ready"
    assert set(body["models"]) == {"embedding", "cap_router", "tool_intent",
                                   "tts", "stt"}

    manager._states["embedding"].state = lmr.LOADING
    not_ready = await api_main.health_ready()
    assert not_ready.status_code == 503
    assert json.loads(not_ready.body)["status"] == "not_ready"

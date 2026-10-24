"""LLM connection pooling: one transport per origin, credential wrappers cached beside it.

These tests run against a real local HTTP server (not a fake client) so they can observe
TCP accepts and the Authorization header actually on the wire. The invariants being
locked in:

* N sequential calls on one pinned channel open ONE TCP connection;
* credentials are isolated per wrapper even though they share that connection;
* evicting a wrapper (LRU, TTL, explicit invalidation) never closes the transport and
  never disturbs an in-flight request;
* the transport is closed exactly once, at shutdown, and only through
  ``aclose_llm_clients``.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from core.config import settings
from core.infrastructure import llm as llm_mod
from core.infrastructure.llm import OpenAILLM, aclose_llm_clients, invalidate_channel
from core.infrastructure.request_context import (
    reset_request_llm_channel,
    set_request_llm_channel,
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        auth = self.headers.get("Authorization")
        self.server.requests.append({"path": self.path, "auth": auth})

        if self.path.startswith("/slow/"):
            time.sleep(0.4)
        if self.path.startswith("/redir/"):
            self.send_response(308)
            self.send_header("Location", "/v1/chat/completions")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        payload = json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant",
                            "content": json.dumps({"echo": auth})},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepts = 0
        self.requests: list[dict] = []

    def get_request(self):
        conn, addr = super().get_request()
        self.accepts += 1
        return conn, addr


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
async def _clean_pool():
    """Every test starts and ends with an empty pool (module state is process-wide)."""
    await aclose_llm_clients()
    yield
    await aclose_llm_clients()


def _url(server: _Server, prefix: str = "/v1") -> str:
    return f"http://127.0.0.1:{server.server_address[1]}{prefix}"


def _pin(base_url: str, api_key: str, model: str = "test-model"):
    return set_request_llm_channel((model, base_url, api_key))


def _count_aclose(monkeypatch):
    """Replace ``httpx.AsyncClient.aclose`` with a counting pass-through."""
    calls: list[httpx.AsyncClient] = []
    orig = httpx.AsyncClient.aclose

    async def counting(client):
        calls.append(client)
        await orig(client)

    monkeypatch.setattr(httpx.AsyncClient, "aclose", counting)
    return calls


async def _one_call(llm: OpenAILLM) -> dict:
    return await llm.complete_json_once("say json", "Return a JSON object.")


# -- 1: reuse ------------------------------------------------------------------------

async def test_sequential_calls_open_one_connection(server):
    llm = OpenAILLM(require_channel=True)
    token = _pin(_url(server), "sk-reuse")
    try:
        for _ in range(5):
            assert (await _one_call(llm))["echo"] == "Bearer sk-reuse"
    finally:
        reset_request_llm_channel(token)
    assert server.accepts == 1, f"expected 1 TCP accept, saw {server.accepts}"
    assert len(server.requests) == 5


async def test_one_transport_per_origin_shared_by_all_credentials(server):
    url = _url(server)
    a = llm_mod._get_wrapper(url, "sk-a")
    b = llm_mod._get_wrapper(url, "sk-b")
    assert a is not b
    assert a._client is b._client
    assert llm_mod._get_wrapper(url, "sk-a") is a


async def test_different_origins_get_different_transports(server):
    a = llm_mod._get_wrapper(_url(server), "sk-a")
    # Same host, different port => different origin, different transport. No request is
    # made through ``b``, so nothing needs to listen on that port.
    b = llm_mod._get_wrapper(f"http://127.0.0.1:{server.server_address[1] + 1}/v1", "sk-a")
    assert a._client is not b._client


async def test_per_call_timeout_reuses_the_same_transport(server):
    llm = OpenAILLM(require_channel=True)
    token = _pin(_url(server), "sk-timeout")
    try:
        base, _ = llm._call_channel()
        quick, _ = llm._call_channel(timeout=3.0)
        retried, _ = llm._call_channel(max_retries=0)
    finally:
        reset_request_llm_channel(token)
    assert quick is not base
    assert quick._client is base._client
    assert quick.timeout == 3.0
    assert base.max_retries == 2  # SDK default; pooling does not change it
    assert retried.max_retries == 0
    assert retried._client is base._client


# -- 2: credential isolation ---------------------------------------------------------

async def test_auth_headers_are_isolated_per_credential(server):
    url = _url(server)
    a = llm_mod._get_wrapper(url, "sk-tenant-a")
    b = llm_mod._get_wrapper(url, "sk-tenant-b")
    messages = [{"role": "user", "content": "hi"}]
    ra, rb = await asyncio.gather(
        a.chat.completions.create(model="test-model", messages=messages),
        b.chat.completions.create(model="test-model", messages=messages),
    )
    assert json.loads(ra.choices[0].message.content)["echo"] == "Bearer sk-tenant-a"
    assert json.loads(rb.choices[0].message.content)["echo"] == "Bearer sk-tenant-b"
    assert all(r["auth"] in ("Bearer sk-tenant-a", "Bearer sk-tenant-b")
               for r in server.requests)


# -- 3: eviction is a reference removal, never a close -------------------------------

async def test_in_flight_request_survives_wrapper_eviction(server):
    llm = OpenAILLM(require_channel=True)
    url = _url(server, prefix="/slow/v1")
    token = _pin(url, "sk-inflight")
    try:
        task = asyncio.create_task(_one_call(llm))
        while not server.requests:  # the request is on the wire before we evict
            await asyncio.sleep(0.01)
        assert invalidate_channel(base_url=url, api_key="sk-inflight") == 1
        assert (await task)["echo"] == "Bearer sk-inflight"
    finally:
        reset_request_llm_channel(token)


async def test_evicted_wrapper_remains_usable(server):
    url = _url(server)
    held = llm_mod._get_wrapper(url, "sk-held")
    assert invalidate_channel(base_url=url, api_key="sk-held") == 1
    fresh = llm_mod._get_wrapper(url, "sk-held")
    assert fresh is not held
    resp = await held.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert json.loads(resp.choices[0].message.content)["echo"] == "Bearer sk-held"


async def test_lru_eviction_never_closes_the_transport(server, monkeypatch):
    monkeypatch.setattr(llm_mod, "_WRAPPER_CACHE_MAX", 3)
    close_calls = _count_aclose(monkeypatch)

    url = _url(server)
    first = llm_mod._get_wrapper(url, "sk-key-0")
    for i in range(1, 6):
        llm_mod._get_wrapper(url, f"sk-key-{i}")

    assert len(llm_mod._WRAPPERS) == 3
    assert close_calls == []
    assert not first._client.is_closed
    resp = await first.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content


async def test_ttl_expiry_rebuilds_the_wrapper_but_keeps_the_transport(server, monkeypatch):
    # TTL 0 => the entry is always stale on the next lookup.
    monkeypatch.setattr(llm_mod, "_WRAPPER_CACHE_TTL_SECONDS", 0)
    url = _url(server)
    first = llm_mod._get_wrapper(url, "sk-ttl")
    second = llm_mod._get_wrapper(url, "sk-ttl")
    assert second is not first
    assert second._client is first._client


async def test_aclose_closes_transports_exactly_once(server, monkeypatch):
    close_calls = _count_aclose(monkeypatch)

    llm_mod._get_wrapper(_url(server), "sk-shutdown")
    await aclose_llm_clients()
    await aclose_llm_clients()

    assert len(close_calls) == 1
    assert llm_mod._TRANSPORTS == {}
    assert llm_mod._WRAPPERS == {}


# -- 4: transport semantics preserved ------------------------------------------------

async def test_redirects_are_still_followed(server):
    wrapper = llm_mod._get_wrapper(_url(server, prefix="/redir"), "sk-redirect")
    resp = await wrapper.chat.completions.create(
        model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert json.loads(resp.choices[0].message.content)["echo"] == "Bearer sk-redirect"
    assert [r["path"] for r in server.requests] == [
        "/redir/chat/completions", "/v1/chat/completions"]


async def test_default_timeout_and_retry_budget_are_unchanged(server):
    wrapper = llm_mod._get_wrapper(_url(server), "sk-defaults")
    assert wrapper.timeout == settings.llm_timeout_seconds  # per-call wall time, unchanged
    assert wrapper.max_retries == 2  # SDK default, unchanged (was never passed before)


async def test_cached_keys_never_hold_plaintext_credentials(server):
    url = _url(server)
    llm_mod._get_wrapper(url, "sk-super-secret")
    for base_url, fingerprint in llm_mod._WRAPPERS:
        assert "sk-" not in base_url
        assert "secret" not in fingerprint
        assert len(fingerprint) == 64
        int(fingerprint, 16)  # sha256 hex


async def test_no_resolved_channel_keeps_the_legacy_client_path(server, monkeypatch):
    def _explode(*args, **kwargs):  # the pool must not be touched without a channel
        raise AssertionError("pool used for a call with no resolved channel")

    monkeypatch.setattr(llm_mod, "_get_wrapper", _explode)
    plain = OpenAILLM(model="self-model")
    client, model = plain._call_channel()
    assert client is plain.client
    assert model == "self-model"

"""Same-turn search reuse, tool-call ordering, and embedding attribution.

Deterministic only: no network, no database. The reuse cache is the per-turn
``AgentTurn.search_results`` dict, so each test binds a turn explicitly (or drives the
real loop, which binds one itself).
"""
import asyncio
import logging
from contextlib import contextmanager

import httpx

from agent import (
    ReactLoopAgent,
    SystemPrompt,
    ToolOutput,
    ToolRuntime,
    define_tool,
    text_block,
)
from agent.engine.context import AgentTurn, _TURN_CTX
from agent.engine.decisions import ToolExecution
from agent.harness import FakeLLM, assistant
from agent.memory.service import memory_search_tool
from agent.tools import search_reuse
from agent.tools.embedding_source import embedding_source
from core.infrastructure.request_context import get_request_embedding_source
from core.infrastructure.vector import TEIEmbedder

from apps.api.tools import web_search_tool
from apps.api.tools.rag_search_tool import _UNAVAILABLE, register as register_rag


@contextmanager
def _turn():
    """Bind a fresh AgentTurn to the current context, restoring it afterwards."""
    turn = AgentTurn(user_msg="hi")
    token = _TURN_CTX.set(turn)
    try:
        yield turn
    finally:
        _TURN_CTX.reset(token)


# ── pure reuse semantics ────────────────────────────────────────────────────────


def test_key_normalizes_whitespace_and_case_only():
    a = search_reuse.key("rag_search", "  MFU   Model\nFLOPs ", 8, None)
    b = search_reuse.key("rag_search", "mfu model flops", 8)
    assert a == b
    # Re-wording is a different search: normalization never rewrites words.
    assert search_reuse.key("rag_search", "mfu flops", 8) != a


def test_different_tool_top_k_or_scope_is_not_a_hit():
    with _turn():
        base = search_reuse.key("rag_search", "q", 5, None)
        search_reuse.record(base, [{"id": "c1"}])

        assert search_reuse.lookup(base) == [{"id": "c1"}]
        assert search_reuse.lookup(search_reuse.key("memory_search", "q", 5)) is None
        assert search_reuse.lookup(search_reuse.key("rag_search", "q", 8)) is None
        assert search_reuse.lookup(search_reuse.key("rag_search", "q", 5, "domain-a")) is None


def test_empty_result_list_is_still_a_cache_hit():
    with _turn():
        key = search_reuse.key("web_search", "q", 5)
        search_reuse.record(key, [])
        assert search_reuse.lookup(key) == []


def test_no_turn_is_a_no_op():
    key = search_reuse.key("rag_search", "q", 5)
    assert search_reuse.lookup(key) is None
    search_reuse.record(key, [{"id": "c1"}])  # must not raise


def test_non_list_values_are_not_cached():
    with _turn():
        key = search_reuse.key("rag_search", "q", 5)
        search_reuse.record(key, {"notice": "nope"})
        assert search_reuse.lookup(key) is None


def test_turns_do_not_share_results():
    key = search_reuse.key("rag_search", "q", 5)
    with _turn() as first:
        search_reuse.record(key, [{"id": "c1"}])
        assert first.search_results
    with _turn():
        assert search_reuse.lookup(key) is None


# ── one tool at a time, through the real runtime ───────────────────────────────


class _CountingRetriever:
    def __init__(self, hits=None, exc=None):
        self.calls = []
        self.hits = hits if hits is not None else [{"id": "c1", "text": "chunk"}]
        self.exc = exc

    async def retrieve(self, query, top_k, filters):
        self.calls.append((query, top_k, dict(filters)))
        if self.exc is not None:
            raise self.exc
        return self.hits


class _RagCtx:
    def __init__(self, retriever):
        self._retriever = retriever

    def resolve(self, key):
        assert key == "retrieval"
        return self._retriever


def _rag_runtime(retriever):
    runtime = ToolRuntime()
    register_rag(runtime, _RagCtx(retriever), None)
    return runtime


async def test_rag_search_reuses_an_identical_successful_call():
    retriever = _CountingRetriever()
    runtime = _rag_runtime(retriever)

    with _turn():
        first = await runtime.execute(
            ToolExecution(call_id="a", name="rag_search", arguments={"query": "MFU"})
        )
        second_exec = ToolExecution(
            call_id="b", name="rag_search", arguments={"query": "  mfu  ", "top_k": 5}
        )
        second = await runtime.execute(second_exec)

    assert len(retriever.calls) == 1
    assert second.value == first.value
    assert second_exec.trace["reused"] is True


async def test_rag_search_different_top_k_executes_normally():
    retriever = _CountingRetriever()
    runtime = _rag_runtime(retriever)

    with _turn():
        await runtime.execute(
            ToolExecution(call_id="a", name="rag_search", arguments={"query": "MFU", "top_k": 5})
        )
        await runtime.execute(
            ToolExecution(call_id="b", name="rag_search", arguments={"query": "MFU", "top_k": 8})
        )

    assert [call[1] for call in retriever.calls] == [5, 8]


async def test_rag_search_degraded_notice_is_never_cached():
    retriever = _CountingRetriever(exc=RuntimeError("embedding 503"))
    runtime = _rag_runtime(retriever)

    with _turn():
        first = await runtime.execute(
            ToolExecution(call_id="a", name="rag_search", arguments={"query": "MFU"})
        )
        second = await runtime.execute(
            ToolExecution(call_id="b", name="rag_search", arguments={"query": "MFU"})
        )

    assert len(retriever.calls) == 2
    assert first.value == _UNAVAILABLE
    assert second.value == _UNAVAILABLE


class _MemoryHit:
    def __init__(self, content):
        self.content = content

    def to_dict(self):
        return {"content": self.content}


class _MemoryService:
    def __init__(self):
        self.calls = []

    async def recall_all(self, query, top_k):
        self.calls.append((query, top_k))
        return [_MemoryHit("remembered")]


async def test_memory_search_reuses_an_identical_call():
    service = _MemoryService()
    runtime = ToolRuntime()
    runtime.register(memory_search_tool(service))

    with _turn():
        first = await runtime.execute(
            ToolExecution(call_id="a", name="memory_search", arguments={"query": "preference"})
        )
        second_exec = ToolExecution(
            call_id="b", name="memory_search", arguments={"query": "Preference ", "top_k": 5}
        )
        second = await runtime.execute(second_exec)

    assert len(service.calls) == 1
    assert second.value == first.value == [{"content": "remembered"}]
    assert second_exec.trace["reused"] is True


class _WebProvider:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return {"status": "ok", "results": [{"title": "t", "url": "https://example.test/a"}]}


class _WebCtx:
    def __init__(self, provider):
        self._provider = provider

    def resolve(self, key):
        assert key == "web_search"
        return self._provider


async def test_web_search_reuses_an_identical_call(monkeypatch):
    async def _no_serps(query, top_k):
        return []

    async def _no_pages(results):
        return []

    monkeypatch.setattr(web_search_tool, "_scrape_serps", _no_serps)
    monkeypatch.setattr(web_search_tool, "_crawl_pages", _no_pages)
    provider = _WebProvider()
    runtime = ToolRuntime()
    web_search_tool.register(runtime, _WebCtx(provider), None)

    with _turn():
        first = await runtime.execute(
            ToolExecution(call_id="a", name="web_search", arguments={"query": "latest news"})
        )
        second_exec = ToolExecution(
            call_id="b", name="web_search", arguments={"query": " latest  NEWS "}
        )
        second = await runtime.execute(second_exec)

    assert len(provider.calls) == 1
    assert second.value == first.value
    assert second_exec.trace["reused"] is True


# ── tool-call ordering ───────────────────────────────────────────────────────────


def _recording_tool(name: str, seen: list):
    async def body(args, exec):
        seen.append((name, exec.call_index, args["id"]))
        return {"ok": True}

    return define_tool(
        name=name,
        description=name,
        parameters={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        output=ToolOutput(schema={"type": "object"}, render=lambda args, value: [text_block("ok")]),
        execute=body,
        is_concurrency_safe=True,
    )


async def test_call_index_follows_turn_order_across_steps():
    seen = []
    runtime = ToolRuntime()
    runtime.register(_recording_tool("alpha", seen))
    runtime.register(_recording_tool("beta", seen))
    runtime.register(_recording_tool("gamma", seen))
    llm = FakeLLM([
        {
            "content": None,
            "tool_calls": [
                {"id": "t0", "name": "alpha", "arguments": '{"id": "0"}'},
                {"id": "t1", "name": "beta", "arguments": '{"id": "1"}'},
                {"id": "t2", "name": "gamma", "arguments": '{"id": "2"}'},
            ],
        },
        {"content": None, "tool_calls": [
            {"id": "t3", "name": "alpha", "arguments": '{"id": "3"}'},
        ]},
        assistant("done"),
    ])

    await ReactLoopAgent(llm, runtime, SystemPrompt()).run("hi")

    # Same-step calls keep call order; the next step continues the numbering.
    assert sorted(seen, key=lambda item: item[1]) == [
        ("alpha", 0, "0"), ("beta", 1, "1"), ("gamma", 2, "2"), ("alpha", 3, "3"),
    ]


# ── embedding attribution ───────────────────────────────────────────────────────


def test_embedding_source_restores_the_previous_value():
    with _turn():
        assert get_request_embedding_source() is None
        with embedding_source("rag_search"):
            with embedding_source("memory_search", call_id="c2", call_index=3):
                assert get_request_embedding_source() == {
                    "source": "memory_search",
                    "call_id": "c2",
                    "call_index": 3,
                }
            assert get_request_embedding_source()["source"] == "rag_search"
        assert get_request_embedding_source() is None


async def test_embedding_source_is_isolated_between_concurrent_tasks():
    observed = {}

    async def _worker(source: str, call_id: str):
        with embedding_source(source, call_id=call_id):
            await asyncio.sleep(0)  # let the sibling task interleave
            observed[source] = dict(get_request_embedding_source())

    await asyncio.gather(
        _worker("rag_search", "r1"),
        _worker("memory_search", "m1"),
    )

    assert observed["rag_search"]["source"] == "rag_search"
    assert observed["rag_search"]["call_id"] == "r1"
    assert observed["memory_search"]["source"] == "memory_search"
    assert observed["memory_search"]["call_id"] == "m1"
    assert get_request_embedding_source() is None


async def _embedder_with(handler) -> TEIEmbedder:
    embedder = TEIEmbedder()
    await embedder._client.aclose()
    embedder._client = httpx.AsyncClient(
        base_url="http://tei.test", transport=httpx.MockTransport(handler)
    )
    return embedder


async def test_tei_embed_logs_the_tool_call_that_asked_for_it(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    embedder = await _embedder_with(handler)
    caplog.set_level(logging.INFO, logger="core.infrastructure.vector")
    try:
        with embedding_source("rag_search", call_id="call-7", call_index=2):
            vectors = await embedder.embed(["hello"])
    finally:
        await embedder._client.aclose()

    assert vectors == [[0.1, 0.2, 0.3]]
    line = caplog.text
    assert "source=rag_search" in line
    assert "call_id=call-7" in line
    assert "call_index=2" in line
    assert "batch=1" in line
    assert "dims=3" in line


async def test_tei_embed_without_a_tool_context_is_unattributed(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.1, 0.2]])

    embedder = await _embedder_with(handler)
    caplog.set_level(logging.INFO, logger="core.infrastructure.vector")
    try:
        await embedder.embed(["ingest batch"])
    finally:
        await embedder._client.aclose()

    assert "source=unattributed" in caplog.text

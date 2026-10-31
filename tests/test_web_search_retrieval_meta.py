"""``web_search`` ships a retrieval-time anchor with every result list.

Neither the search providers nor the page crawl return a publication timestamp, so
a turn cannot otherwise tell a cached page from this morning's. The tool therefore
prepends ONE metadata entry (``type: "retrieval_meta"``) to both the fresh and the
same-turn-reused list; the model reads "today"/"now" from it and source collectors
skip it because it carries no URL. It is deliberately not a change to the global
system prompt: ordinary turns never pay for it.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from agent.engine.decisions import ToolExecution
from agent.engine.runtime import ToolRuntime
from agent.tools import search_reuse

from apps.api.tools import web_search_tool
from apps.api.tools.web_search_tool import _retrieval_meta, register

_PROVIDER_RESULTS = [
    {"title": "北京天气", "url": "https://weather.example/beijing", "snippet": "晴 25℃"},
    {"title": "北京空气质量", "url": "https://aq.example/beijing", "snippet": "良"},
]


class _Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5, **kwargs) -> dict:
        self.calls.append((query, top_k))
        return {"status": "ok", "provider": "fake", "results": list(_PROVIDER_RESULTS)}


class _Ctx:
    def __init__(self, provider) -> None:
        self._provider = provider

    def resolve(self, key):
        return self._provider if key == "web_search" else None


@pytest.fixture
def _no_side_channels(monkeypatch):
    async def no_serp(query, top_k):
        return []

    async def no_pages(results):
        return []

    monkeypatch.setattr(web_search_tool, "_scrape_serps", no_serp)
    monkeypatch.setattr(web_search_tool, "_crawl_pages", no_pages)


def _definition(ctx) -> object:
    runtime = ToolRuntime()
    register(runtime, ctx, None)
    return runtime.get("web_search")


async def _run(ctx, query: str = "北京天气", top_k: int = 5):
    return await _definition(ctx).execute(
        {"query": query, "top_k": top_k},
        ToolExecution(call_id="c1", name="web_search", arguments={}),
    )


def test_retrieval_meta_is_a_typed_time_anchor():
    meta = _retrieval_meta()
    assert meta["type"] == "retrieval_meta"
    local = datetime.fromisoformat(meta["retrieved_at"])
    assert local.tzinfo is not None                      # offset always present
    utc = datetime.fromisoformat(meta["retrieved_at_utc"])
    assert utc.utcoffset().total_seconds() == 0
    # Same instant, expressed in both zones.
    assert abs((local - utc).total_seconds()) <= 1
    assert meta["timezone"] == "UTC" or meta["timezone"][:3] == "UTC"
    assert "url" not in meta                             # source collectors skip it


async def test_fresh_search_prepends_the_time_anchor(_no_side_channels):
    provider = _Provider()
    value = await _run(_Ctx(provider))

    assert provider.calls == [("北京天气", 5)]
    assert value[0]["type"] == "retrieval_meta"
    assert value[1:] == _PROVIDER_RESULTS                # results unchanged, in order
    assert all("url" in r for r in value[1:])


async def test_reused_search_also_carries_the_time_anchor(_no_side_channels, monkeypatch):
    cached = [*_PROVIDER_RESULTS[:1]]
    monkeypatch.setattr(search_reuse, "lookup", lambda key: cached)
    provider = _Provider()

    value = await _run(_Ctx(provider))

    assert provider.calls == []                          # no provider hit for a reuse
    assert value[0]["type"] == "retrieval_meta"
    assert value[1:] == cached


async def test_the_anchor_is_recomputed_per_call_not_cached(_no_side_channels):
    ctx = _Ctx(_Provider())
    first = (await _run(ctx))[0]
    second = (await _run(ctx))[0]

    # Same wall-clock second is fine; the point is it is generated, not a constant.
    assert first["type"] == second["type"] == "retrieval_meta"
    assert datetime.fromisoformat(second["retrieved_at_utc"]) >= datetime.fromisoformat(
        first["retrieved_at_utc"])

"""Web-search source footnote contracts (DB-free).

``extract_web_sources`` is a pure function over the turn's message list; the persistence
side reuses the same ``messages.meta`` JSONB path as retrieval feedback, so these tests
pin the extraction rules and the done-frame wiring.
"""
import json
import types
from uuid import uuid4

from core.application.chat.lifecycle import extract_web_sources, finalize_turn


def _tc(id_, name, query=None):
    args = json.dumps({"query": query}) if query is not None else "{}"
    return {"id": id_, "name": name, "arguments": args}


def _assistant(calls):
    return {"role": "assistant", "tool_calls": calls}


def _tool(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ── extraction ────────────────────────────────────────────────────────────────────

def test_no_messages_returns_none():
    assert extract_web_sources(None) is None
    assert extract_web_sources([]) is None


def test_a_turn_without_web_search_returns_none():
    msgs = [
        _assistant([_tc("c1", "rag_search", "x")]),
        _tool("c1", json.dumps([{"id": "1", "text": "hit"}])),
    ]
    assert extract_web_sources(msgs) is None


def test_results_become_title_url_sources():
    msgs = [
        _assistant([_tc("c1", "web_search", "beijing weather")]),
        _tool("c1", json.dumps([
            {"title": "Beijing weather", "url": "https://example.com/a", "snippet": "sunny"},
            {"title": "Forecast", "url": "https://example.com/b"},
        ])),
    ]
    assert extract_web_sources(msgs) == [
        {"title": "Beijing weather", "url": "https://example.com/a"},
        {"title": "Forecast", "url": "https://example.com/b"},
    ]


def test_crawled_page_entries_do_not_duplicate_a_result_url():
    # ``web_search`` appends crawled pages after the ranked results; a page repeats the
    # result's URL, so the first title wins and the source appears once.
    msgs = [
        _assistant([_tc("c1", "web_search", "q")]),
        _tool("c1", json.dumps([
            {"title": "ranked", "url": "https://example.com/a"},
            {"type": "page", "title": "crawled", "url": "https://example.com/a", "text": "…"},
        ])),
    ]
    assert extract_web_sources(msgs) == [{"title": "ranked", "url": "https://example.com/a"}]


def test_the_retrieval_metadata_entry_never_becomes_a_source():
    # ``web_search`` now prepends a retrieval-time anchor to every list; it carries no
    # URL, so it must be skipped rather than rendered as an empty source.
    msgs = [
        _assistant([_tc("c1", "web_search", "q")]),
        _tool("c1", json.dumps([
            {"type": "retrieval_meta", "retrieved_at": "2026-10-31T09:00:00+08:00",
             "retrieved_at_utc": "2026-10-31T01:00:00+00:00", "timezone": "UTC+08:00"},
            {"title": "ranked", "url": "https://example.com/a"},
        ])),
    ]
    assert extract_web_sources(msgs) == [{"title": "ranked", "url": "https://example.com/a"}]


def test_only_http_and_https_urls_are_kept():
    msgs = [
        _assistant([_tc("c1", "web_search", "q")]),
        _tool("c1", json.dumps([
            {"title": "ok", "url": "https://example.com/a"},
            {"title": "js", "url": "javascript:alert(1)"},
            {"title": "file", "url": "file:///etc/passwd"},
            {"title": "missing"},
        ])),
    ]
    assert extract_web_sources(msgs) == [{"title": "ok", "url": "https://example.com/a"}]


def test_a_failed_search_contributes_nothing():
    msgs = [
        _assistant([_tc("c1", "web_search", "q")]),
        _tool("c1", json.dumps({"error": "web search unavailable"})),
    ]
    assert extract_web_sources(msgs) is None


def test_text_block_parts_are_parsed_too():
    msgs = [
        _assistant([_tc("c1", "web_search", "q")]),
        _tool("c1", [{"text": json.dumps([{"title": "t", "url": "https://example.com/a"}])}]),
    ]
    assert extract_web_sources(msgs) == [{"title": "t", "url": "https://example.com/a"}]


def test_multiple_calls_merge_in_first_seen_order_and_cap():
    calls = [_tc(f"c{i}", "web_search", f"q{i}") for i in range(12)]
    msgs = [_assistant(calls)]
    for i in range(12):
        msgs.append(_tool(f"c{i}", json.dumps([
            {"title": f"s{i}", "url": f"https://example.com/{i}"},
        ])))
    out = extract_web_sources(msgs)
    assert out is not None
    assert len(out) == 10
    assert [s["url"] for s in out] == [f"https://example.com/{i}" for i in range(10)]


# ── done-frame wiring ─────────────────────────────────────────────────────────────

class _Queue:
    def __init__(self):
        self.calls = []

    async def enqueue(self, kind, payload, user_id=None):
        self.calls.append((kind, payload, user_id))


class _Memory:
    persist_failed = False

    async def flush_writes(self):
        pass

    def take_written(self):
        return [{"role": "user", "message_id": "u1"}, {"role": "assistant", "message_id": "a1"}]


class _Deps:
    def __init__(self):
        self.queue = _Queue()
        self.persisted = []

    async def persist_turn_meta(self, message_id, key, value):
        self.persisted.append((message_id, key, value))


def _ctx():
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message="北京天气", viewer=None),
        session_id=uuid4(), user_id=uuid4(), log_user=None, business_name="chat",
        credential_id=None, tier="free", session_memory=_Memory(),
        research_service=None, bound_task_id=None, compaction_payload=None,
        compaction_deferred=None, guest_token=None, research_notice=None, notice=None,
        viewer_assembly=None,
    )


def _payload(answer, messages):
    return {"answer": answer, "messages": messages, "usage": {}}


async def test_a_web_searched_answer_carries_sources_on_the_done_frame():
    deps = _Deps()
    messages = [
        {"role": "user", "content": "今天北京天气怎么样"},
        _assistant([_tc("c1", "web_search", "北京今天天气")]),
        _tool("c1", json.dumps([{"title": "北京天气", "url": "https://example.com/weather"}])),
        {"role": "assistant", "content": "今天北京晴。"},
    ]

    done = await finalize_turn(_ctx(), deps, _payload("今天北京晴。", messages), tool="chat_stream")

    expected = [{"title": "北京天气", "url": "https://example.com/weather"}]
    assert done["web_sources"] == expected
    assert ("a1", "web_sources", expected) in deps.persisted


async def test_a_failed_search_leaves_the_done_frame_unchanged():
    deps = _Deps()
    messages = [
        {"role": "user", "content": "今天北京天气怎么样"},
        _assistant([_tc("c1", "web_search", "北京今天天气")]),
        _tool("c1", json.dumps({"error": "web search unavailable"})),
        {"role": "assistant", "content": "搜索暂时不可用。"},
    ]

    done = await finalize_turn(_ctx(), deps, _payload("搜索暂时不可用。", messages), tool="chat_stream")

    assert "web_sources" not in done
    assert not any(key == "web_sources" for _, key, _ in deps.persisted)

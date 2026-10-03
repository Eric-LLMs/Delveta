"""Unit surface for ``argument_acquisition.extractor`` (Phase 4).

Locks the EXTRACTION-ONLY adapter contract: ``_parse_args`` accepts exactly the
three documented reply shapes and rejects everything else; ``extract`` posts the
pinned decoding payload to the local OpenAI-compatible endpoint and maps every
transport/malformed failure to :class:`ExtractionUnavailable` (fail-closed to the
Agent) — while a WELL-FORMED empty reply is a normal empty extraction, never a
fabricated value.

No endpoint, no model, no network: ``httpx.MockTransport`` stands in for the
local server, so the tests pin behavior, not a deployment.
"""
from __future__ import annotations

import types

import httpx
import pytest

from core.application.chat.intent_funnel.argument_acquisition import context_bundle
from core.application.chat.intent_funnel.argument_acquisition import extractor as ex
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
    _parse_args,
)
from core.config import settings

SOURCE_QUERY = context_bundle.SOURCE_QUERY


def _entry(params: dict | None = None):
    return types.SimpleNamespace(
        capability_id="cap-x", tool_binding="demo_tool",
        parameters=params or {"pages": {"type": "string", "required": True,
                                        "description": "page range"}},
    )


def _bundle(query: str = "make a folder called Notes"):
    return context_bundle.build(query, source=SOURCE_QUERY)


def _patch_transport(monkeypatch, handler):
    """Route the extractor's ``httpx.AsyncClient`` through a MockTransport.

    Patch the client NAME on the shared httpx module (monkeypatch restores it);
    the real class is captured first so the factory builds a genuine client that
    simply carries our transport — real status/JSON/exception machinery runs."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ex.httpx, "AsyncClient", factory)


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


# ── _parse_args: the three documented shapes ─────────────────────────────────────


def test_parse_args_json_wrapper_shape():
    # (1) the asked-for wire: {"arguments": {...}}
    got = _parse_args('{"arguments": {"name": "Notes"}}', ["name"])
    assert got == {"name": "Notes"}


def test_parse_args_bare_slot_map_shape():
    # (2) the wrapper dropped: the object IS the slot map (keys ⊆ requested)
    got = _parse_args('{"name": "Notes", "colour": "red"}', ["name", "colour"])
    assert got == {"name": "Notes", "colour": "red"}


def test_parse_args_markdown_lines_shape():
    # (3) the argmax-drift shape: "- slot: value" lines, filtered to requested
    text = "Here you go:\n- name: Notes\n* colour: red\nnoise: dropped"
    got = _parse_args(text, ["name", "colour"])
    assert got == {"name": "Notes", "colour": "red"}  # "noise" is not requested


def test_parse_args_bare_map_with_unrequested_key_yields_no_args():
    # shape (2) requires keys ⊆ requested; a stray key makes it NOT the slot map.
    assert _parse_args('{"name": "Notes", "extra": "x"}', ["name"]) == {}


def test_parse_args_arguments_not_an_object_raises():
    with pytest.raises(ExtractionUnavailable):
        _parse_args('{"arguments": "oops"}', ["name"])


def test_parse_args_malformed_reply_raises():
    with pytest.raises(ExtractionUnavailable):
        _parse_args("", ["name"])


# ── extract: transport / response handling ───────────────────────────────────────


async def test_extract_success_returns_values_and_source(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url",
                        "http://localhost:18091/v1")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return _reply('{"arguments": {"pages": "3"}}')

    _patch_transport(monkeypatch, handler)
    values, source = await ex.extract(query="go to page 3", entry=_entry(),
                                      model_slots=["pages"], bundle=_bundle())
    assert values == {"pages": "3"}
    assert source == SOURCE_QUERY
    assert seen["url"].endswith("/v1/chat/completions")


async def test_extract_transport_failure_raises(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url",
                        "http://localhost:18091/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(ExtractionUnavailable):
        await ex.extract(query="q", entry=_entry(), model_slots=["pages"],
                         bundle=_bundle())


async def test_extract_http_status_error_raises(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url",
                        "http://localhost:18091/v1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(ExtractionUnavailable):
        await ex.extract(query="q", entry=_entry(), model_slots=["pages"],
                         bundle=_bundle())


async def test_extract_empty_wellformed_reply_is_empty_extraction(monkeypatch):
    # A well-formed reply with no values is NOT an error: the Binder then lands
    # MISSING and the Agent owns the clarification. No value is fabricated.
    monkeypatch.setattr(settings, "chat_tool_intent_local_url",
                        "http://localhost:18091/v1")
    _patch_transport(monkeypatch, lambda request: _reply('{"arguments": {}}'))
    values, source = await ex.extract(query="q", entry=_entry(),
                                      model_slots=["pages"], bundle=_bundle())
    assert values == {}
    assert source == SOURCE_QUERY


async def test_extract_malformed_body_raises(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url",
                        "http://localhost:18091/v1")
    _patch_transport(monkeypatch, lambda request: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(ExtractionUnavailable):
        await ex.extract(query="q", entry=_entry(), model_slots=["pages"],
                         bundle=_bundle())


async def test_extract_without_endpoint_raises(monkeypatch):
    monkeypatch.setattr(settings, "chat_tool_intent_local_url", "")
    with pytest.raises(ExtractionUnavailable):
        await ex.extract(query="q", entry=_entry(), model_slots=["pages"],
                         bundle=_bundle())

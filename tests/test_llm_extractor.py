"""Unit surface for ``argument_acquisition.llm_extractor`` (Qwen → LLM).

Locks the approved two-stage seam:

1. the local extractor (Qwen) is the normal path — a complete answer returns
   immediately and the request's LLM is never called;
2. the request's OWN LLM (``deps.llm.complete``) re-extracts the FULL parameter
   set only when Qwen timed out, raised, returned an unusable reply, or left a
   required slot empty;
3. an LLM reply that is itself incomplete/failed is RETURNED (or, when the local
   stage produced nothing, raised) so the funnel's existing required check and
   Binder route the turn to the Agent — this module never fabricates values.

The LLM client is the caller's: no model/base_url/api_key override is passed, so
the request's channel, connection pool and billing context apply unchanged.
No network: the local stage is replaced by a fake; the LLM is a recording double.
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.intent_funnel.argument_acquisition import context_bundle
from core.application.chat.intent_funnel.argument_acquisition import llm_extractor as llm_ex
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
)
from core.config import settings

SOURCE_QUERY = context_bundle.SOURCE_QUERY
SOURCE_TURNS = context_bundle.SOURCE_CONVERSATION_5_USER_TURNS


def _entry(params: dict | None = None):
    return types.SimpleNamespace(
        capability_id="cap-create-folder",
        tool_binding="create_folder",
        description="Create a folder in the user's drive.",
        parameters=params or {
            "name": {"type": "string", "required": True,
                     "description": "folder name"},
            "color": {"type": "string", "required": False,
                      "description": "optional label colour"},
        },
    )


def _bundle(query: str = "make a folder called Notes", *, history=(), source=SOURCE_QUERY):
    return context_bundle.build(query, history=history, source=source)


class _FakeLLM:
    """Recording stand-in for the request-scoped ``LLM``.

    ``reply`` is the ``complete()`` return value; ``error`` is raised instead.
    Every call's prompt and kwargs are captured so the test can prove WHICH
    fields were passed (and which were deliberately not)."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def complete(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.reply


def _patch_qwen(monkeypatch, *, values=None, error=None, record=None):
    """Replace stage 1 with a fake that records its call kwargs."""

    async def fake_extract(*, query, entry, model_slots, bundle, prompt=None):
        if record is not None:
            record.update(
                query=query, entry=entry, model_slots=tuple(model_slots),
                bundle=bundle, prompt=prompt,
            )
        if error is not None:
            raise error
        return dict(values or {}), bundle.source

    monkeypatch.setattr(llm_ex, "_qwen_extract", fake_extract)


# ── 1. Qwen complete → no LLM call ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_qwen_complete_never_calls_the_llm(monkeypatch):
    _patch_qwen(monkeypatch, values={"name": "Notes"})
    llm = _FakeLLM(reply='{"arguments": {"name": "WRONG"}}')

    values, source = await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"name": "Notes"}
    assert source == SOURCE_QUERY
    assert llm.calls == []


@pytest.mark.asyncio
async def test_qwen_complete_ignores_missing_optional_slot(monkeypatch):
    # Only REQUIRED slots trigger the fallback; an omitted optional is normal.
    _patch_qwen(monkeypatch, values={"name": "Notes"})
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    values, _ = await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"name": "Notes"}
    assert llm.calls == []


# ── 2–4. Qwen timeout / exception / format error → LLM ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "qwen_error",
    [
        ExtractionUnavailable("extractor unreachable/bad reply: TimeoutError()"),
        ExtractionUnavailable("extractor unreachable/bad reply: ConnectError()"),
        ExtractionUnavailable("extractor content reply is not JSON"),
    ],
    ids=["timeout", "transport", "format"],
)
async def test_qwen_failure_falls_back_to_the_llm(monkeypatch, qwen_error):
    _patch_qwen(monkeypatch, error=qwen_error)
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    values, source = await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"name": "Notes"}
    assert source == SOURCE_QUERY
    assert len(llm.calls) == 1


# ── 5. Qwen incomplete required → LLM re-extracts the FULL set ──────────────────


@pytest.mark.asyncio
async def test_qwen_incomplete_required_triggers_full_reextraction(monkeypatch):
    # Qwen fills the optional slot only: the required slot is empty.
    _patch_qwen(monkeypatch, values={"color": "red"})
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes", "color": "red"}}')

    values, _ = await llm_ex.extract(
        llm=llm, query="make a folder called Notes, red please", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    # Both stages' values survive: the LLM's full set wins where it spoke.
    assert values == {"name": "Notes", "color": "red"}
    prompt, _kwargs = llm.calls[0]
    # FULL registry definition — not just the missing slot.
    assert "name (string, required" in prompt
    assert "color (string, optional" in prompt
    assert "<- fill this" in prompt
    # The local stage's partial values are handed to the fallback.
    assert "color: 'red'" in prompt


# ── 6. LLM success → returned for the Binder ────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_success_returns_values_for_the_binder(monkeypatch):
    _patch_qwen(monkeypatch, error=ExtractionUnavailable("timeout"))
    llm = _FakeLLM(reply='{"name": "Notes", "color": "blue"}')

    values, source = await llm_ex.extract(
        llm=llm, query="make Notes in blue", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"name": "Notes", "color": "blue"}
    assert source == SOURCE_QUERY


# ── 7–8. LLM failure / still incomplete → Agent owns the turn ───────────────────


@pytest.mark.asyncio
async def test_llm_failure_with_no_qwen_values_is_unavailable(monkeypatch):
    _patch_qwen(monkeypatch, error=ExtractionUnavailable("qwen down"))
    llm = _FakeLLM(error=RuntimeError("channel down"))

    with pytest.raises(ExtractionUnavailable) as excinfo:
        await llm_ex.extract(
            llm=llm, query="make a folder called Notes", entry=_entry(),
            model_slots=("name", "color"), bundle=_bundle(),
        )
    message = str(excinfo.value)
    assert "qwen down" in message
    assert "channel down" in message


@pytest.mark.asyncio
async def test_llm_incomplete_reply_is_returned_not_raised(monkeypatch):
    # The funnel's required check / Binder already routes an incomplete reply to
    # the Agent; the extractor must not swallow the partial truth.
    _patch_qwen(monkeypatch, error=ExtractionUnavailable("qwen down"))
    llm = _FakeLLM(reply='{"arguments": {"color": "red"}}')

    values, _ = await llm_ex.extract(
        llm=llm, query="make a folder", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"color": "red"}


@pytest.mark.asyncio
async def test_llm_format_error_returns_the_qwen_partial(monkeypatch):
    _patch_qwen(monkeypatch, values={"color": "red"})
    llm = _FakeLLM(reply="here you go: name is Notes")

    values, source = await llm_ex.extract(
        llm=llm, query="make a folder", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"color": "red"}
    assert source == SOURCE_QUERY


# ── 9. Existing LLM client/channel is reused ────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_call_reuses_the_caller_channel(monkeypatch):
    _patch_qwen(monkeypatch, error=ExtractionUnavailable("timeout"))
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name",), bundle=_bundle(),
    )

    _prompt, kwargs = llm.calls[0]
    # No client construction / channel override: only the funnel's per-call guard.
    assert set(kwargs) == {"system_prompt", "timeout"}
    assert kwargs["timeout"] == settings.llm_timeout_seconds
    assert "model" not in kwargs and "base_url" not in kwargs and "api_key" not in kwargs


@pytest.mark.asyncio
async def test_no_llm_client_keeps_the_partial_qwen_values(monkeypatch):
    _patch_qwen(monkeypatch, values={"color": "red"})

    values, _ = await llm_ex.extract(
        llm=None, query="make a folder", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
    )

    assert values == {"color": "red"}


# ── 10. Query, existing arguments and conversation context are passed ──────────


@pytest.mark.asyncio
async def test_prompt_carries_query_values_and_recent_turns(monkeypatch):
    _patch_qwen(monkeypatch, error=ExtractionUnavailable("timeout"))
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')
    bundle = _bundle(
        "call it Notes",
        history=[{"role": "user", "content": "I keep my work in folders"},
                 {"role": "assistant", "content": "ignored"},
                 {"role": "user", "content": "make a folder"}],
        source=SOURCE_TURNS,
    )

    await llm_ex.extract(
        llm=llm, query="call it Notes", entry=_entry(),
        model_slots=("name", "color"), bundle=bundle,
    )

    prompt, _kwargs = llm.calls[0]
    assert "<user_sentence>call it Notes</user_sentence>" in prompt
    assert "- I keep my work in folders" in prompt
    assert "- make a folder" in prompt
    assert "ignored" not in prompt  # only the user's own turns are sanctioned
    assert "cap-create-folder" in prompt


# ── prompt is OPTIONAL, in both directions ─────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_omitted_uses_the_frozen_system_prompts(monkeypatch):
    record: dict = {}
    _patch_qwen(monkeypatch, values={"name": "Notes"}, record=record)
    llm = _FakeLLM()

    await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name",), bundle=_bundle(),
    )

    assert record["prompt"] is None


@pytest.mark.asyncio
async def test_prompt_is_forwarded_to_both_stages(monkeypatch):
    record: dict = {}
    _patch_qwen(monkeypatch, values={"color": "red"}, record=record)
    llm = _FakeLLM(reply='{"arguments": {"name": "Notes"}}')

    await llm_ex.extract(
        llm=llm, query="make a folder called Notes", entry=_entry(),
        model_slots=("name", "color"), bundle=_bundle(),
        prompt="CAPABILITY PROMPT",
    )

    assert record["prompt"] == "CAPABILITY PROMPT"
    _prompt, kwargs = llm.calls[0]
    system_prompt = kwargs["system_prompt"]
    # The capability prompt rides with the strict extraction contract.
    assert system_prompt.startswith(llm_ex.SYSTEM_LLM_EXTRACT)
    assert "CAPABILITY PROMPT" in system_prompt

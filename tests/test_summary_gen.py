"""``summary_gen`` — the tool body behind ``cap-summary``.

The capability handler settles WHICH source and which page scope; this module pins
what the tool does with that: it READS the source first, hands the real text to
the generation pipeline, and returns the summary to the chat unless the caller
named an explicit output directory.

Lanes pinned here:

* workspace ``path`` — validated + extracted by the pipeline; no ``output_dir``
  ⇒ the rendered Markdown is the tool result and nothing is written;
* ``asset_id`` — resolved through the shared authorized reader
  (``api.tools.document_content``): drive ACL first, page-scoped extraction for
  ``pages``, never a path fabricated from the id;
* ``context`` — the conversation transcript, with no document read at all;
* every unreadable / out-of-range / image / empty / multi-source request raises a
  readable error instead of summarizing something else.
"""
from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest
from agent import Context
from agent.engine.decisions import ToolExecution
from agent.engine.runtime import ToolRuntime
from api.tools import document_content
from core.infrastructure.ingest import UnsupportedFileType
from core.infrastructure.storage import LocalStorage, object_key

from apps.api.tools.toolkit.errors import ToolKitError
from apps.api.tools.toolkit.pipeline import ToolKitPipeline
from apps.api.tools.toolkit.plugins import build_toolkit_plugin
from apps.api.tools.toolkit.sources import from_text
from tests._drive_fakes import make_drive

SUMMARY_DATA = {
    "title": "Deep Dive",
    "executive_summary": "The document explains retrieval.",
    "key_points": [{"point": "RRF blends scores.", "citations": ["[doc.md:1]"]}],
    "sections": [{"heading": "Method", "summary": "Hybrid retrieval.",
                  "citations": ["[doc.md:2]"]}],
    "qa": [{"question": "How?", "answer": "RRF.", "citations": ["[doc.md:3]"]}],
}


class _LLM:
    """Scripted generation model; every prompt is captured for content assertions."""

    def __init__(self, replies=None, *, fail_json=False, fallback="not json"):
        self.replies = list(replies or [SUMMARY_DATA])
        self.fail_json = fail_json
        self.fallback = fallback
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def complete_json(self, text, system, timeout=None):
        self.prompts.append(text)
        self.systems.append(system)
        if self.fail_json:
            raise RuntimeError("provider unavailable")
        if not self.replies:
            raise ValueError("no reply queued")
        return self.replies.pop(0)

    async def complete(self, text, system, timeout=None):
        self.prompts.append(text)
        self.systems.append(system)
        return self.fallback


@pytest.fixture(autouse=True)
def _reset_request_user():
    token = document_content.request_user.set(None)
    yield
    document_content.request_user.reset(token)


def _ctx(tmp_path) -> Context:
    ctx = Context()
    ctx.provide("storage", LocalStorage(tmp_path / "store"))
    ctx.provide("session_factory", object)     # unused: the drive is monkeypatched
    return ctx


def _runtime(tmp_path, llm, *, workspace=None) -> ToolRuntime:
    plugin = build_toolkit_plugin("summary", llm, workspace=workspace or tmp_path,
                                  ctx=_ctx(tmp_path))
    runtime = ToolRuntime()
    runtime.register(plugin.tools[0])
    return runtime


async def _call(runtime: ToolRuntime, **args):
    return await runtime.execute(ToolExecution("s1", "summary_gen", args))


def _err(res) -> str:
    """Error text of a failed tool result (the failure carries no ``value``)."""
    return str(res.error.message) + " " + " ".join(
        str(getattr(b, "text", b)) for b in res.content)


async def _asset(monkeypatch, tmp_path, name: str, data: str | bytes):
    """Store ``data`` as an asset of a fresh user; return (asset_id, user_id)."""
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    await LocalStorage(tmp_path / "store").put(object_key(sha), data)
    drive = make_drive(tmp_path / "drive")
    uid = uuid4()
    asset = await drive.assets.create(uid, name, object_sha256=sha)
    monkeypatch.setattr(document_content, "DriveService", lambda sf: drive)
    document_content.request_user.set(uid)
    return str(asset.id), uid


def _pdf_pages(texts: list[str]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for text in texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


# ── workspace-file lane ─────────────────────────────────────────────────────────


async def test_path_without_output_dir_returns_the_markdown_and_writes_nothing(tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\nRRF blends two rankings.\n",
                                     encoding="utf-8")
    llm = _LLM()
    res = await _call(_runtime(tmp_path, llm), path="doc.md")

    assert res.is_error is False
    assert "# Deep Dive" in str(res.value)
    assert "The document explains retrieval." in str(res.value)
    # The actual file content reached the generator (read first, then summarize).
    assert "RRF blends two rankings." in llm.prompts[0]
    # No artifact was written: the chat window is the destination.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc.md"]


async def test_path_with_an_explicit_output_dir_persists_the_artifact(tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\nBody.\n", encoding="utf-8")
    res = await _call(_runtime(tmp_path, _LLM()), path="doc.md",
                      output_dir="exports")

    assert res.is_error is False
    written = list((tmp_path / "exports").glob("doc_*.md"))
    assert len(written) == 1
    assert "# Deep Dive" in written[0].read_text(encoding="utf-8")
    # A persisted run reports its location, not the (possibly long) body.
    assert "exports" in str(res.value) and "Generated summary output" in str(res.value)


async def test_an_unusable_output_dir_is_an_error_not_a_chat_only_summary(tmp_path):
    # The user explicitly asked to save. A destination outside the workspace is a
    # readable error — never a silent return of the summary in chat.
    (tmp_path / "doc.md").write_text("# Title\n\nBody.\n", encoding="utf-8")
    res = await _call(_runtime(tmp_path, _LLM()), path="doc.md",
                      output_dir="../escape")

    assert res.is_error is True
    assert "escapes workspace" in _err(res)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc.md"]


# ── asset lane: ACL → storage → page-scoped extraction → summary ───────────────


async def test_asset_full_document_text_reaches_the_generator(monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "notes.txt",
                               "quarterly retrieval numbers")
    llm = _LLM()
    res = await _call(_runtime(tmp_path, llm), asset_id=asset_id)

    assert res.is_error is False
    assert "# Deep Dive" in str(res.value)
    assert "quarterly retrieval numbers" in llm.prompts[0]


async def test_asset_lane_with_an_explicit_output_dir_persists_the_artifact(
        monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "notes.txt",
                               "quarterly retrieval numbers")
    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id,
                      output_dir="exports")

    assert res.is_error is False
    written = list((tmp_path / "exports").glob("*.md"))
    assert len(written) == 1
    assert "# Deep Dive" in written[0].read_text(encoding="utf-8")
    assert "Generated summary output" in str(res.value)


async def test_asset_single_page_reads_only_that_page(monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "paper.pdf",
                               _pdf_pages(["page one text", "page two text"]))
    llm = _LLM()
    res = await _call(_runtime(tmp_path, llm), asset_id=asset_id, pages="2")

    assert res.is_error is False
    prompt = llm.prompts[0]
    assert "page two text" in prompt
    assert "page one text" not in prompt


async def test_asset_page_range_reads_exactly_that_range(monkeypatch, tmp_path):
    asset_id, _ = await _asset(
        monkeypatch, tmp_path, "paper.pdf",
        _pdf_pages(["page one text", "page two text", "page three text"]))
    llm = _LLM()
    res = await _call(_runtime(tmp_path, llm), asset_id=asset_id, pages="1-2")

    assert res.is_error is False
    prompt = llm.prompts[0]
    assert "page one text" in prompt and "page two text" in prompt
    assert "page three text" not in prompt


async def test_conversation_context_is_summarized_without_reading_a_document(
        monkeypatch, tmp_path):
    class _NoDrive:
        def __init__(self, *_a, **_kw):
            raise AssertionError("the document must not be read for a chat summary")

    monkeypatch.setattr(document_content, "DriveService", _NoDrive)
    llm = _LLM()
    res = await _call(_runtime(tmp_path, llm),
                      context="**user:** 我们聊了混合检索\n\n**assistant:** RRF 融合两路分数",
                      asset_id="an-open-document")   # injected by the Binder

    assert res.is_error is False
    assert "我们聊了混合检索" in llm.prompts[0]
    assert "an-open-document" not in llm.prompts[0]


# ── honest failures: never summarize something else ────────────────────────────


async def test_a_foreign_asset_is_reported_as_not_readable(monkeypatch, tmp_path):
    asset_id, _owner = await _asset(monkeypatch, tmp_path, "notes.txt", "secret")
    document_content.request_user.set(uuid4())    # another user's turn

    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id)
    assert res.is_error is True
    assert "not readable" in _err(res)


async def test_an_out_of_range_page_is_reported(monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "paper.pdf",
                               _pdf_pages(["only page"]))
    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id, pages="5")
    assert res.is_error is True
    assert "out of range" in _err(res)


async def test_pages_on_a_format_without_a_page_axis_is_reported(monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "notes.txt", "plain text")
    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id, pages="2")
    assert res.is_error is True
    assert "no page axis" in _err(res)


async def test_an_image_asset_points_at_the_vision_tool(monkeypatch, tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    asset_id, _ = await _asset(monkeypatch, tmp_path, "shot.png", png)
    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id)
    assert res.is_error is True
    assert "vision" in _err(res)


async def test_an_unsupported_format_is_reported(monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "legacy.ppt", b"\xd0\xcf\x11\xe0x")

    async def _unsupported(_data, _name, _llm):
        raise UnsupportedFileType("no extractor for .ppt")

    monkeypatch.setattr(document_content, "extract_document_text", _unsupported)
    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id)
    assert res.is_error is True
    assert "cannot extract text" in _err(res)


async def test_a_document_with_no_extractable_text_is_reported(monkeypatch, tmp_path):
    asset_id, _ = await _asset(monkeypatch, tmp_path, "scan.pdf", b"%PDF-1.4 empty")

    async def _empty(_data, _name, _llm):
        return "   \n"

    monkeypatch.setattr(document_content, "extract_document_text", _empty)
    res = await _call(_runtime(tmp_path, _LLM()), asset_id=asset_id)
    assert res.is_error is True
    assert "no extractable text" in _err(res)


async def test_a_call_with_no_source_is_a_readable_error(tmp_path):
    res = await _call(_runtime(tmp_path, _LLM()))
    assert res.is_error is True
    assert "exactly ONE content source" in _err(res)


@pytest.mark.parametrize("args", [
    {"paths": ["a.md"], "path": "b.md"},
    {"paths": ["a.md"], "asset_id": "asset-1"},
    {"path": "a.md", "asset_id": "asset-1"},
    {"context": "chat", "path": "a.md"},
    {"path": "a.md", "pages": "3"},
    {"context": "chat", "pages": "3"},
])
async def test_conflicting_or_misapplied_slots_are_rejected(tmp_path, args):
    res = await _call(_runtime(tmp_path, _LLM()), **args)
    assert res.is_error is True


async def test_model_failure_surfaces_as_a_tool_error(tmp_path):
    (tmp_path / "doc.md").write_text("# Title\n\nBody.\n", encoding="utf-8")
    res = await _call(_runtime(tmp_path, _LLM(fail_json=True)), path="doc.md")
    assert res.is_error is True
    assert "not valid JSON" in _err(res)


# ── pipeline seam: the two output modes ────────────────────────────────────────


async def test_run_sources_without_persist_returns_text_and_writes_nothing(tmp_path):
    result = await ToolKitPipeline(_LLM(), "summary", workspace=tmp_path).run_sources(
        [from_text("chat", "we discussed hybrid retrieval")], persist=False,
        stem="conversation")

    assert result.files == []
    assert "# Deep Dive" in result.text
    assert list(tmp_path.iterdir()) == []


async def test_run_sources_with_output_dir_persists_and_keeps_the_text(tmp_path):
    pipe = ToolKitPipeline(_LLM(), "summary", workspace=tmp_path)
    result = await pipe.run_sources([from_text("chat", "we discussed retrieval")],
                                    output_dir="out", persist=True, stem="chat")

    assert len(result.files) == 1
    assert result.files[0].startswith(str(tmp_path / "out"))
    assert "# Deep Dive" in result.text


async def test_run_sources_requires_a_source(tmp_path):
    with pytest.raises(ToolKitError, match="no content"):
        await ToolKitPipeline(_LLM(), "summary", workspace=tmp_path).run_sources([])

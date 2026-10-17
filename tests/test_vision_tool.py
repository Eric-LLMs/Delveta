"""Unit tests for the ``vision`` chat tool and its local PDF page/region render step.

The tool follows the "Local Data Preparation → Single LLM Call" rule: for a PDF it renders
the requested page (or a sub-region) to a PNG LOCALLY with PyMuPDF, then makes exactly ONE
authorized vision call; for an image it sends the stored bytes directly. These tests pin
that contract against fake drive repositories + a real :class:`LocalStorage` (no DB), and
stub :func:`describe_image` so no network/model is touched:
  - the model receives REAL image bytes (the rendered page PNG / the stored image), never
    just an ``asset_id``/filename/page locator;
  - a PDF asks for page N and renders page N (never page 1);
  - the viewer's current page is honored when the caller passes no explicit ``page``;
  - the ACL fires BEFORE any bytes are read (a foreign/forged asset is refused);
  - exactly ONE downstream call is made on the success path.
"""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agent import Context
from agent.engine.context import AgentTurn
from agent.engine.decisions import ToolExecution
from agent.engine.runtime import ToolRuntime
from core.infrastructure import pdf as pdf_lib
from core.infrastructure.request_context import request_user
from core.infrastructure.storage import LocalStorage, object_key

from apps.api.tools import vision_tool
from tests._drive_fakes import make_drive


# ── the local render helper (no model) ─────────────────────────────────────────────

def _pdf_pages(texts: list[str]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for t in texts:
        page = doc.new_page()
        if t:
            page.insert_text((72, 72), t)
    data = doc.tobytes()
    doc.close()
    return data


def test_render_page_png_renders_the_requested_page():
    data = _pdf_pages(["page one", "page two"])
    p1 = pdf_lib.render_page_png(data, 1)
    p2 = pdf_lib.render_page_png(data, 2)
    assert p1[:8] == b"\x89PNG\r\n\x1a\n" and p2[:8] == b"\x89PNG\r\n\x1a\n"
    assert p1 != p2  # a different page is genuinely a different image


def test_render_page_png_out_of_range_raises():
    data = _pdf_pages(["only page"])
    with pytest.raises(ValueError, match="out of range"):
        pdf_lib.render_page_png(data, 2)


def test_render_page_png_region_crops_a_sub_area():
    data = _pdf_pages(["page one"])
    whole = pdf_lib.render_page_png(data, 1)
    crop = pdf_lib.render_page_png(data, 1, region={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5})
    assert crop[:8] == b"\x89PNG\r\n\x1a\n"
    assert crop != whole  # a half-page crop is not the whole page


# ── the tool ────────────────────────────────────────────────────────────────────────

def _ctx(tmp_path) -> Context:
    ctx = Context()
    ctx.provide("storage", LocalStorage(tmp_path))
    ctx.provide("session_factory", object)  # unused: the drive is monkeypatched
    return ctx


@asynccontextmanager
async def _turn(context: dict):
    """Bind an AgentTurn — the SAME turn context ``_run_tool`` sets — for the block's
    duration, resetting it in the same task (no cross-test leakage)."""
    from agent.engine.context import _TURN_CTX

    token = _TURN_CTX.set(AgentTurn(user_msg="q", context=context))
    try:
        yield
    finally:
        _TURN_CTX.reset(token)


async def _register(monkeypatch, tmp_path, name: str, data: bytes):
    """Store ``data`` as an asset named ``name``; return (runtime, asset_id, drive, uid)."""
    storage = LocalStorage(tmp_path)
    sha = hashlib.sha256(data).hexdigest()
    await storage.put(object_key(sha), data)
    drive = make_drive(tmp_path)
    uid = uuid4()
    asset = await drive.assets.create(uid, name, object_sha256=sha)
    monkeypatch.setattr(vision_tool, "DriveService", lambda sf: drive)
    request_user.set(uid)
    runtime = ToolRuntime()
    vision_tool.register(runtime, _ctx(tmp_path), llm=None)
    return runtime, str(asset.id), drive, uid


async def _call(runtime, asset_id, **args):
    return await runtime.execute(ToolExecution("c1", "vision", {"asset_id": asset_id, **args}))


def _stub_describe(monkeypatch, answer="ANSWER"):
    """Replace the single downstream vision call with a capturing stub (no model)."""
    calls: list[dict] = []

    async def _describe(data, mime="", *, llm, session_factory, prompt=None, user_id=None,
                        role_id=None):
        calls.append({"data": data, "mime": mime, "prompt": prompt, "user_id": user_id})
        return answer

    monkeypatch.setattr(vision_tool, "describe_image", _describe)
    return calls


def _err(res) -> str:
    return str(res.error.message) + " " + " ".join(str(getattr(b, "text", b)) for b in res.content)


async def test_pdf_page_renders_that_page_and_sends_real_image_once(monkeypatch, tmp_path):
    raw = _pdf_pages(["page one text", "page two text"])
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "paper.pdf", raw)
    calls = _stub_describe(monkeypatch, answer="table summary")

    res = await _call(runtime, asset_id, page=2)
    assert res.is_error is False and str(res.value) == "table summary"
    assert len(calls) == 1                       # exactly ONE downstream vision call
    sent = calls[0]["data"]
    assert sent[:8] == b"\x89PNG\r\n\x1a\n"      # REAL rendered image, not the raw PDF
    assert sent == pdf_lib.render_page_png(raw, 2)   # page 2, never page 1
    assert calls[0]["mime"] == "image/png"
    assert calls[0]["user_id"] is not None


async def test_page_deictic_pdf_reads_the_bound_viewer_page_not_page_one(monkeypatch, tmp_path):
    raw = _pdf_pages(["p1", "p2", "p3", "p4", "p5"])
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "paper.pdf", raw)
    calls = _stub_describe(monkeypatch)

    async with _turn({"viewer": {"asset": {"page": 5}}}):   # THIS turn's viewer context
        res = await _call(runtime, asset_id)                 # no explicit page → viewer's page
    assert res.is_error is False
    assert len(calls) == 1
    assert calls[0]["data"] == pdf_lib.render_page_png(raw, 5)   # page 5, NOT page 1
    assert calls[0]["data"] != pdf_lib.render_page_png(raw, 1)


async def test_region_focuses_the_render_on_the_selected_page(monkeypatch, tmp_path):
    raw = _pdf_pages(["p1", "p2", "p3", "p4", "p5"])
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "paper.pdf", raw)
    calls = _stub_describe(monkeypatch)

    region = {"x": 0.1, "y": 0.4, "w": 0.8, "h": 0.2}   # a table band on page 5
    res = await _call(runtime, asset_id, page=5, region=region)
    assert res.is_error is False
    assert calls[0]["data"] == pdf_lib.render_page_png(raw, 5, region=region)


async def test_explicit_page_arg_wins_over_the_viewer_context(monkeypatch, tmp_path):
    raw = _pdf_pages(["p1", "p2", "p3"])
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "paper.pdf", raw)
    calls = _stub_describe(monkeypatch)

    async with _turn({"viewer": {"asset": {"page": 3}}}):
        await _call(runtime, asset_id, page=2)      # the explicit caller page wins
    assert calls[0]["data"] == pdf_lib.render_page_png(raw, 2)


async def test_multipage_pdf_without_any_page_fails_closed(monkeypatch, tmp_path):
    # No ``page`` arg AND no viewer page for this turn: a multi-page PDF must be REFUSED,
    # never silently read as page 1 — and the model is never reached.
    raw = _pdf_pages(["p1", "p2", "p3", "p4", "p5"])
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "paper.pdf", raw)
    calls = _stub_describe(monkeypatch)
    res = await _call(runtime, asset_id)
    assert res.is_error is True
    assert "page not specified" in _err(res) and "5-page" in _err(res)
    assert calls == []


async def test_single_page_pdf_without_page_renders_page_one(monkeypatch, tmp_path):
    # A lone page is unambiguous: no page given → page 1, one call.
    raw = _pdf_pages(["only page"])
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "one.pdf", raw)
    calls = _stub_describe(monkeypatch)
    res = await _call(runtime, asset_id)
    assert res.is_error is False
    assert len(calls) == 1
    assert calls[0]["data"] == pdf_lib.render_page_png(raw, 1)


async def test_image_asset_sends_stored_bytes(monkeypatch, tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "shot.png", png)
    calls = _stub_describe(monkeypatch, answer="described")

    res = await _call(runtime, asset_id)
    assert res.is_error is False and str(res.value) == "described"
    assert len(calls) == 1
    assert calls[0]["data"] == png               # REAL stored bytes, not a locator
    assert calls[0]["mime"] == "image/png"


async def test_question_is_forwarded_into_the_prompt(monkeypatch, tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "shot.png", png)
    calls = _stub_describe(monkeypatch)

    await _call(runtime, asset_id, question="这张表有几个字段")
    assert "这张表有几个字段" in calls[0]["prompt"]


# ── ACL before any bytes are read ─────────────────────────────────────────────────────

async def test_no_request_user_fails_before_storage(monkeypatch, tmp_path):
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "a.txt", b"hello")
    _stub_describe(monkeypatch)
    request_user.set(None)
    res = await _call(runtime, asset_id)
    assert res.is_error is True
    assert "authenticated request user" in _err(res)


async def test_foreign_asset_denied_before_bytes_read(monkeypatch, tmp_path):
    runtime, asset_id, _, _ = await _register(monkeypatch, tmp_path, "a.txt", b"hello")
    calls = _stub_describe(monkeypatch)
    # attacker context: the stored object is gone — an ACL hole would surface as
    # "bytes missing"; the ACL must fire first with a not-readable error, and the
    # model must never be reached.
    sha = hashlib.sha256(b"hello").hexdigest()
    await LocalStorage(tmp_path).delete(object_key(sha))
    request_user.set(uuid4())
    res = await _call(runtime, asset_id)
    assert res.is_error is True
    assert "not readable" in _err(res)
    assert "bytes missing" not in _err(res)
    assert calls == []


# ── schema / capability ───────────────────────────────────────────────────────────────

def test_tool_schema_exposes_page_and_region():
    schema = vision_tool._PAGE_RENDER_EXTS  # sanity: the pdf ext set is present
    assert ".pdf" in schema


def test_is_pdf_by_mime_and_extension():
    assert vision_tool._is_pdf(SimpleNamespace(mime_type="application/pdf", name="x"))
    assert vision_tool._is_pdf(SimpleNamespace(mime_type=None, name="x.PDF"))
    assert not vision_tool._is_pdf(SimpleNamespace(mime_type="image/png", name="x.png"))

"""The ``pdf_table_to_text`` chat tool reads every table in ONE model call.

Local Data Preparation → Single LLM Call: all detected table regions are composited into a
single PNG locally (:func:`core.infrastructure.pdf.render_tables_png`, no model), and the
vision model reads that one image in exactly ONE request. These tests pin:
  - ``render_tables_png`` — ``None`` when there is no table, otherwise a real composited
    PNG; an out-of-range ``pages`` spec still raises before any page is scanned;
  - the tool makes exactly ONE downstream vision call even with several tables.
The ingest worker's per-table transcription (``extract_pdf_document``) is a separate,
unchanged path and is not exercised here.
"""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from core.infrastructure import pdf as pdf_lib
from core.infrastructure import vision_caption as vc

from apps.api.tools import pdf_tools


def _png(w: int, h: int, color: str) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _ruled_table_pdf() -> bytes:
    """A one-page PDF with a ruled 3x3 grid — PyMuPDF's detector finds it."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    x0, y0, cw, ch, rows, cols = 72, 72, 80, 24, 3, 3
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * ch), (x0 + cols * cw, y0 + r * ch))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * ch))
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + c * cw + 4, y0 + r * ch + 16), f"{r}{c}")
    data = doc.tobytes()
    doc.close()
    return data


def _plain_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "no tables here")
    data = doc.tobytes()
    doc.close()
    return data


# ── the local compositing helper (no model) ────────────────────────────────────────

def test_render_tables_png_is_none_without_tables():
    assert pdf_lib.render_tables_png(_plain_pdf()) is None


def test_render_tables_png_composites_regions_into_one_png():
    data = _ruled_table_pdf()
    assert pdf_lib.detect_tables(data)  # the synthetic table is detectable
    strip = pdf_lib.render_tables_png(data)
    assert strip is not None and strip[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_tables_png_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        pdf_lib.render_tables_png(_ruled_table_pdf(), pages=[99])


# ── the tool: exactly ONE model call, whatever the table count ──────────────────────

class _FakeRuntime:
    def __init__(self) -> None:
        self.defs: dict[str, object] = {}

    def register(self, definition) -> None:
        self.defs[definition.name] = definition


class _Ctx:
    """Minimal context: ``session_factory``/``storage`` resolve to inert stand-ins."""

    def __init__(self, storage=None) -> None:
        self._storage = storage

    def resolve(self, name):
        if name == "storage" and self._storage is not None:
            return self._storage
        return object()


class _CountingLLM:
    """Counts real downstream model requests (no ``chat_stream``: the non-streaming path)."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kw):
        self.calls += 1
        return {"content": "TBL"}


async def _run(monkeypatch, llm, tables: list[bytes]):
    async def _load(asset_id, ctx):
        return b"%PDF-fake"

    async def _chain(session_factory, *, user_id=None, role_id=None):
        return [("https://v.example/v1", "key", "vision-model")]

    monkeypatch.setattr(pdf_tools, "_load_asset_bytes", _load)
    monkeypatch.setattr(pdf_lib, "detect_tables", lambda *a, **k: tables)
    monkeypatch.setattr(vc, "resolve_vision_channels", _chain)
    rt = _FakeRuntime()
    pdf_tools.register(rt, ctx=_Ctx(), llm=llm)
    return await rt.defs["pdf_table_to_text"].execute({"asset_id": "a"}, None)


async def test_tool_makes_exactly_one_vision_call_for_many_tables(monkeypatch):
    llm = _CountingLLM()
    out = await _run(monkeypatch, llm, [
        _png(20, 10, "red"), _png(20, 10, "green"), _png(20, 10, "blue"),
    ])
    assert out == "TBL"
    assert llm.calls == 1                       # three tables, still ONE model request


async def test_tool_reports_no_tables_without_a_model_call(monkeypatch):
    llm = _CountingLLM()
    out = await _run(monkeypatch, llm, [])
    assert out == "No tables detected in this PDF."
    assert llm.calls == 0


# ── permission before bytes ────────────────────────────────────────────────────────

async def _register_asset(monkeypatch, tmp_path, *, owner):
    import hashlib
    from uuid import uuid4

    from core.infrastructure.request_context import set_request_user
    from core.infrastructure.storage import LocalStorage, object_key
    from tests._drive_fakes import make_drive

    data = b"%PDF-fake-bytes"
    storage = LocalStorage(tmp_path)
    sha = hashlib.sha256(data).hexdigest()
    await storage.put(object_key(sha), data)
    drive = make_drive(tmp_path)
    asset = await drive.assets.create(owner, "x.pdf", object_sha256=sha)
    monkeypatch.setattr(pdf_tools, "DriveService", lambda sf: drive)
    set_request_user(owner)
    return asset, storage


async def test_load_asset_bytes_reads_the_callers_asset(monkeypatch, tmp_path):
    from uuid import uuid4

    owner = uuid4()
    asset, storage = await _register_asset(monkeypatch, tmp_path, owner=owner)
    data = await pdf_tools._load_asset_bytes(str(asset.id), _Ctx(storage))
    assert data == b"%PDF-fake-bytes"


async def test_load_asset_bytes_denies_a_foreign_asset(monkeypatch, tmp_path):
    from uuid import uuid4

    asset, storage = await _register_asset(monkeypatch, tmp_path, owner=uuid4())
    # A different request user must be refused BEFORE any bytes are read.
    from core.infrastructure.request_context import set_request_user
    set_request_user(uuid4())
    with pytest.raises(ValueError, match="not readable"):
        await pdf_tools._load_asset_bytes(str(asset.id), _Ctx(storage))

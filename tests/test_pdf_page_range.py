"""PDF page RANGE — the bottom layer (``pdf.py``) + tool layer (``pdf_tools.py``).

Phase 2-B pins the three legal states of a page window end to end:

* omitted (``pages=None``)        -> the whole document;
* a complete spec (``"3"``/``"1-3"``) -> exactly those pages, nothing else;
* an out-of-range / malformed spec -> a hard error that BUBBLES out of the
  tool (never a silent widening back to the whole document).

The window is 1-based and the page markers carry the REAL page number, so a
windowed read can never be mistaken for a whole-document read downstream.
"""
from __future__ import annotations

import pymupdf
import pytest
from core.infrastructure import pdf as pdf_lib

from apps.api.tools import pdf_tools


def _grid(page, x0, y0, cw, ch, rows, cols):
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * ch), (x0 + cols * cw, y0 + r * ch))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * ch))


def _two_page_pdf() -> bytes:
    """Two pages, each carrying a text marker and one detectable table."""
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    p1, p2 = doc[0], doc[1]  # pages must be (re)fetched after the second new_page()
    p1.insert_text((72, 72), "ALPHA")
    p2.insert_text((72, 72), "BETA")
    _grid(p1, 72, 120, 80, 24, 2, 2)
    p1.insert_text((76, 136), "X1")
    _grid(p2, 72, 120, 80, 24, 2, 2)
    p2.insert_text((76, 136), "A1")
    data = doc.tobytes()
    doc.close()
    return data


# ── pdf.py: extract_pdf_text ──────────────────────────────────────────────────

def test_extract_pdf_text_default_reads_every_page():
    text = pdf_lib.extract_pdf_text(_two_page_pdf())
    assert "ALPHA" in text and "BETA" in text


def test_extract_pdf_text_single_page_window_excludes_the_rest():
    text = pdf_lib.extract_pdf_text(_two_page_pdf(), pages=[2])
    assert "BETA" in text and "ALPHA" not in text


def test_extract_pdf_text_complete_range_preserves_order():
    text = pdf_lib.extract_pdf_text(_two_page_pdf(), pages=[1, 2])
    assert text.index("ALPHA") < text.index("BETA")


def test_extract_pdf_text_page_markers_use_the_real_page_number():
    text = pdf_lib.extract_pdf_text(_two_page_pdf(), page_markers=True, pages=[2])
    assert "[[PAGE:2]]" in text and "[[PAGE:1]]" not in text


@pytest.mark.parametrize("bad", [[99], [0], [-1], [1, 99]])
def test_extract_pdf_text_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="out of range"):
        pdf_lib.extract_pdf_text(_two_page_pdf(), pages=bad)


# ── pdf.py: detect_tables ─────────────────────────────────────────────────────

def test_detect_tables_default_reads_every_page():
    assert len(pdf_lib.detect_tables(_two_page_pdf())) == 2


def test_detect_tables_single_page_window():
    data = _two_page_pdf()
    assert len(pdf_lib.detect_tables(data, pages=[1])) == 1
    assert len(pdf_lib.detect_tables(data, pages=[2])) == 1
    assert len(pdf_lib.detect_tables(data, pages=[1, 2])) == 2


@pytest.mark.parametrize("bad", [[99], [0]])
def test_detect_tables_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="out of range"):
        pdf_lib.detect_tables(_two_page_pdf(), pages=bad)


# ── pdf_tools.py: the ``pages`` arg window ────────────────────────────────────

class _FakeRuntime:
    def __init__(self):
        self.defs: dict[str, object] = {}

    def register(self, definition):
        self.defs[definition.name] = definition


class _FakeLLM:
    async def chat(self, messages):
        return {"content": "TABLE::" + str(len(messages))}


@pytest.fixture
def pdf_tools_with(monkeypatch):
    """Register both PDF tools with the asset-bytes fetch stubbed to ``data``."""
    def _build(data: bytes) -> dict:
        async def _fake_load(asset_id, ctx):
            return data

        monkeypatch.setattr(pdf_tools, "_load_asset_bytes", _fake_load)
        rt = _FakeRuntime()
        pdf_tools.register(rt, ctx=None, llm=_FakeLLM())
        return rt.defs

    return _build


def test_window_helper_states():
    assert pdf_tools._window({}) is None
    assert pdf_tools._window({"pages": "3"}) == [3]
    assert pdf_tools._window({"pages": "1-3"}) == [1, 2, 3]


@pytest.mark.parametrize("bad", ["", "abc", "1-", "5-2", "1,,3", 5, ["1"]])
def test_window_helper_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        pdf_tools._window({"pages": bad})


async def test_tool_extract_text_omitted_pages_reads_whole_document(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    out = await defs["pdf_extract_text"].execute({"asset_id": "a"}, None)
    assert "ALPHA" in out and "BETA" in out


async def test_tool_extract_text_single_page_window(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    out = await defs["pdf_extract_text"].execute({"asset_id": "a", "pages": "2"}, None)
    assert "BETA" in out and "ALPHA" not in out


async def test_tool_extract_text_out_of_range_fails_closed(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    with pytest.raises(ValueError, match="out of range"):
        await defs["pdf_extract_text"].execute({"asset_id": "a", "pages": "99"}, None)


async def test_tool_extract_text_malformed_spec_never_widens_to_whole_doc(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    with pytest.raises(ValueError):
        await defs["pdf_extract_text"].execute(
            {"asset_id": "a", "pages": "not-a-page"}, None
        )


async def test_tool_table_to_text_window_excludes_other_page(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    out = await defs["pdf_table_to_text"].execute({"asset_id": "a", "pages": "2"}, None)
    # exactly the page-2 table was transcribed — a broken window would transcribe 2.
    assert out.count("TABLE::") == 1


async def test_tool_table_to_text_out_of_range_fails_closed(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    with pytest.raises(ValueError, match="out of range"):
        await defs["pdf_table_to_text"].execute({"asset_id": "a", "pages": "99"}, None)


def test_both_tools_declare_pages_as_optional_string(pdf_tools_with):
    defs = pdf_tools_with(_two_page_pdf())
    for name in ("pdf_extract_text", "pdf_table_to_text"):
        props = defs[name].parameters["properties"]
        assert props["pages"]["type"] == "string"
        assert defs[name].parameters["required"] == ["asset_id"]

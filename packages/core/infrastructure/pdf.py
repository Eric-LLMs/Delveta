"""PDF text extraction: body text via PyMuPDF + table detection → image → vision LLM.

The strategy mirrors the LLMs-Lab/RAG reference (table → image → read → text) but drops
the torch/Table-Transformer detector: PyMuPDF's built-in ``page.find_tables()`` locates
table bounding boxes with no ML stack, and Delveta's existing chat LLM (default
``gpt-4o-mini``) reads the rendered table image via an OpenAI ``image_url`` content part.

Degradation: a single table transcription failure logs and skips that table — the ingest
job must never fail because one table could not be read (same philosophy as
``contextualize_chunks``). If the pinned upstream model is not vision-capable, every
table falls back to the body text alone and the file still indexes.
"""
from __future__ import annotations

import asyncio
import base64
import logging

import pymupdf  # PyMuPDF (the ``fitz`` name is a deprecated alias)

log = logging.getLogger(__name__)

MAX_TABLES = 20


def _selected_pages(page_count: int, pages: list[int] | None) -> list[int]:
    """Resolve a 1-based page selection against ``page_count``.

    ``pages=None`` means the whole document (every page, in order). A given list is used
    as-is after bounds validation: any 1-based page outside ``1..page_count`` is a hard
    :class:`ValueError` — a partial or out-of-range selection must never silently widen
    back to the whole document (fail-closed, never a quiet full-document fallback).
    """
    if pages is None:
        return list(range(1, page_count + 1))
    out_of_range = [p for p in pages if p < 1 or p > page_count]
    if out_of_range:
        raise ValueError(
            f"page(s) {out_of_range} out of range: this PDF has {page_count} page(s)"
        )
    return list(pages)


def extract_pdf_text(
    content: bytes, *, page_markers: bool = False, pages: list[int] | None = None
) -> str:
    """Return the plain-text body of a PDF, pages newline-joined.

    With ``page_markers=True`` each page is prefixed with a ``[[PAGE:n]]`` sentinel so the
    RAG chunker can later map a chunk back to the page(s) it covers. Markers are stripped
    before a chunk is stored (see ``build_chunks(on_split=...)``); the sentinel text is
    unlikely to collide with real content and survives whitespace collapsing.

    ``pages`` (1-based) restricts the read to a window of pages; ``None`` is the whole
    document. An out-of-range page raises (never a silent full-document fallback), and the
    sentinel carries the page's REAL number, not its position in the window.
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    try:
        selected = _selected_pages(doc.page_count, pages)
        if page_markers:
            return "".join(f"\n[[PAGE:{p}]]\n{doc[p - 1].get_text('text')}" for p in selected)
        return "\n".join(doc[p - 1].get_text("text") for p in selected)
    finally:
        doc.close()


def detect_tables(
    content: bytes, max_tables: int = MAX_TABLES, pages: list[int] | None = None
) -> list[bytes]:
    """Render each detected table region to a PNG (bytes) for the vision LLM.

    A page whose table detection raises is skipped entirely (degrade, never crash); the
    total is capped at ``max_tables`` so a pathological PDF cannot fan out too many vision
    calls. ``pages`` (1-based) restricts detection to a window of pages; ``None`` is the
    whole document, and an out-of-range page raises before any page is scanned.
    """
    doc = pymupdf.open(stream=content, filetype="pdf")
    try:
        images: list[bytes] = []
        for p in _selected_pages(doc.page_count, pages):
            page = doc[p - 1]
            try:
                tables = page.find_tables()
            except Exception:  # noqa: BLE001 - detection hiccup on one page
                continue
            for table in tables.tables:
                if len(images) >= max_tables:
                    return images
                pix = page.get_pixmap(clip=table.bbox)
                images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


async def pdf_table_to_text(png: bytes, llm, prompt: str | None = None) -> str:
    """Ask the vision LLM to transcribe one table image into text."""
    prompt = prompt or (
        "Transcribe the table in this image to text. Preserve rows and columns, keep "
        "numbers and values exact, and output only the transcribed text."
    )
    resp = await llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(png)}},
                ],
            }
        ]
    )
    return (resp.get("content") or "").strip()


async def extract_pdf_document(
    content: bytes,
    llm,
    *,
    max_concurrency: int = 4,
    max_tables: int = MAX_TABLES,
    page_markers: bool = False,
) -> str:
    """Body text + transcribed tables, joined into one document text for chunking.

    With ``page_markers=True`` each page's text is preceded by a ``[[PAGE:n]]`` sentinel so
    chunks can be attributed to the page(s) they cover (tables stay appended at the end
    without a marker and fall back to the running page).
    """
    body = extract_pdf_text(content, page_markers=page_markers)
    tables = detect_tables(content, max_tables=max_tables)
    if not tables:
        return body

    sem = asyncio.Semaphore(max_concurrency)

    async def one(png: bytes) -> str:
        async with sem:
            try:
                return await pdf_table_to_text(png, llm)
            except Exception as exc:  # noqa: BLE001 - skip a table, never fail ingest
                log.warning("pdf table transcription failed, skipped: %s", exc)
                return ""

    transcribed = await asyncio.gather(*(one(p) for p in tables))
    parts = [body] + [t for t in transcribed if t]
    return "\n\n".join(parts)

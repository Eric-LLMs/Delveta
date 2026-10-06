"""``pdf_extract_text`` / ``pdf_table_to_text``: pull searchable text out of drive PDFs.

Body text comes from PyMuPDF; tables are located by ``page.find_tables()``, rendered to
images, and transcribed by the vision LLM — the table → image → read → text strategy from
the LLMs-Lab/RAG reference, minus the torch table-transformer. Both tools read the asset's
stored bytes via the shared storage + session_factory provided in ``deps.py``, so an agent
can inspect a drive PDF that the ingest worker would otherwise handle automatically.
"""
from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block
from core.infrastructure import pdf as pdf_lib
from core.infrastructure.drive_repositories import SqlAssetRepository
from core.infrastructure.pdf_pages import parse_pages_spec
from core.infrastructure.storage import object_key

log = logging.getLogger(__name__)

_TABLE_PROMPT = (
    "Transcribe the table in this image to text. Preserve rows and columns, keep numbers "
    "and values exact, and output only the transcribed text."
)

_PAGES_PARAM = {
    "type": "string",
    "description": "Page specification (e.g. '3', '1-3') from viewer context. "
    "Omit to process the whole document.",
}


def _window(args: dict) -> list[int] | None:
    """Parse the optional ``pages`` arg into a 1-based page list.

    Absent/None -> ``None`` (the whole document). A present spec is parsed strictly: a
    malformed or empty spec raises (fail-closed) — the call must never silently widen a
    page-scoped request back to the whole document.
    """
    raw = args.get("pages")
    if raw is not None and not isinstance(raw, str):
        raise ValueError('pages must be a string like "3" or "1-3"')
    return parse_pages_spec(raw) if raw is not None else None


async def _load_asset_bytes(asset_id: str, ctx: Context) -> bytes:
    """Fetch a drive asset's stored object bytes by ``asset_id``."""
    repo = SqlAssetRepository(ctx.resolve("session_factory"))
    asset = await repo.get(UUID(asset_id))
    if asset is None or not asset.object_sha256:
        raise ValueError(f"asset {asset_id} not found or has no stored object")
    data = await ctx.resolve("storage").get(object_key(asset.object_sha256))
    if data is None:
        raise ValueError(f"object bytes missing for asset {asset_id}")
    return data


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def pdf_extract_text(args: dict, exec: ToolExecution) -> str:
        pages = _window(args)
        data = await _load_asset_bytes(args["asset_id"], ctx)
        # fitz is CPU-bound; run it off the event loop.
        return await asyncio.to_thread(pdf_lib.extract_pdf_text, data, pages=pages)

    async def pdf_table_to_text(args: dict, exec: ToolExecution) -> str:
        pages = _window(args)
        data = await _load_asset_bytes(args["asset_id"], ctx)
        tables = await asyncio.to_thread(pdf_lib.detect_tables, data, pages=pages)
        if not tables:
            return "No tables detected in this PDF."

        sem = asyncio.Semaphore(4)

        async def one(png: bytes) -> str:
            async with sem:
                try:
                    return await pdf_lib.pdf_table_to_text(png, llm, _TABLE_PROMPT)
                except Exception as exc:  # noqa: BLE001 - skip a table, never error out
                    log.warning("pdf table transcription failed, skipped: %s", exc)
                    return ""

        transcribed = await asyncio.gather(*(one(p) for p in tables))
        text = "\n\n".join(t for t in transcribed if t)
        return text or "Table transcription returned no text."

    runtime.register(
        define_tool(
            name="pdf_extract_text",
            description="Extract the plain-text body of a PDF from the cloud drive. "
            "Returns the page text (tables excluded; use pdf_table_to_text for those).",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Cloud-drive asset id of the PDF.",
                    },
                    "pages": _PAGES_PARAM,
                },
                "required": ["asset_id"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=pdf_extract_text,
        )
    )

    runtime.register(
        define_tool(
            name="pdf_table_to_text",
            description="Detect tables in a cloud-drive PDF, render each table to an image, "
            "and transcribe it to text with a vision LLM. Returns the table text.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Cloud-drive asset id of the PDF.",
                    },
                    "pages": _PAGES_PARAM,
                },
                "required": ["asset_id"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=pdf_table_to_text,
        )
    )

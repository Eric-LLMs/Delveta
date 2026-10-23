"""``read_document``: extract the text content of a document the user attached to the chat
or has open in the viewer.

Attachments ride on the chat payload as a drive ``asset_id``; the agent only sees the
``[Attached: <name> (asset_id <id>)]`` note or the trusted Viewer Access Context stub.
Without a reader it can guess but never open
the file, so every "parse this PDF / summarize this Word doc" request fails. This tool
closes that gap: it loads the asset bytes from storage and runs the same extractor the
ingest worker uses — PDF (PyMuPDF body text, tables via the vision LLM), .docx
(python-docx), legacy .doc (antiword), Excel (.xlsx via openpyxl), PowerPoint
(.pptx/.potx/.ppsx slide text + speaker notes via python-pptx), plus plain text /
markdown / csv / json and subtitles. Images are routed to the ``vision`` tool instead,
and the output is capped so one huge document cannot flood the agent's context window.
With ``pages`` the read is page/slide-scoped: PyMuPDF/python-pptx touch ONLY the requested
pages (no whole-document pass, no table/vision work on unrequested pages), out-of-range
specs fail the whole call before extraction, formats without a page axis are rejected, and
the output cap applies once to the merged multi-page result.
Legacy .doc additionally gets an embedded-image recovery pass: antiword can only leave a
``[pic]`` placeholder, so ``scan_doc_images`` recovers the actual inline pictures, saves
them as derived drive assets (same ``RAG 图片/<doc>`` folder and dedupe the ingest worker
uses), and lists their asset_ids in the tool output so the agent can feed them to
``vision``.

The authorized read itself lives in :mod:`api.tools.document_content` — ``summary_gen``
reads through the same module, so the ACL / storage / extraction rules have one owner.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block
from api.tools.document_content import UnsupportedDocument, read_asset_document
from core.application.drive_service import DriveService
from core.infrastructure.doc_images import scan_doc_images
from core.infrastructure.pdf_pages import parse_pages_spec

log = logging.getLogger(__name__)

# One tool result must not blow up the agent's context window; the tail is dropped with a
# truncation note so the agent can tell the user it saw only the first part. It caps the
# FINAL MERGED output — page-scoped multi-page results truncate once, never per page.
MAX_OUTPUT_CHARS = 20_000

# Characters that would turn a doc title into nested folders / broken names.
_FOLDER_BAD = re.compile(r'[\\/:*?"<>|]')


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        total = len(text)
        return f"{text[:MAX_OUTPUT_CHARS]}\n\n[...truncated; {total} chars total]"
    return text


async def _mint_doc_images(data: bytes, asset, ctx: Context) -> str:
    """Save a legacy .doc's inline pictures as drive assets; return a footer listing ids.

    Dedupes on ``(source_asset_id, sha256)`` — an image already extracted by the RAG
    worker (or a previous chat turn) resolves to the existing asset instead of a
    duplicate row. Any failure degrades to the plain text result: minting must never
    break ``read_document``.
    """
    try:
        images, skipped = await asyncio.to_thread(scan_doc_images, data)
        if not images:
            if skipped:
                return (
                    f"\n\n[{skipped} embedded metafile image(s) could not be rendered "
                    "on this server; the [pic] placeholder above is all there is.]"
                )
            return ""
        stem = _FOLDER_BAD.sub("_", Path(asset.name or "doc").stem).strip(". ") or "doc"
        folder = f"RAG 图片/{stem}"
        drive = DriveService(ctx.resolve("session_factory"))
        ids: list[str] = []
        for img in images:
            digest = hashlib.sha256(img["data"]).hexdigest()
            existing = await drive.assets.get_by_source_content(asset.id, digest)
            if existing is not None:
                ids.append(str(existing.id))
                continue
            saved = await drive.save_artifact(
                asset.user_id,
                img["name"],
                img["mime"],
                img["data"],
                folder_path=folder,
                workspace_id=asset.workspace_id,
                source_asset_id=asset.id,
            )
            ids.append(str(saved.id))
        lines = "\n".join(
            f"- doc_image {i + 1}: asset_id {aid}" for i, aid in enumerate(ids)
        )
        extra = f" ({skipped} metafile image(s) not renderable)" if skipped else ""
        return (
            f"\n\n[This .doc contains {len(ids)} embedded image(s){extra}. They are saved "
            "in the cloud drive; call the `vision` tool with an asset_id below to analyze "
            f"the picture or table it shows:\n{lines}]"
        )
    except Exception as exc:  # noqa: BLE001 — image recovery is best-effort
        log.warning("doc image minting failed for asset %s: %s", asset.id, exc)
        return ""


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def read_document(args: dict, exec: ToolExecution) -> str:
        asset_id = args["asset_id"]
        # ── pages tri-state BEFORE any I/O: omitted/None → legacy full read; empty string
        # → hard error (must never be mistaken for "read everything"); non-empty → parse.
        pages_raw = args.get("pages")
        if pages_raw is not None and not isinstance(pages_raw, str):
            raise ValueError('pages must be a string like "3" or "1-3"')
        pages = parse_pages_spec(pages_raw) if pages_raw is not None else None

        try:
            doc = await read_asset_document(ctx, llm, asset_id, pages=pages,
                                            caller="read_document")
        except UnsupportedDocument as exc:
            return (
                f"Cannot extract text from '{exc.name}': {exc.reason} — you must NOT "
                "substitute another document from the conversation for this one; tell "
                "the user this file could not be read."
            )
        if doc.is_image:
            return (
                f"'{doc.name}' is an image. Call the `vision` tool with asset_id {asset_id} "
                "to analyze its visual content."
            )

        if pages is not None:
            text = _truncate(doc.text)
            return f"'{doc.name}' — pages {','.join(str(p) for p in pages)}:\n\n" + text

        text = doc.text
        footer = ""
        if doc.ext == ".doc":
            footer = await _mint_doc_images(doc.data, doc.asset, ctx)
        if not text.strip():
            return (
                f"'{doc.name}' was parsed but contains no extractable text — it may be a "
                "scanned/image-only document — and you must NOT summarize a different "
                "document in its place." + footer
            )
        return _truncate(text) + footer

    runtime.register(
        define_tool(
            name="read_document",
            description="Extract the text content of a document the user attached to the "
            "chat or has open in the viewer (PDF, Word .doc/.docx, Excel .xlsx/.xlsm, "
            "PowerPoint .pptx/.potx/.ppsx, "
            "txt/markdown/csv/json, subtitles). Attachments arrive as "
            "[Attached: <filename> (asset_id <id>)]; the Viewer Access Context note hands "
            "you an asset_id the same way. Whenever the "
            "user asks about the content of such a document, you MUST call this tool "
            "with that asset_id instead of guessing or claiming the file cannot be read. "
            "Pass `pages` to read specific pages/slides (\"3\", \"2,5\", \"1-3\") — "
            "Only PDF and PowerPoint slide-based formats (PPTX/POTX/PPSX) support pages; "
            "for any other format omit the parameter (page-scoped requests are rejected, "
            "never answered with a full-document substitute). "
            "Embedded pictures in a legacy .doc come back with their own asset_ids at the "
            "end of the result — pass those ids to the `vision` tool to read the image or "
            "table. For attached images and screenshots use the `vision` tool instead.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Cloud-drive asset id of the document "
                        "(from the [Attached: ...] or Viewer Access Context note).",
                    },
                    "pages": {
                        "type": "string",
                        "description": '1-based page/slide spec, e.g. "3", "2,5", "1-3". '
                        "Omit to read the whole document. Only PDF and PowerPoint "
                        "slide-based formats (PPTX/POTX/PPSX) support pages.",
                    },
                },
                "required": ["asset_id"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=read_document,
        )
    )

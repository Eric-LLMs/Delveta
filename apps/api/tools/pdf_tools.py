"""``pdf_extract_text`` / ``pdf_table_to_text``: pull searchable text out of drive PDFs.

Body text comes from PyMuPDF; tables are located by ``page.find_tables()``, all detected
table regions are composited into ONE image locally, and that image is read by the vision
model in a single call (Local Data Preparation → Single LLM Call). Both tools read the
asset's stored bytes via the shared storage + session_factory provided in ``deps.py``, so
an agent can inspect a drive PDF that the ingest worker would otherwise handle
automatically.

The table call carries the user's ORIGINAL request: the vision model decides what to do
with the table (transcribe it exactly when the user asked for its data, explain/summarize/
compare it when that is what they asked). The image is prepared locally; the intent is
never decided by a second model call and no transcription-then-synthesis pass exists.
"""
from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block
from agent.engine.context import current_turn
from core.application.drive_service import DriveError, DriveService
from core.infrastructure import pdf as pdf_lib
from core.infrastructure.pdf_pages import parse_pages_spec
from core.infrastructure.request_context import get_request_user_id, request_user
from core.infrastructure.storage import object_key
from core.infrastructure.vision_caption import (
    VisionNotAuthorized,
    VisionUnsupported,
    describe_image,
)

log = logging.getLogger(__name__)

# Used only when no user request is in scope (defensive: this tool runs inside a chat
# turn, whose ``user_msg`` is the request the model must answer).
_TRANSCRIBE_PROMPT = (
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


def _table_prompt(user_query: str) -> str:
    """The text part of the ONE vision request for a table image.

    The user's own sentence is the instruction: the model reads the table and does what
    was asked of it — transcribe the data faithfully, or explain / summarize / compare the
    content — instead of being pinned to a transcription-only order. Exactness and
    no-invention are the only standing constraints. With no request in scope (a headless
    caller) the historical transcription order stands unchanged.
    """
    query = (user_query or "").strip()
    if not query:
        return _TRANSCRIBE_PROMPT
    return (
        "下面的图片是这份 PDF 中表格区域的内容。\n"
        f"用户的要求：{query}\n\n"
        "请按用户的要求处理这张图片：要求提取或转写时，逐字保留表格中的行列、数字和"
        "取值，不得改写；要求解释、总结、分析或比较时，直接给出对应的回答，不要把整张"
        "表格先逐行抄一遍（只在必要时引用个别单元格）。"
        "只使用图片中能看到的信息，不得编造图片中不存在的内容。"
    )


async def _load_asset_bytes(asset_id: str, ctx: Context) -> bytes:
    """Fetch a drive asset's stored object bytes by ``asset_id``, after the ACL check.

    Permission is decided before any bytes are read (the same rule ``read_document``
    and ``vision`` follow): a foreign/unknown asset id raises here, so an arbitrary
    UUID cannot be used to read another user's file.
    """
    uid = request_user.get()
    if uid is None:
        raise ValueError("pdf tools require an authenticated request user context")
    drive = DriveService(ctx.resolve("session_factory"))
    try:
        asset = await drive.ensure_asset_readable(uid, UUID(str(asset_id)))
    except DriveError as exc:
        raise ValueError(f"asset {asset_id} not readable: {exc}") from None
    if not asset.object_sha256:
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
        # LOCAL data preparation: every detected table region is composited into ONE image
        # (no model), then the vision model reads it in a SINGLE call — a table request is
        # exactly ONE downstream request, never one call per table.
        strip = await asyncio.to_thread(pdf_lib.render_tables_png, data, pages=pages)
        if strip is None:
            return "No tables detected in this PDF."
        # The user's own sentence goes with the image; the vision model answers it. On a
        # live chat turn the shared entry streams every delta to the user (content sink)
        # and returns exactly that text, so done.answer and the saved row cannot diverge.
        turn = current_turn()
        user_query = str(getattr(turn, "user_msg", "") or "") if turn is not None else ""
        try:
            return await describe_image(
                strip,
                "image/png",
                llm=llm,
                session_factory=ctx.resolve("session_factory"),
                prompt=_table_prompt(user_query),
                user_id=get_request_user_id(),
            )
        except VisionNotAuthorized as exc:
            return f"Table analysis refused: {exc}"
        except VisionUnsupported as exc:
            return f"Table analysis not supported: {exc}"

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
            description="Detect tables in a cloud-drive PDF, render all detected tables into "
            "ONE image, and have the vision model read it with the user's request in a "
            "single call. Returns the model's answer: the exact table data when the user "
            "asked for it, an explanation / summary / comparison when that is what they "
            "asked for.",
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

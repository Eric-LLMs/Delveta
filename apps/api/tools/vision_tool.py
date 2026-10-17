"""``vision``: read a drive image (or a rendered document page/region) with the user's
authorized models — LOCAL data preparation, ONE specialized LLM call.

The agent's main model may be text-only, so it cannot see the screenshots and images
users attach to the chat — it only sees the ``[Attached: name (asset_id)]`` note. This
tool resolves the asset bytes from the cloud drive, sends them to the vision-capable
models the requesting user's role is authorized for, and returns the first real analysis
so the agent can discuss the visual content.

When the asset is a **PDF** (``page`` given, or the caller is looking at a document), the
page — or a sub-``region`` of it — is rendered to an image LOCALLY with PyMuPDF before
the model call: locating/rendering content is deterministic CPU work, never a model
request, so a table/chart/figure page question costs exactly ONE downstream vision call
(the "Local Data Preparation → Single LLM Call" rule). No second model request is made to
extract or transcribe first.

Access is checked against the drive ACL (``DriveService.ensure_asset_readable``) BEFORE
any bytes are read, so a forged ``asset_id`` for another user's file is refused, never
silently analyzed.

Channel selection follows the unified permission funnel
(``core.infrastructure.vision_caption.resolve_vision_channels``): the candidate list is
exclusively the role's database-authorized models — vision-marked names tried first, the
rest as a capability gamble; guests use the anonymous tier under its existing daily
quota; a model-less role downgrades onto that tier and is refused honestly once the
free allowance runs out. The resolution and the ``describe_image`` trial chain live in
``core.infrastructure.vision_caption`` so the RAG ingest worker (image captioning at
index time) can reuse them without importing this module.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from agent import Context, ToolExecution, ToolOutput, ToolRuntime, define_tool, text_block
from agent.engine.context import current_turn
from core.application.drive_service import DriveError, DriveService
from core.infrastructure import pdf as pdf_lib
from core.infrastructure.request_context import get_request_user_id
from core.infrastructure.storage import object_key
from core.infrastructure.vision_caption import (
    DEFAULT_PROMPT as _DEFAULT_PROMPT,
    VisionNotAuthorized,
    VisionUnsupported,
    describe_image,
    mime_for as _mime_for,
)

# Formats the tool renders to an image when a page is in scope (a page axis exists only for
# PDF today; other text formats go through ``read_document`` instead).
_PAGE_RENDER_EXTS = {".pdf"}


def _is_pdf(asset: Any) -> bool:
    mime = (getattr(asset, "mime_type", None) or "").lower()
    ext = Path(getattr(asset, "name", None) or "").suffix.lower()
    return mime == "application/pdf" or ext in _PAGE_RENDER_EXTS


def _viewer_page_from_turn() -> int | None:
    """The open viewer's current page for THIS turn, or ``None``.

    The registry does not give ``cap-vision`` a ``pages`` slot, so a page-scoped document
    question reaches the tool as ``viewer.asset_id`` alone; the page is read here from the
    SAME turn assembly the funnel built (never parsed out of the sentence).
    """
    turn = current_turn()
    ctx_dict = getattr(turn, "context", None) or {}
    asset = (ctx_dict.get("viewer") or {}).get("asset") or {}
    page = asset.get("page")
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


async def _load_asset(asset_id: str, ctx: Context, user_id: UUID) -> tuple[Any, bytes]:
    """Fetch a drive asset's stored bytes plus its row, after the drive ACL check.

    Permission is decided up-front (mirrors ``read_document``): a foreign/unknown asset
    raises here, so no bytes are ever read for an asset the request user may not see.
    """
    drive = DriveService(ctx.resolve("session_factory"))
    try:
        asset = await drive.ensure_asset_readable(user_id, UUID(str(asset_id)))
    except DriveError as exc:
        raise ValueError(f"asset {asset_id} not readable: {exc}") from None
    if not asset.object_sha256:
        raise ValueError(f"asset {asset_id} not found or has no stored object")
    data = await ctx.resolve("storage").get(object_key(asset.object_sha256))
    if data is None:
        raise ValueError(f"object bytes missing for asset {asset_id}")
    return asset, data


def register(runtime: ToolRuntime, ctx: Context, llm) -> None:
    async def vision_analyze(args: dict, exec: ToolExecution) -> str:
        asset_id = args["asset_id"]
        question = (args.get("question") or "").strip()
        user_id = get_request_user_id()
        if user_id is None:
            raise ValueError("vision requires an authenticated request user context")
        asset, raw = await _load_asset(asset_id, ctx, user_id)

        # ── LOCAL data preparation: a page-scoped PDF becomes an image here (a model is
        # never asked to locate or transcribe first). Everything else is its own bytes.
        if _is_pdf(asset):
            page = args.get("page")
            if page is None:
                page = _viewer_page_from_turn()
            region = args.get("region") or None
            if page is None:
                # No ``page`` arg AND the viewer declared none for THIS turn. A multi-page
                # PDF must NEVER silently render page 1 — an invisible wrong-page answer
                # the user cannot see. A single-page PDF is unambiguous, so render it.
                count = await asyncio.to_thread(pdf_lib.page_count, raw)
                if count != 1:
                    raise ValueError(
                        f"page not specified for a {count}-page PDF — pass `page`, or "
                        "open the page in the viewer; the tool never silently reads page 1"
                    )
                page_num = 1
            else:
                page_num = int(page)
            data = await asyncio.to_thread(
                pdf_lib.render_page_png, raw, page_num, region=region
            )
            mime = "image/png"
        else:
            data, mime = raw, (_mime_for(asset) or "")

        prompt = f"基于所附图片回答以下问题：{question}" if question else _DEFAULT_PROMPT
        try:
            return await describe_image(
                data,
                mime,
                llm=llm,
                session_factory=ctx.resolve("session_factory"),
                prompt=prompt,
                user_id=user_id,
            )
        except VisionNotAuthorized as exc:
            # Surface the authorization failure honestly — the agent tells the user instead
            # of the tool silently trying another key.
            return f"Vision analysis refused: {exc}"
        except VisionUnsupported as exc:
            return f"Image processing not supported: {exc}"

    runtime.register(
        define_tool(
            name="vision",
            description="Analyze an image or screenshot that the user attached to the chat, "
            "or a page/region of a cloud-drive PDF the user is looking at. Attachments "
            "arrive as [Attached: <filename> (<asset_id>)]. Whenever the user references "
            "an attached image or screenshot, you MUST call this tool with that asset_id to "
            "inspect the visual content before answering — even if the filename seems "
            "descriptive. For a PDF, pass `page` (and optionally a fractional `region`) to "
            "read that page visually — the page is rendered locally and read by the model "
            "in ONE call. The configured vision model returns its analysis, or answers the "
            "user's question about the content. Pass the user's specific question via "
            "`question` when they ask about a detail.",
            parameters={
                "type": "object",
                "properties": {
                    "asset_id": {
                        "type": "string",
                        "description": "Cloud-drive asset id of the attached image, "
                        "screenshot, or PDF (from the [Attached: ...] note or viewer context).",
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional question about the image/page. Omit to get "
                        "a general analysis of the visual content.",
                    },
                    "page": {
                        "type": "integer",
                        "description": "1-based PDF page to render and read. Omit to use "
                        "the page open in the viewer. A multi-page PDF with no page in view "
                        "is refused (never a silent page 1); a single-page PDF reads page 1. "
                        "Ignored for image assets.",
                    },
                    "region": {
                        "type": "object",
                        "description": "Optional sub-region of the rendered page, as "
                        "fractions {x, y, w, h} of the page (0..1, top-left origin). Use it "
                        "to focus the model on a table/chart/figure. Omit for the whole page.",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                    },
                },
                "required": ["asset_id"],
            },
            output=ToolOutput(
                schema={"type": "string"}, render=lambda args, value: [text_block(value)]
            ),
            execute=vision_analyze,
        )
    )

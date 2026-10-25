"""Turn material for chat-facing content tools (``translate`` / ``summary_gen``).

Both tools must answer from the SAME per-turn material: the user's sentence, the
recent chat, the open viewer's identity, the current document page's extracted
text, and — for video — the subtitle window plus an optional captured frame.

This module is the ONE place that reads those from the turn context. It never
calls the user-facing LLM: the decision / translation / summary call belongs to
the tool. Document text is extracted LOCALLY by the authorized reader
(``read_asset_document``: drive ACL -> storage -> pdf/pptx/text extraction), so
carrying a page into a prompt costs no extra model call.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

from agent.engine.context import current_turn
from api.tools.asset_bytes import load_asset_bytes
from api.tools.document_content import read_asset_document
from core.infrastructure.request_context import get_request_user_id

logger = logging.getLogger(__name__)

# Viewer kinds whose content is a document the authorized reader can extract.
DOCUMENT_KINDS = frozenset({"pdf", "office", "text", "markdown"})

# One page of text is plenty for a translation; a whole-document read is bounded
# so a 300-page PDF cannot become the request's payload.
MAX_DOCUMENT_CHARS = 12_000

# A captured frame beyond this is refused rather than silently truncated by the
# provider; the tool degrades to subtitles and says why.
MAX_FRAME_BYTES = 8 * 1024 * 1024

_CHAT_ROLES = frozenset({"user", "assistant"})
_SUBTITLE_KINDS = ("subtitle_window", "full_subtitles")
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})


@dataclass(frozen=True)
class ChatMessage:
    """One prior chat row (user/assistant only, never internal agent state)."""

    role: str
    content: str

    def render(self) -> str:
        return f"{self.role}: {self.content}"


@dataclass(frozen=True)
class ViewerInfo:
    """What this turn's viewer declares — identity only, no content."""

    kind: str = ""
    name: str = ""
    asset_id: str = ""
    page: int | None = None
    t_ms: int | None = None

    @property
    def is_document(self) -> bool:
        return bool(self.asset_id) and self.kind in DOCUMENT_KINDS

    @property
    def page_spec(self) -> str:
        return str(self.page) if self.page is not None else ""


@dataclass(frozen=True)
class DocumentSlice:
    """The authorized text of one document (optionally one page range)."""

    asset_id: str
    name: str
    pages: tuple[int, ...]
    text: str
    truncated: bool = False


def _field(msg: Any, name: str) -> str:
    return str((msg.get(name) if isinstance(msg, dict)
                else getattr(msg, name, "")) or "")


def turn_query() -> str:
    """The current user message for THIS turn, or ``""`` outside a turn."""
    turn = current_turn()
    return str(getattr(turn, "user_msg", "") or "")


def turn_history() -> tuple[ChatMessage, ...]:
    """The turn's prior chat rows (user + assistant), in original order."""
    turn = current_turn()
    rows: list[ChatMessage] = []
    for msg in (getattr(turn, "history", None) or ()):
        role = _field(msg, "role")
        content = _field(msg, "content").strip()
        if role in _CHAT_ROLES and content:
            rows.append(ChatMessage(role=role, content=content))
    return tuple(rows)


def recent_messages(count: int) -> tuple[ChatMessage, ...]:
    """The last ``count`` chat rows, oldest first."""
    if count <= 0:
        return ()
    rows = turn_history()
    return rows[-count:]


def viewer_info() -> ViewerInfo:
    """The open viewer's declared identity for this turn (empty when none)."""
    turn = current_turn()
    context = getattr(turn, "context", None) or {}
    asset = (context.get("viewer") or {}).get("asset") or {}
    page = asset.get("page")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None
    t_ms = asset.get("t_ms")
    return ViewerInfo(
        kind=str(asset.get("kind") or "").lower(),
        name=str(asset.get("name") or ""),
        asset_id=str(asset.get("asset_id") or ""),
        page=page,
        t_ms=int(t_ms) if isinstance(t_ms, (int, float)) else None,
    )


def viewer_blocks() -> list:
    """The viewer assembly's reference blocks for this turn (may be empty)."""
    turn = current_turn()
    context = getattr(turn, "context", None) or {}
    return list((context.get("viewer") or {}).get("blocks") or [])


def video_subtitles() -> str:
    """The current video subtitle window (and full transcript when assembled)."""
    parts: list[str] = []
    for block in viewer_blocks():
        if str(getattr(block, "kind", "") or "") not in _SUBTITLE_KINDS:
            continue
        text = str(getattr(block, "text", "") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


async def read_document_slice(ctx, llm, asset_id: str, *,
                              pages: list[int] | None = None,
                              caller: str = "tool") -> DocumentSlice:
    """Read one authorized document (or its page range) into bounded text.

    Raises ``ValueError`` when the asset is unreadable, an image, or has no
    extractable text — the caller reports that honestly, never substitutes
    another source.
    """
    doc = await read_asset_document(ctx, llm, asset_id, pages=pages, caller=caller)
    if doc.is_image:
        raise ValueError(
            f"'{doc.name}' is an image; the document reader cannot extract text "
            "from it")
    text = doc.text.strip()
    if not text:
        raise ValueError(
            f"'{doc.name}' contains no extractable text (it may be a scanned, "
            "image-only document)")
    truncated = len(text) > MAX_DOCUMENT_CHARS
    return DocumentSlice(
        asset_id=str(asset_id),
        name=doc.name,
        pages=tuple(pages or ()),
        text=text[:MAX_DOCUMENT_CHARS],
        truncated=truncated,
    )


def _image_mime(asset: Any, name: str) -> str:
    mime = str(getattr(asset, "mime_type", "") or "").lower()
    if mime in _IMAGE_MIMES:
        return mime
    lowered = name.lower()
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


async def capture_image_data_url(ctx, asset_id: str, *,
                                 caller: str = "tool") -> str:
    """A captured picture as a ``data:`` URL for a multimodal call.

    ``asset_id`` comes from the deferred-capture resume, so it can only be the
    client's real upload. Raises ``ValueError`` when the request has no user, the
    asset is unreadable/empty, or the frame exceeds the size cap.
    """
    user_id = get_request_user_id()
    if user_id is None:
        raise ValueError(f"{caller} requires an authenticated request user context")
    asset, raw = await load_asset_bytes(ctx, asset_id, user_id=user_id)
    if not raw:
        raise ValueError(f"captured frame {asset_id} is empty")
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError(
            f"captured frame {asset_id} is {len(raw)} bytes, above the "
            f"{MAX_FRAME_BYTES}-byte cap")
    mime = _image_mime(asset, str(getattr(asset, "name", "") or ""))
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

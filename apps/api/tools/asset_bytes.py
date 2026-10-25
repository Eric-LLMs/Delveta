"""Authorized asset-byte loading for tools that read media.

Permission is decided BEFORE any bytes are read, exactly like ``read_document``:
a drive file goes through the drive ACL; an owned ``scope='chat'`` capture
(screenshot, page/region capture, video frame) is authorized owner-only in the
chat plane, which the drive plane deliberately refuses (404).

``vision`` and ``translate`` both need this rule and must not drift apart, so it
lives here once. The caller passes the tool's runtime ``Context`` and the
request's user id; nothing here invents an id or reads an unauthorized asset.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from core.application.drive_service import DriveError, DriveService
from core.infrastructure.storage import object_key


async def owned_chat_asset(drive: DriveService, user_id: UUID, asset_id: UUID) -> Any:
    """The caller's OWNED ``scope='chat'`` capture, or ``None``.

    Screenshots, page/region captures and video frames live in the chat plane,
    which the drive plane deliberately refuses (404) — but they are exactly what
    the media tools are asked to read. Any other scope, owner or deleted row
    returns ``None``.
    """
    try:
        asset = await drive.assets.get(asset_id)
    except Exception:  # noqa: BLE001 — an unreadable row is simply not authorized
        return None
    if asset is None or asset.deleted_at is not None:
        return None
    if (asset.scope or "drive") != "chat" or asset.user_id != user_id:
        return None
    return asset


async def load_asset_bytes(ctx, asset_id: str, *,
                           user_id: UUID) -> tuple[Any, bytes]:
    """Fetch an asset's stored bytes plus its row, after the ACL check.

    A foreign/unknown asset raises here, so no bytes are ever read for an asset
    the request user may not see.
    """
    drive = DriveService(ctx.resolve("session_factory"))
    aid = UUID(str(asset_id))
    try:
        asset = await drive.ensure_asset_readable(user_id, aid)
    except DriveError as exc:
        asset = await owned_chat_asset(drive, user_id, aid)
        if asset is None:
            raise ValueError(f"asset {asset_id} not readable: {exc}") from None
    if not asset.object_sha256:
        raise ValueError(f"asset {asset_id} not found or has no stored object")
    data = await ctx.resolve("storage").get(object_key(asset.object_sha256))
    if data is None:
        raise ValueError(f"object bytes missing for asset {asset_id}")
    return asset, data

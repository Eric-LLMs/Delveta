"""Chat-asset service: the session-plane lifecycle of chat attachments.

A chat asset is a document / image / frame stored in the same CAS as the cloud drive but
``scope='chat'`` — invisible to the entire drive plane (Files / search / picker / trash /
download) and reachable only through the Session it belongs to. The authority for "this
asset belongs to this chat" is the ``chat_attachments`` table; this service is the single
place that creates, mounts, and frees those links, so ``ref_count`` is only ever changed
here (or in :class:`DriveService`) inside a defined operation.

Two shapes of link (design §4):

- ``owned`` — the asset was created FOR the chat (``assets.scope='chat'``); deleting the
  chat frees the asset and its share of the physical object.
- ``reference`` — a normal drive file merely mounted into the chat (a viewer payload, a
  ``+``-picked file); deleting the chat drops the link and never touches the drive file.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import UUID

from core.infrastructure.db import ChatAttachmentModel
from core.infrastructure.drive_repositories import (
    SqlAssetAclRepository,
    SqlAssetRepository,
    SqlChatAttachmentRepository,
    SqlGlobalObjectRepository,
)
from core.infrastructure.storage import Storage, get_storage, object_key

_CHAT = "chat"


class ChatAssetService:
    """Owns the chat_attachments lifecycle: draft create / reference mount / free on delete."""

    def __init__(
        self,
        session_factory,
        storage: Storage | None = None,
        *,
        objects: SqlGlobalObjectRepository | None = None,
        assets: SqlAssetRepository | None = None,
        attachments: SqlChatAttachmentRepository | None = None,
        acl: SqlAssetAclRepository | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.storage = storage or get_storage()
        self.objects = objects or SqlGlobalObjectRepository(session_factory, storage=self.storage)
        self.assets = assets or SqlAssetRepository(session_factory)
        self.attachments = attachments or SqlChatAttachmentRepository(session_factory)
        self.acl = acl or SqlAssetAclRepository(session_factory)

    # ── create ────────────────────────────────────────────────────────────────

    async def create_owned_draft(
        self,
        user_id: UUID,
        name: str,
        data: bytes,
        mime_type: str | None,
        *,
        source: str = "upload",
        locator: dict | None = None,
    ) -> ChatAttachmentModel:
        """Store bytes in the CAS, create a ``scope='chat'`` asset, open an ``owned`` draft.

        The object bytes are written under the per-SHA lock (``upsert_and_increment`` with
        ``data``), so a concurrent collect can never leave the object ``live`` with no file.
        """
        sha = hashlib.sha256(data).hexdigest()
        await self.objects.upsert_and_increment(
            sha, len(data), object_key(sha), mime_type, data=data
        )
        asset = await self.assets.create(
            user_id,
            name,
            mime_type=mime_type,
            size=len(data),
            object_sha256=sha,
            file_status="READY",
            rag_status="NOT_STARTED",
            scope=_CHAT,
        )
        return await self.attachments.create_draft(
            user_id, asset.id, ownership="owned", source=source, locator=locator
        )

    async def mount_reference_draft(
        self,
        user_id: UUID,
        asset_id: UUID,
        *,
        source: str = "viewer",
        locator: dict | None = None,
    ) -> ChatAttachmentModel | None:
        """Mount a readable drive file into a chat as an ``owned=False`` draft link.

        Returns ``None`` when the asset is not a drive file the caller may read (wrong scope,
        missing, foreign) — the caller treats that as "no attachment", never an error.
        """
        asset = await self.assets.get_active(asset_id)
        if asset is None or (asset.scope or "drive") != "drive":
            return None
        if not await self._can_read(user_id, asset):
            return None
        return await self.attachments.create_draft(
            user_id, asset_id, ownership="reference", source=source, locator=locator
        )

    async def _can_read(self, user_id: UUID, asset) -> bool:
        """Owner / shared-workspace / ACL read check for a drive asset (design §4 check 2)."""
        if asset.user_id == user_id:
            return True
        if asset.workspace_id is not None:
            from core.infrastructure.drive_repositories import SqlWorkspaceRepository

            role = await SqlWorkspaceRepository(self.session_factory).member_role(
                asset.workspace_id, user_id
            )
            if role is not None:
                return True
        return await self.acl.permission_for(asset.id, user_id) is not None

    async def draft_ids_for_assets(self, user_id: UUID, asset_ids: list[UUID]) -> list[UUID]:
        """Resolve the caller's unbound draft link ids for these owned assets.

        At Send a viewer frame/roi arrives as the ASSET id on the selection
        (``image_asset_id``, design §7); the draft opened at upload is bound to the message
        through its id. Foreign / already-bound / unknown assets resolve to nothing.
        """
        return await self.attachments.find_draft_ids(user_id, asset_ids)

    # ── free (counted asset retirement) ───────────────────────────────────────

    async def _retire_owned_asset(self, asset_id: UUID) -> None:
        """Logical delete of an OWNED chat asset: soft-delete the row + free its ref-count.

        Design §8: user-initiated delete is *logical*. The row is marked ``deleted_at`` so it
        is immediately invisible to the session plane, and its share of the CAS object is
        released with the guarded, row-atomic ``free`` (which never touches the file). The
        physical file delete is deferred to the background GC (:meth:`gc`, design §9.7),
        which rechecks references/state/asset rows after the grace window before collecting.

        A missing, foreign, non-chat or already-retired asset is a no-op — locating an owned
        asset must never fail a chat delete. Never hard-deletes anything.
        """
        asset = await self.assets.get(asset_id)
        if (
            asset is None
            or (asset.scope or "drive") != _CHAT
            or asset.deleted_at is not None
        ):
            return
        await self.assets.soft_delete(asset.id)
        if asset.object_sha256:
            await self.objects.free(asset.object_sha256)

    async def _retire_rows(self, rows: list[ChatAttachmentModel]) -> None:
        for row in rows:
            if row.ownership == "owned":
                await self._retire_owned_asset(row.asset_id)

    # ── delete ────────────────────────────────────────────────────────────────

    async def delete_attachment(self, user_id: UUID, attachment_id: UUID) -> bool:
        """Delete one link; free its owned asset, drop a reference link only. Owner-scoped."""
        row = await self.attachments.get(attachment_id)
        if row is None or row.user_id != user_id:
            return False
        removed = await self.attachments.delete(attachment_id)
        if removed is not None:
            await self._retire_rows([removed])
        return True

    async def delete_for_message(self, message_id: UUID) -> None:
        """Delete a message's links; retire owned assets (logical), drop reference links."""
        await self._retire_rows(await self.attachments.delete_for_message(message_id))

    async def delete_for_session(self, session_id: UUID) -> None:
        """Delete a session's links (bound + draft); retire owned assets (logical)."""
        await self._retire_rows(await self.attachments.delete_for_session(session_id))

    async def detach_message(self, message_id: UUID) -> None:
        """Keep a message's links (and their owned assets) when only the row is deleted.

        The edit-and-regenerate path: the client deletes the old question but must keep the
        screenshot it was built around. The links are re-pinned to session scope so the asset
        keeps a reference and is freed later with the session, never orphaned.
        """
        await self.attachments.detach_message(message_id)

    # ── read ──────────────────────────────────────────────────────────────────

    async def read_owned_bytes(self, user_id: UUID, asset_id: UUID) -> tuple[str | None, bytes] | None:
        """Read an OWNED chat asset's bytes for the caller (design §4 check 1).

        Owner-only (``assets.user_id = caller`` AND ``scope='chat'``). Used to inline an
        uploaded screenshot BEFORE its draft is bound, when no session-scoped
        ``chat_attachments`` lookup exists yet; ``None`` on any mismatch so the caller keeps
        its existing fallback (the ``vision`` tool path).
        """
        asset = await self.assets.get(asset_id)
        if (
            asset is None
            or asset.deleted_at is not None
            or (asset.scope or "drive") != _CHAT
            or asset.user_id != user_id
            or not asset.object_sha256
        ):
            return None
        data = await self.storage.get(object_key(asset.object_sha256))
        if data is None:
            return None
        return asset.mime_type, data

    async def list_for_session_with_assets(self, session_id: UUID):
        return await self.attachments.list_for_session_with_assets(session_id)

    async def get_for_session(self, attachment_id: UUID, session_id: UUID):
        return await self.attachments.get_for_session(attachment_id, session_id)

    async def read_bytes(self, attachment_id: UUID, session_id: UUID) -> bytes | None:
        """Stream an attachment's bytes, authorized on its session (§5.2). None if unresolved.

        Double-checked: the ``chat_attachments`` row must belong to ``session_id``; for an
        ``owned`` asset the asset must be ``scope='chat'``; for a ``reference`` the drive
        visibility is re-checked at read time, so a revoked ACL revokes chat access too.
        """
        att = await self.attachments.get_for_session(attachment_id, session_id)
        if att is None:
            return None
        asset = await self.assets.get(att.asset_id)
        if asset is None or asset.deleted_at is not None:
            return None
        if att.ownership == "owned":
            if (asset.scope or "drive") != _CHAT:
                return None
        elif not await self._can_read(att.user_id, asset):
            return None
        if not asset.object_sha256:
            return None
        return await self.storage.get(object_key(asset.object_sha256))

    # ── GC ────────────────────────────────────────────────────────────────────

    async def draft_gc(self, now: datetime | None = None) -> dict:
        """Reap abandoned drafts whose TTL has passed: retire owned assets + delete the rows."""
        rows = await self.attachments.list_expired_drafts(now=now or datetime.now(timezone.utc))
        if not rows:
            return {"expired": 0}
        removed = await self.attachments.delete_by_ids([r.id for r in rows])
        await self._retire_rows(removed)
        return {"expired": len(removed)}

    async def gc(
        self, *, grace_seconds: int = 3600, now: datetime | None = None
    ) -> dict:
        """Background reclamation sweep (design §8, §9.7) — the periodic, idempotent GC.

        Three phases, all idempotent and safe to replay:

        1. **Drafts** past their TTL: retire owned assets (logical) + drop the rows.
        2. **Tombstones**: hard-delete owned chat asset rows soft-deleted longer than the
           grace window. Their ref-count share was already freed at logical-delete time, so
           the rows are pure tombstones once grace passes — this bounds them so a chat asset
           whose object is shared with a still-live drive file is never un-reclaimable.
        3. **Objects**: collect ``pending_delete`` objects past grace. The physical file
           delete + row delete run under the per-SHA advisory lock (design §9.3(3)), after a
           reference/state recheck; this is also the recovery path for a lost GC job.

        Object collection must follow tombstone reaping: it rechecks *live* asset rows and
        deletes the soft-deleted chat rows guarding the ``object_sha256`` FK.
        """
        now = now or datetime.now(timezone.utc)
        drafts = await self.draft_gc(now=now)
        cutoff = now - timedelta(seconds=grace_seconds)
        assets_reaped = await self.assets.purge_deleted_chat_assets(cutoff)
        objects = await self.objects.sweep_pending(grace_seconds=grace_seconds)
        return {
            "drafts": drafts.get("expired", 0),
            "assets_reaped": assets_reaped,
            "objects": objects,
        }

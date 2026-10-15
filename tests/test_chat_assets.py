"""Chat-asset plane: the session-scoped lifecycle of chat attachments (design §4–§9).

Hand-rolled fakes mirror the SQL repository surface (no DB), the same style as
``tests/_drive_fakes.py``: the CAS + asset rows come from the drive fakes, and a small
``FakeChatAttachments`` mirrors ``SqlChatAttachmentRepository``. The service runs unchanged.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from core.application.chat_asset_service import ChatAssetService
from core.infrastructure.db import ChatAttachmentModel
from tests._drive_fakes import FakeAcl, FakeAssets, FakeObjects, FakeWorkspaces
from core.infrastructure.storage import LocalStorage, object_key


class FakeChatAttachments:
    """Mirror SqlChatAttachmentRepository over an in-memory dict."""

    def __init__(self) -> None:
        self.rows: dict[UUID, ChatAttachmentModel] = {}

    def _make(self, user_id, asset_id, *, ownership, source, status, locator,
              session_id=None, message_id=None, bound_at=None, expires_at=None):
        row = ChatAttachmentModel(
            id=uuid4(), asset_id=asset_id, user_id=user_id, session_id=session_id,
            message_id=message_id, ownership=ownership, source=source, status=status,
            locator=locator or {}, created_at=datetime.now(timezone.utc),
            bound_at=bound_at, expires_at=expires_at,
        )
        self.rows[row.id] = row
        return row

    async def create_draft(self, user_id, asset_id, *, ownership="owned", source="upload",
                           locator=None, ttl_seconds=86400):
        return self._make(
            user_id, asset_id, ownership=ownership, source=source, status="draft",
            locator=locator,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )

    async def create_reference(self, user_id, asset_id, session_id, message_id, *,
                               source="viewer", locator=None):
        return self._make(
            user_id, asset_id, ownership="reference", source=source, status="bound",
            locator=locator, session_id=session_id, message_id=message_id,
            bound_at=datetime.now(timezone.utc),
        )

    async def bind(self, attachment_ids, user_id, session_id, message_id):
        out = []
        for aid in attachment_ids:
            row = self.rows.get(aid)
            if row is None or row.user_id != user_id or row.status != "draft":
                continue
            row.status = "bound"
            row.session_id = session_id
            row.message_id = message_id
            row.bound_at = datetime.now(timezone.utc)
            row.expires_at = None
            out.append(row)
        return out

    async def get(self, attachment_id):
        return self.rows.get(attachment_id)

    async def get_for_session(self, attachment_id, session_id):
        row = self.rows.get(attachment_id)
        if row is None or row.session_id != session_id:
            return None
        return row

    async def find_draft_ids(self, user_id, asset_ids):
        wanted = set(asset_ids)
        return [
            r.id for r in self.rows.values()
            if r.user_id == user_id and r.asset_id in wanted and r.status == "draft"
        ]

    async def list_for_message(self, message_id):
        return [r for r in self.rows.values() if r.message_id == message_id]

    async def delete(self, attachment_id):
        return self.rows.pop(attachment_id, None)

    async def delete_for_message(self, message_id):
        removed = [r for r in self.rows.values() if r.message_id == message_id]
        for r in removed:
            self.rows.pop(r.id, None)
        return removed

    async def delete_for_session(self, session_id):
        removed = [r for r in self.rows.values() if r.session_id == session_id]
        for r in removed:
            self.rows.pop(r.id, None)
        return removed

    async def list_expired_drafts(self, now=None):
        cutoff = now or datetime.now(timezone.utc)
        return [
            r for r in self.rows.values()
            if r.status == "draft" and r.expires_at is not None and r.expires_at <= cutoff
        ]

    async def delete_by_ids(self, attachment_ids):
        removed = []
        for aid in attachment_ids:
            row = self.rows.pop(aid, None)
            if row is not None:
                removed.append(row)
        return removed

    async def detach_message(self, message_id):
        n = 0
        for r in self.rows.values():
            if r.message_id == message_id:
                r.message_id = None
                n += 1
        return n


def make_service(tmp_path):
    """A ChatAssetService over the drive fakes + a real LocalStorage under tmp_path."""
    storage = LocalStorage(tmp_path)
    objects = FakeObjects(storage)
    assets = FakeAssets(FakeWorkspaces())
    attachments = FakeChatAttachments()
    acl = FakeAcl()
    svc = ChatAssetService(
        None, storage=storage, objects=objects, assets=assets, attachments=attachments,
        acl=acl,
    )
    return svc, objects, assets, attachments


# ── create ────────────────────────────────────────────────────────────────────────

async def test_create_owned_draft_stores_chat_asset_and_draft(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user = uuid4()

    row = await svc.create_owned_draft(user, "shot.png", b"PNGDATA", "image/png",
                                       source="screenshot")

    asset = await assets.get(row.asset_id)
    assert asset.scope == "chat"          # drive-plane invisible (§3.1)
    assert asset.user_id == user
    assert asset.rag_status == "NOT_STARTED"
    assert row.ownership == "owned"
    assert row.source == "screenshot"
    assert row.status == "draft"
    assert row.session_id is None and row.message_id is None
    assert row.expires_at is not None     # TTL-bounded (§6.1)
    # Bytes landed in the CAS and are resolvable by sha.
    sha = asset.object_sha256
    assert objects.rows[sha].ref_count == 1
    assert await svc.storage.get(objects.rows[sha].storage_key) == b"PNGDATA"


async def test_chat_scope_asset_is_invisible_to_the_drive_plane(tmp_path):
    """§5.1: every by-id drive endpoint resolves through ``get_active``, so a chat asset
    must resolve to None there (→ 404) exactly like a missing id — never a 403."""
    svc, objects, assets, attachments = make_service(tmp_path)
    user = uuid4()

    row = await svc.create_owned_draft(user, "shot.png", b"x", "image/png")
    drive_asset = await assets.create(user, "d.png", scope="drive", file_status="READY")

    assert await assets.get_active(row.asset_id) is None       # chat → drive-plane 404
    assert await assets.get_active(drive_asset.id) is not None  # drive → resolvable


async def test_mount_reference_draft_rejects_chat_scope_and_foreign(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    owner, other = uuid4(), uuid4()

    chat_asset = await assets.create(owner, "c.png", scope="chat", file_status="READY")
    assert await svc.mount_reference_draft(owner, chat_asset.id) is None  # not a drive file

    drive_asset = await assets.create(owner, "d.png", scope="drive", file_status="READY")
    assert await svc.mount_reference_draft(other, drive_asset.id) is None  # no read access

    ref = await svc.mount_reference_draft(owner, drive_asset.id, source="viewer")
    assert ref is not None
    assert ref.ownership == "reference" and ref.status == "draft"
    assert ref.asset_id == drive_asset.id


# ── read (§4 double check) ─────────────────────────────────────────────────────────

async def test_read_bytes_requires_matching_session(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session = uuid4(), uuid4()
    row = await svc.create_owned_draft(user, "s.png", b"bytes", "image/png")
    row.status, row.session_id = "bound", session

    assert await svc.read_bytes(row.id, session) == b"bytes"
    assert await svc.read_bytes(row.id, uuid4()) is None  # foreign session → 404


async def test_read_bytes_owned_requires_chat_scope(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session = uuid4(), uuid4()
    row = await svc.create_owned_draft(user, "s.png", b"bytes", "image/png")
    row.status, row.session_id = "bound", session
    # A tampered owned link pointing at a drive asset must not stream.
    (await assets.get(row.asset_id)).scope = "drive"
    assert await svc.read_bytes(row.id, session) is None


async def test_read_bytes_reference_rechecks_acl(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    owner, session = uuid4(), uuid4()
    drive_asset = await assets.create(owner, "d.png", scope="drive", file_status="READY")
    drive_asset.object_sha256 = "sha-drive"
    key = object_key("sha-drive")
    objects.rows["sha-drive"] = type(
        "O", (), {"storage_key": key, "ref_count": 1, "state": "live"}
    )()
    await svc.storage.put(key, b"drive-bytes")

    ref = await svc.mount_reference_draft(owner, drive_asset.id)
    ref.status, ref.session_id = "bound", session

    # Re-grant not needed: the owner reads their own drive file.
    assert await svc.read_bytes(ref.id, session) == b"drive-bytes"

    # Revoke: hand the asset to a non-owner (no ACL) → the read-time check fails.
    drive_asset.user_id = uuid4()
    assert await svc.read_bytes(ref.id, session) is None


# ── delete (§8) ────────────────────────────────────────────────────────────────────

async def test_delete_for_session_frees_owned_drops_reference(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session = uuid4(), uuid4()

    owned = await svc.create_owned_draft(user, "s.png", b"owned", "image/png")
    owned.status, owned.session_id = "bound", session
    owned_asset_id = owned.asset_id
    sha = (await assets.get(owned_asset_id)).object_sha256

    drive_asset = await assets.create(user, "d.png", scope="drive", file_status="READY")
    ref = await svc.mount_reference_draft(user, drive_asset.id)
    ref.status, ref.session_id = "bound", session

    await svc.delete_for_session(session)

    # Owned: LOGICAL delete only — row soft-deleted, ref-count freed (object pending),
    # but nothing physical is gone yet (GC does that after the grace window).
    owned_row = await assets.get(owned_asset_id)
    assert owned_row is not None and owned_row.deleted_at is not None
    assert objects.rows[sha].ref_count == 0
    assert objects.rows[sha].state == "pending_delete"
    assert sha in objects.rows                       # file/row NOT yet collected
    # Reference: link dropped, drive file untouched.
    assert ref.id not in attachments.rows
    assert await assets.get(drive_asset.id) is not None


async def test_delete_for_message_only_frees_owned(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session, message = uuid4(), uuid4(), uuid4()

    owned = await svc.create_owned_draft(user, "s.png", b"x", "image/png")
    owned.status, owned.session_id, owned.message_id = "bound", session, message
    owned_asset_id = owned.asset_id

    drive_asset = await assets.create(user, "d.png", scope="drive", file_status="READY")
    ref = await svc.mount_reference_draft(user, drive_asset.id)
    ref.status, ref.session_id, ref.message_id = "bound", session, message

    await svc.delete_for_message(message)

    owned_row = await assets.get(owned_asset_id)
    assert owned_row is not None and owned_row.deleted_at is not None   # logical delete
    assert await assets.get(drive_asset.id) is not None                  # reference untouched
    assert ref.id not in attachments.rows


async def test_detach_message_keeps_owned_asset(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session, message = uuid4(), uuid4(), uuid4()

    owned = await svc.create_owned_draft(user, "s.png", b"x", "image/png")
    owned.status, owned.session_id, owned.message_id = "bound", session, message
    owned_asset_id = owned.asset_id

    await svc.detach_message(message)

    # The asset survives the message delete; the link is re-pinned to the session.
    assert await assets.get(owned_asset_id) is not None
    assert attachments.rows[owned.id].message_id is None
    assert attachments.rows[owned.id].session_id == session


# ── draft GC (§6.1) ────────────────────────────────────────────────────────────────

async def test_draft_gc_reaps_expired_and_frees_owned(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user = uuid4()

    fresh = await svc.create_owned_draft(user, "fresh.png", b"f", "image/png")
    stale = await attachments.create_draft(
        user, (await assets.create(user, "s.png", scope="chat")).id,
        ownership="owned", ttl_seconds=-1,
    )
    stale_asset_id = stale.asset_id

    result = await svc.draft_gc()

    assert result == {"expired": 1}
    assert fresh.id in attachments.rows            # untouched
    assert stale.id not in attachments.rows        # reaped
    stale_row = await assets.get(stale_asset_id)   # owned asset logically deleted
    assert stale_row is not None and stale_row.deleted_at is not None


async def test_gc_reclaims_tombstone_and_collects_object(tmp_path):
    """The full logical-delete → GC cycle: user delete is logical; GC does the physics."""
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session = uuid4(), uuid4()
    owned = await svc.create_owned_draft(user, "s.png", b"gc", "image/png")
    owned.status, owned.session_id = "bound", session
    asset_id = owned.asset_id
    sha = (await assets.get(asset_id)).object_sha256

    await svc.delete_for_session(session)
    # Logical delete only: tombstone present, object still on the CAS.
    assert await assets.get(asset_id) is not None
    assert sha in objects.rows

    result = await svc.gc(grace_seconds=0)

    assert result["assets_reaped"] == 1
    assert result["objects"]["collected"] == 1
    assert await assets.get(asset_id) is None   # tombstone reaped
    assert sha not in objects.rows              # object + file collected


async def test_delete_message_vs_detach_message_are_distinct(tmp_path):
    """delete_assets=True ⇒ logical delete; detach (edit) ⇒ link re-pinned, asset kept."""
    svc, objects, assets, attachments = make_service(tmp_path)
    user, session = uuid4(), uuid4()

    # Path 1: real delete → asset soft-deleted.
    del_msg = uuid4()
    owned_del = await svc.create_owned_draft(user, "a.png", b"a", "image/png")
    owned_del.status, owned_del.session_id, owned_del.message_id = "bound", session, del_msg
    await svc.delete_for_message(del_msg)
    assert (await assets.get(owned_del.asset_id)).deleted_at is not None

    # Path 2: edit/regenerate (detach) → asset untouched, link re-pinned to session.
    edit_msg = uuid4()
    owned_edit = await svc.create_owned_draft(user, "b.png", b"b", "image/png")
    owned_edit.status, owned_edit.session_id, owned_edit.message_id = "bound", session, edit_msg
    await svc.detach_message(edit_msg)
    assert (await assets.get(owned_edit.asset_id)).deleted_at is None
    assert attachments.rows[owned_edit.id].message_id is None
    assert attachments.rows[owned_edit.id].session_id == session


# ── viewer frame/roi draft resolution (§7) ─────────────────────────────────────────

async def test_draft_ids_for_assets_scoped_to_caller_and_draft(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, other = uuid4(), uuid4()

    mine = await svc.create_owned_draft(user, "f.png", b"f", "image/png", source="frame")
    bound = await svc.create_owned_draft(user, "b.png", b"b", "image/png")
    bound.status = "bound"
    theirs = await svc.create_owned_draft(other, "t.png", b"t", "image/png")

    ids = await svc.draft_ids_for_assets(
        user, [mine.asset_id, bound.asset_id, theirs.asset_id, uuid4()]
    )

    assert ids == [mine.id]


async def test_read_owned_bytes_is_owner_only_and_chat_scope(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user, other = uuid4(), uuid4()

    row = await svc.create_owned_draft(user, "s.png", b"inline", "image/png")
    assert await svc.read_owned_bytes(user, row.asset_id) == ("image/png", b"inline")
    assert await svc.read_owned_bytes(other, row.asset_id) is None  # foreign owner

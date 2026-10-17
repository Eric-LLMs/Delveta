"""``attach_note`` authorization regression (design §4 / §5.1).

A message-level attach comes in two ownership shapes and each must hit the *right*
authorization check — the defect this guards against is the owned shape being run through
the drive-only check, which reports 404 for every ``scope='chat'`` asset by design and so
aborted the whole turn before any model call:

  * ``owned=True``  — an upload / screenshot / viewer page region capture: a ``scope='chat'``
    asset owned by the caller. Authorized as a drive file OR an owned chat asset.
  * ``owned=False`` — a ``＋``-picked / viewer-open drive reference: authorized through the
    drive ACL, never through the chat-asset owner shortcut.

Fakes mirror the SQL repository surface (no DB), the same style as ``test_chat_assets.py``:
CAS + asset rows come from the drive fakes and ``DriveService`` runs unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.application.chat.context import attach_note
from tests._drive_fakes import make_drive


def _body(asset_id, *, owned, name="probe.png", mime_type="image/png"):
    return SimpleNamespace(
        attach={
            "kind": "asset",
            "asset_id": str(asset_id),
            "owned": owned,
            "name": name,
            "mime_type": mime_type,
        }
    )


# ── owned chat-scoped asset (the fixed path) ────────────────────────────────────────

async def test_owned_chat_asset_is_authorized_for_its_owner(tmp_path):
    """The whole point of the fix: an owned ``scope='chat'`` asset the caller uploaded must
    produce a note, not a 403 — the drive plane can never see it (§5.1)."""
    drive = make_drive(tmp_path)
    owner = uuid4()
    asset = await drive.assets.create(owner, "shot.png", scope="chat", file_status="READY")

    note = await attach_note(_body(asset.id, owned=True), drive, owner, inline=False)

    assert note is not None
    assert note.startswith("[Attached:")
    assert str(asset.id) in note
    # text-only chat model + image → the ``vision`` tool path.
    assert "vision" in note


async def test_owned_chat_asset_from_foreign_user_is_refused(tmp_path):
    """Owner-only: another user's draft/pending chat asset must not be readable."""
    drive = make_drive(tmp_path)
    owner, other = uuid4(), uuid4()
    asset = await drive.assets.create(owner, "shot.png", scope="chat", file_status="READY")

    with pytest.raises(PermissionError):
        await attach_note(_body(asset.id, owned=True), drive, other, inline=False)


async def test_owned_deleted_chat_asset_is_refused(tmp_path):
    """A soft-deleted chat asset is gone even to its owner (mirrors ``read_owned_bytes``)."""
    drive = make_drive(tmp_path)
    owner = uuid4()
    asset = await drive.assets.create(owner, "shot.png", scope="chat", file_status="READY")
    await drive.assets.soft_delete(asset.id)

    with pytest.raises(PermissionError):
        await attach_note(_body(asset.id, owned=True), drive, owner, inline=False)


async def test_owned_missing_asset_is_refused(tmp_path):
    drive = make_drive(tmp_path)
    with pytest.raises(PermissionError):
        await attach_note(_body(uuid4(), owned=True), drive, uuid4(), inline=False)


async def test_owned_drive_file_still_resolves(tmp_path):
    """Legacy shape kept: an owned attach that is actually a readable drive file still works."""
    drive = make_drive(tmp_path)
    owner = uuid4()
    asset = await drive.assets.create(owner, "doc.pdf", scope="drive", file_status="READY")

    note = await attach_note(
        _body(asset.id, owned=True, name="doc.pdf", mime_type="application/pdf"),
        drive, owner, inline=False,
    )

    assert note is not None and "read_document" in note


# ── reference attach: the drive ACL is the ONLY correct check ───────────────────────

async def test_reference_drive_asset_readable_by_owner(tmp_path):
    drive = make_drive(tmp_path)
    owner = uuid4()
    asset = await drive.assets.create(owner, "doc.pdf", scope="drive", file_status="READY")

    note = await attach_note(
        _body(asset.id, owned=False, name="doc.pdf", mime_type="application/pdf"),
        drive, owner, inline=False,
    )

    assert note is not None and note.startswith("[Attached:")


async def test_reference_drive_asset_from_foreign_user_is_refused(tmp_path):
    """A reference is a shared drive asset: no read access ⇒ refused, and it must NOT slip
    through the chat-asset owner shortcut (the asset is drive-scoped anyway)."""
    drive = make_drive(tmp_path)
    owner, other = uuid4(), uuid4()
    asset = await drive.assets.create(owner, "doc.pdf", scope="drive", file_status="READY")

    with pytest.raises(PermissionError):
        await attach_note(_body(asset.id, owned=False), drive, other, inline=False)


async def test_reference_chat_scoped_asset_is_refused(tmp_path):
    """A ``reference`` pointing at a chat-scoped asset is not a drive file → refused (the
    caller could not have picked it from the drive plane)."""
    drive = make_drive(tmp_path)
    owner = uuid4()
    asset = await drive.assets.create(owner, "shot.png", scope="chat", file_status="READY")

    with pytest.raises(PermissionError):
        await attach_note(_body(asset.id, owned=False), drive, owner, inline=False)


# ── inline hint shape ───────────────────────────────────────────────────────────────

async def test_owned_chat_image_inline_hint_forbids_vision_tool(tmp_path):
    """When the routed model is vision-capable the pixels ride inline, so the note must tell
    the model to answer from the embedded image and NOT call the ``vision`` tool."""
    drive = make_drive(tmp_path)
    owner = uuid4()
    asset = await drive.assets.create(owner, "shot.png", scope="chat", file_status="READY")

    note = await attach_note(_body(asset.id, owned=True), drive, owner, inline=True)

    assert note is not None
    assert "INLINE" in note
    assert "Do NOT call the `vision` tool" in note


async def test_non_asset_attach_yields_no_note(tmp_path):
    drive = make_drive(tmp_path)
    user = uuid4()
    body = SimpleNamespace(attach={"kind": "something_else", "asset_id": str(uuid4())})
    assert await attach_note(body, drive, user, inline=False) is None

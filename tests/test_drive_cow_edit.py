"""Drive text edit — copy-on-write semantics (rule 2026-09-28, cap-edit-file drive plane).

Exclusive asset → in-place (same asset_id). Asset referenced by another reader →
the original row and bytes are NEVER touched: a successor asset carries the edit,
the owner's reference switches to it, grantees keep resolving the original.
"""
import hashlib
from uuid import UUID, uuid4

import pytest

from core.application.drive_service import READY, DriveError
from core.infrastructure.storage import object_key
from tests._drive_fakes import make_drive

ORIGINAL = "# Scope\n\noriginal body line\n"


async def _seed_note(svc, owner, name="scope.md", content=ORIGINAL, folder=None):
    data = content.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    await svc.storage.put(object_key(digest), data)
    await svc.objects.upsert_and_increment(digest, len(data), object_key(digest),
                                           "text/markdown")
    return await svc.assets.create(owner, name, folder_path=folder,
                                   mime_type="text/markdown", size=len(data),
                                   object_sha256=digest,
                                   file_status=READY, rag_status="NOT_STARTED")


async def test_exclusive_edit_is_in_place(tmp_path):
    svc = make_drive(tmp_path)
    owner = uuid4()
    a = await _seed_note(svc, owner)

    res = await svc.edit_text(owner, a.id, old_text="original body line",
                              new_text="edited body line")
    assert res["mode"] == "in_place"
    assert res["asset"]["id"] == str(a.id)
    text = await svc.read_text(owner, a.id)
    assert "edited body line" in text and "original body line" not in text
    # path resolution still lands on the same asset
    found = await svc.resolve_personal_path(owner, "My Drive/scope.md")
    assert found["asset_id"] == str(a.id)


async def test_shared_edit_is_copy_on_write(tmp_path):
    svc = make_drive(tmp_path)
    owner, other = uuid4(), uuid4()
    a = await _seed_note(svc, owner, folder="IntentTest")
    await svc.acl.grant(a.id, other, "read")
    original_sha = a.object_sha256

    res = await svc.edit_text(owner, a.id, old_text="original body",
                              new_text="edited body")
    assert res["mode"] == "cow"
    new_id = res["asset"]["id"]
    assert new_id != str(a.id)

    # 1. the owner now sees the EDITED content
    assert "edited body" in await svc.read_text(owner, UUID(new_id))
    # the owner's reference switched: same path resolves to the successor
    found = await svc.resolve_personal_path(owner, "My Drive/IntentTest/scope.md")
    assert found["asset_id"] == new_id
    # the retired original left the owner's listing
    listed = {o["id"] for o in await svc.list_files(owner)}
    assert new_id in listed and str(a.id) not in listed

    # 2. the other reader still sees the ORIGINAL content, at the original asset
    still = await svc.read_text(other, a.id)
    assert still == ORIGINAL

    # 3. the original row points at untouched bytes (same sha, same object)
    fresh = await svc.assets.get(a.id)
    assert fresh.object_sha256 == original_sha
    assert (await svc.storage.get(object_key(original_sha))).decode() == ORIGINAL
    assert (fresh.meta or {}).get("cow_retired_by") == new_id
    # and A/B do not share one mutable asset any more
    succ = await svc.assets.get(UUID(new_id))
    assert succ.user_id == owner
    assert succ.object_sha256 != original_sha


async def test_grantee_edit_does_not_mutate_original(tmp_path):
    svc = make_drive(tmp_path)
    owner, grantee = uuid4(), uuid4()
    a = await _seed_note(svc, owner)
    await svc.acl.grant(a.id, grantee, "write")

    res = await svc.edit_text(grantee, a.id, old_text="original", new_text="mine")
    assert res["mode"] == "cow"
    # the grantee's successor is editor-owned, at the grantee's own root
    succ = await svc.assets.get(UUID(res["asset"]["id"]))
    assert succ.user_id == grantee and succ.folder_path is None
    # the original is byte-identical and carries no retirement mark for anyone else
    assert await svc.read_text(owner, a.id) == ORIGINAL
    orig = await svc.assets.get(a.id)
    assert not (orig.meta or {}).get("cow_retired_by")


async def test_edit_requires_existing_old_text(tmp_path):
    svc = make_drive(tmp_path)
    owner = uuid4()
    a = await _seed_note(svc, owner)
    with pytest.raises(DriveError):
        await svc.edit_text(owner, a.id, old_text="no such line", new_text="x")
    # nothing changed
    assert await svc.read_text(owner, a.id) == ORIGINAL

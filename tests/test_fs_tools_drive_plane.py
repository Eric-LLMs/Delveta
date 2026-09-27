"""fs tools plane dispatch (rule 2026-09-28): read_file / edit_file address ONE
capability across two planes — "My Drive/<path>" resolves to the real Drive asset
(no workspace materialization), anything else stays workspace-rooted.
"""
import hashlib
import uuid

import pytest

from agent.tools import fs_tools
from agent.tools.fs_tools import edit_file_tool, read_file_tool
from core.application.drive_service import READY
from core.infrastructure.storage import object_key
from tests._drive_fakes import make_drive

ORIGINAL = "# Scope\n\nline one\n"


async def _seed_note(svc, owner):
    data = ORIGINAL.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    await svc.storage.put(object_key(digest), data)
    await svc.objects.upsert_and_increment(digest, len(data), object_key(digest),
                                           "text/markdown")
    return await svc.assets.create(owner, "scope.md", mime_type="text/markdown",
                                   size=len(data), object_sha256=digest,
                                   file_status=READY, rag_status="NOT_STARTED")


@pytest.fixture
def as_user(monkeypatch):
    def _set(uid):
        monkeypatch.setattr(fs_tools, "get_request_user_id", lambda: uid)
    return _set


async def test_read_file_dispatches_to_drive(tmp_path, as_user):
    svc = make_drive(tmp_path)
    owner = uuid.uuid4()
    as_user(owner)
    await _seed_note(svc, owner)
    (tmp_path / "local.md").write_text("local content", encoding="utf-8")

    tool = read_file_tool(tmp_path, svc)
    drive_text = await tool.execute({"path": "My Drive/scope.md"}, None)
    assert drive_text == ORIGINAL                      # real asset bytes
    local_text = await tool.execute({"path": "local.md"}, None)
    assert local_text == "local content"               # workspace plane untouched


async def test_edit_file_drive_plane_edits_real_asset(tmp_path, as_user):
    svc = make_drive(tmp_path)
    owner = uuid.uuid4()
    as_user(owner)
    a = await _seed_note(svc, owner)

    tool = edit_file_tool(tmp_path, svc)
    out = await tool.execute({"path": "My Drive/scope.md",
                              "old_text": "line one", "new_text": "line TWO"}, None)
    assert "in place" in out and str(a.id) in out
    assert "line TWO" in await svc.read_text(owner, a.id)
    # NO workspace materialization: the drive plane has no local file for the asset
    assert not any(p.name.startswith("scope.md") for p in tmp_path.iterdir())


async def test_edit_file_drive_shared_asset_reports_cow(tmp_path, as_user):
    svc = make_drive(tmp_path)
    owner, other = uuid.uuid4(), uuid.uuid4()
    as_user(owner)
    a = await _seed_note(svc, owner)
    await svc.acl.grant(a.id, other, "read")

    tool = edit_file_tool(tmp_path, svc)
    out = await tool.execute({"path": "My Drive/scope.md",
                              "old_text": "line one", "new_text": "cow line"}, None)
    assert "new copy" in out
    assert "unaffected" in out
    assert await svc.read_text(other, a.id) == ORIGINAL


async def test_unresolvable_drive_path_is_an_error_not_a_local_fallback(tmp_path, as_user):
    svc = make_drive(tmp_path)
    as_user(uuid.uuid4())
    tool = read_file_tool(tmp_path, svc)
    with pytest.raises(ValueError):
        await tool.execute({"path": "My Drive/ghost.md"}, None)
    # no drive backend wired → drive paths never silently touch the workspace
    bare = read_file_tool(tmp_path, None)
    with pytest.raises(ValueError):
        await bare.execute({"path": "My Drive/scope.md"}, None)

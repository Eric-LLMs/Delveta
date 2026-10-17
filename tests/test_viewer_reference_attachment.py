"""Viewer → Message → Chat Attachments (chat_assets design §6.3).

**Open ≠ Inject**: a viewer payload rides on every turn while a file is open, but it must
NOT become a chat attachment by itself. Only a turn that explicitly names the on-screen
document mounts a ``reference`` link to ``viewer.asset_id``, deduped against an explicit
``attach`` so one drive file can never yield two rows. These tests pin the predicate, the
mount, the dedupe and the abandon/retry lifecycle; the persist→restore round trip against a
real database lives in ``tests/test_chat_assets_integration.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.viewer_context import requests_viewer_document
from core.application.chat.context import _viewer_reference_ids
from tests.test_chat_assets import make_service


def _viewer(**kw):
    base = {
        "follow": True, "mode": None, "asset_id": uuid4(),
        "name": "on-screen.pdf", "kind": "pdf", "provenance": "cloud",
    }
    base.update(kw)
    return SimpleNamespace(**base)


# ── the predicate: explicit on-screen reference, or nothing ────────────────────────

@pytest.mark.parametrize("message,want", [
    ("总结本页", True),                 # local deixis (本页)
    ("解释这张表", True),               # table deixis — the named example
    ("这张图里有什么", True),           # figure deixis
    ("这一段是什么意思", True),         # passage deixis
    ("这篇文章讲了什么", True),         # whole-document scope
    ("summarize this page", True),      # English deixis
    ("第3页的公式推导一下", True),      # explicit page number
    ("写一个 FastAPI 服务", False),     # task phrasing → not about the doc
    ("今天天气怎么样", False),          # unrelated, no deictic/scope
    ("为什么推荐用 Redis", False),      # bare interrogative is NOT an explicit doc request
])
def test_requests_viewer_document(message, want):
    assert requests_viewer_document(message, _viewer()) is want


def test_open_is_not_inject_requires_a_tracked_viewer_with_a_cloud_asset():
    """Merely having a file open (or a closed focus chip, or a local file) never attaches."""
    assert requests_viewer_document("总结本页", None) is False
    assert requests_viewer_document("总结本页", _viewer(follow=False)) is False
    assert requests_viewer_document("总结本页", _viewer(mode="none")) is False
    assert requests_viewer_document("总结本页", _viewer(asset_id=None)) is False


# ── the mount: reference, deduped, permission-checked ──────────────────────────────

async def test_explicit_reference_mounts_a_draft_that_binds_to_the_message(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user = uuid4()
    doc = await assets.create(user, "on-screen.pdf", scope="drive", file_status="READY")

    ids = await _viewer_reference_ids(svc, user, doc.id, None)
    assert len(ids) == 1
    row = attachments.rows[ids[0]]
    assert row.ownership == "reference"       # never a copy, never owned
    assert row.source == "viewer"
    assert row.asset_id == doc.id
    assert row.user_id == user
    assert row.status == "draft"              # bound later, in the message-INSERT tx
    assert row.session_id is None and row.message_id is None

    # Send binds it in the same transaction that inserts the user message (§6.2).
    session_id, message_id = uuid4(), uuid4()
    bound = await attachments.bind(ids, user, session_id, message_id)
    assert [b.status for b in bound] == ["bound"]
    assert bound[0].session_id == session_id
    assert bound[0].message_id == message_id
    assert bound[0].expires_at is None


async def test_same_asset_from_viewer_and_explicit_attach_yields_one_link(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user = uuid4()
    doc = await assets.create(user, "on-screen.pdf", scope="drive", file_status="READY")

    # The explicit ``attach`` path already mounted a reference for this very asset id, so the
    # viewer branch must mount NOTHING — otherwise the reopen view would show two rows.
    ids = await _viewer_reference_ids(svc, user, doc.id, {"asset_id": str(doc.id)})
    assert ids == []
    assert [r for r in attachments.rows.values() if r.asset_id == doc.id] == []


async def test_non_drive_cross_user_and_missing_assets_are_never_mounted(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    owner, other = uuid4(), uuid4()
    chat_asset = await assets.create(owner, "c.png", scope="chat", file_status="READY")
    drive_asset = await assets.create(owner, "d.pdf", scope="drive", file_status="READY")

    assert await _viewer_reference_ids(svc, owner, chat_asset.id, None) == []   # non-drive
    assert await _viewer_reference_ids(svc, other, drive_asset.id, None) == []  # cross-user
    assert await _viewer_reference_ids(svc, owner, uuid4(), None) == []         # missing
    assert attachments.rows == {}


# ── abandon / retry: drafts only, never a bound orphan or a duplicate message ──────

async def test_abandoned_retry_leaves_only_a_draft_never_a_bound_orphan(tmp_path):
    svc, objects, assets, attachments = make_service(tmp_path)
    user = uuid4()
    doc = await assets.create(user, "on-screen.pdf", scope="drive", file_status="READY")

    # A failed send and then a retry each mount a reference draft…
    first = await _viewer_reference_ids(svc, user, doc.id, None)
    second = await _viewer_reference_ids(svc, user, doc.id, None)
    assert len(first) == 1 and len(second) == 1 and first != second

    # …but only the message that actually persists binds ITS draft.
    session_id, message_id = uuid4(), uuid4()
    assert len(await attachments.bind(second, user, session_id, message_id)) == 1
    abandoned = attachments.rows[first[0]]
    assert abandoned.status == "draft" and abandoned.message_id is None  # no bound orphan

    # Draft GC reaps the abandoned one; a reference link frees no drive asset (§8).
    reaped = await svc.draft_gc(now=datetime.now(timezone.utc) + timedelta(days=2))
    assert reaped["expired"] == 1
    assert first[0] not in attachments.rows
    assert await assets.get(doc.id) is not None                            # drive file intact

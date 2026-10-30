"""Chat-session management + agent-operations routes: list/read/rename/delete sessions,
delete a single message, resolve a human-in-the-loop approval, revert a workspace checkpoint.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from uuid import UUID

from agent.security.approvals import get_approval_bridge
from agent.tools.checkpoints import CheckpointError
from api.auth import AuthUser, require_user
from api.deps import get_agent, get_drive_service
from api.routers._shared import byte_stream as _stream
from api.routers._shared import disposition_header as _disposition
from api.schemas import ApprovalResolveRequest, ReconcileRequest, SessionRenameRequest
from api.session_history_cursor import CursorError, decode_cursor, encode_cursor
from api.session_list_cursor import (
    SessionCursorError,
    decode_session_cursor,
    encode_session_cursor,
)
from core.application.chat_asset_service import ChatAssetService
from core.application.drive_service import DriveService
from core.config import settings
from core.infrastructure.db import ChunkModel, MessageModel, SessionLocal, SessionModel
from core.infrastructure.memory import (
    list_sessions_by_ids,
    list_sessions_page,
    load_session_detail,
    reconcile_messages,
)
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select, update

router = APIRouter(tags=["sessions"])

# Chat-asset uploads are screenshots / video frames / roi crops — small PNGs — so a bounded
# single-shot body is enough (no chunked upload). The cap stops an accidental large file from
# being buffered whole; source is closed to the four shapes the design names (§6, §7).
_MAX_CHAT_ATTACHMENT_BYTES = 32 * 1024 * 1024
_CHAT_ATTACHMENT_SOURCES = {"upload", "screenshot", "frame", "roi"}


async def _backfill_imported_rag(session_factory, user_id: UUID, session_id: UUID) -> None:
    """One-time on-read conversion: derive ``imported_rag`` flags from pre-flag chat chunks.

    Before the per-message flag existed, imports were tracked only in chunk meta: single-pair
    chunks keyed by the user-message id (``kind='qa'`` without ``covered``), whole-session
    chunks carrying ``meta['covered']``, and the even-older ``kind='session-qa'`` blanket.
    Reconstructing the flags here lets an already-imported session keep its persistent
    "✓ Imported" state right after the upgrade, without forcing a re-import. If every current
    user message is covered (or a legacy blanket chunk exists) the whole transcript is in the
    repo, so every message gets flagged; otherwise only the covered user messages do (a
    regenerated assistant reply has no chunk identity to recover, and clicking its button is
    an idempotent re-import). Idempotent — a message already flagged stays flagged.
    """
    async with session_factory() as session:
        chunk_rows = (
            await session.execute(
                select(ChunkModel.source_id, ChunkModel.meta).where(
                    ChunkModel.source_type == "chat",
                    ChunkModel.user_id == user_id,
                    ChunkModel.meta["session_id"].astext == str(session_id),
                )
            )
        ).all()
    covered: set[str] = set()
    legacy_blanket = False
    for source_id, meta in chunk_rows:
        meta = meta or {}
        kind = meta.get("kind")
        if kind == "session-qa":
            legacy_blanket = True
        elif kind == "qa":
            cov = meta.get("covered")
            if isinstance(cov, list) and cov:
                covered.update(str(c) for c in cov)
            else:
                covered.add(source_id)  # legacy single-pair: source_id is the user-message id
    # Nothing was ever imported into the repo: return BEFORE reading the session's message
    # rows. History is paged now, and a session-wide read here would undo that for every
    # open (this is the common case — most sessions have no chat chunks).
    if not covered and not legacy_blanket:
        return
    async with session_factory() as session:
        msg_rows = (
            await session.execute(
                select(MessageModel.id, MessageModel.role, MessageModel.imported_rag)
                .where(MessageModel.session_id == session_id)
            )
        ).all()
    already = {str(mid) for mid, _, flag in msg_rows if flag}
    user_ids = [str(mid) for mid, role, _ in msg_rows if role == "user"]
    if legacy_blanket or covered.issuperset(user_ids):
        candidates = [str(mid) for mid, _, _ in msg_rows]  # every message is in the repo
    else:
        candidates = [mid for mid in covered if mid in user_ids]  # covered user messages only
    flag_ids = [mid for mid in candidates if mid not in already]
    if not flag_ids:
        return
    async with session_factory() as session:
        await session.execute(
            update(MessageModel).where(MessageModel.id.in_(flag_ids)).values(imported_rag=True)
        )
        await session.commit()


_MAX_SESSION_IDS = 50


def _parse_session_ids(raw: str) -> list[UUID]:
    """Parse ``ids=a,b,c`` (deduped, order kept); a bad or oversized list is a 400."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[UUID] = []
    seen: set[UUID] = set()
    for part in parts:
        try:
            sid = UUID(part)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="ids must be comma-separated UUIDs"
            ) from None
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    if not out:
        raise HTTPException(status_code=400, detail="ids must not be empty")
    if len(out) > _MAX_SESSION_IDS:
        raise HTTPException(
            status_code=400, detail=f"at most {_MAX_SESSION_IDS} ids per request"
        )
    return out


@router.get("/sessions")
async def get_sessions(
    user: AuthUser = Depends(require_user),
    q: str | None = None,
    kind: str = "chat",
    limit: int = Query(50, ge=1, le=200),
    before: str | None = None,
    ids: str | None = None,
) -> dict:
    """List the authenticated user's sessions (newest first) for one surface.

    ``?kind=chat`` (default) lists ordinary chats; ``?kind=workspace`` lists WorkSpace
    assistant sessions. The two lanes are disjoint: the chat sidebar never shows a
    WorkSpace conversation and vice versa. Research sessions (type 1) belong to their
    task and are listed by neither lane. ``?q=`` filters by title / summary / message
    content (case-insensitive substring).

    Paged by default: ``limit`` (1–200) rows, oldest position echoed as ``next_before``,
    and ``has_more`` says whether an older page exists. ``?before=`` walks strictly
    older. ``?ids=a,b,c`` (≤50) is the pinned-session batch: it returns exactly those of
    the caller's sessions in this lane and ignores paging; ids the caller does not own
    are simply absent.
    """
    if kind not in ("chat", "workspace"):
        raise HTTPException(status_code=400, detail="kind must be 'chat' or 'workspace'")
    q = (q or "").strip()
    # The lane filter already ran in SQL; re-assert it at the route boundary so a session of
    # another kind can never leak into this list (research-session isolation contract).
    lane_type = 0 if kind == "chat" else 2

    if ids is not None:
        if before:
            raise HTTPException(
                status_code=400, detail="ids and before cannot be combined"
            )
        rows = await list_sessions_by_ids(
            SessionLocal, user.user_id, _parse_session_ids(ids), kind=kind
        )
        return {
            "sessions": [s for s in rows if s.get("type") == lane_type],
            "has_more": False,
            "next_before": None,
        }

    position = None
    if before:
        try:
            position = decode_session_cursor(before)
        except SessionCursorError:
            raise HTTPException(status_code=400, detail="invalid cursor") from None

    sessions, has_more = await list_sessions_page(
        SessionLocal, user.user_id, q or None, kind=kind, limit=limit, before=position
    )
    sessions = [s for s in sessions if s.get("type") == lane_type]
    next_before = None
    if has_more and sessions:
        oldest = sessions[-1]
        next_before = encode_session_cursor(
            (datetime.fromisoformat(oldest["created_at"]), UUID(oldest["id"]))
        )
    return {"sessions": sessions, "has_more": has_more, "next_before": next_before}


@router.post("/sessions/attachments")
async def create_chat_attachment(
    request: Request,
    name: str,
    user: AuthUser = Depends(require_user),
    drive: DriveService = Depends(get_drive_service),
    mime_type: str | None = None,
    source: str = "upload",
    locator: str | None = None,
) -> dict:
    """Upload a chat-only draft asset (design §5.2, §6.1): screenshot / frame / roi / file.

    The bytes land in the shared CAS as a ``scope='chat'`` asset that the drive plane can
    never see; a ``draft`` ``chat_attachments`` row (TTL-bounded) links it to the caller. The
    Send that follows binds it to the session + message. ``locator`` is the optional
    ``{t_ms,x,y,w,h}`` (frame/roi) or ``{page}`` (document) JSON, stored verbatim.
    """
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty attachment")
    if len(data) > _MAX_CHAT_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="attachment too large")
    if source not in _CHAT_ATTACHMENT_SOURCES:
        raise HTTPException(status_code=400, detail="unknown attachment source")
    loc: dict | None = None
    if locator:
        try:
            loc = json.loads(locator)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="locator must be JSON")
        if not isinstance(loc, dict):
            raise HTTPException(status_code=400, detail="locator must be a JSON object")
    safe_name = (name or "").strip()[:255] or "attachment"
    row = await ChatAssetService(SessionLocal, storage=drive.storage).create_owned_draft(
        user.user_id, safe_name, data, mime_type, source=source, locator=loc,
    )
    return {
        "attachment_id": str(row.id),
        "asset_id": str(row.asset_id),
        "name": safe_name,
        "mime_type": mime_type,
        "size": len(data),
    }


@router.get("/sessions/{session_id}/attachments/{attachment_id}/content")
async def download_chat_attachment(
    session_id: UUID,
    attachment_id: UUID,
    user: AuthUser = Depends(require_user),
    drive: DriveService = Depends(get_drive_service),
):
    """Stream a chat attachment's bytes (design §5.2).

    Authorizes on **session ownership** first (404 for a foreign/missing session), then goes
    through :meth:`ChatAssetService.read_bytes` — the §4 double check that also re-verifies a
    ``reference`` link's drive ACL at read time. No drive-plane route ever serves a
    ``scope='chat'`` asset, and this one never resolves an id outside its own session.
    """
    async with SessionLocal() as session:
        sess = (
            await session.execute(select(SessionModel).where(SessionModel.id == session_id))
        ).scalar_one_or_none()
    if sess is None or sess.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="session not found")
    chat_assets = ChatAssetService(SessionLocal, storage=drive.storage)
    data = await chat_assets.read_bytes(attachment_id, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="attachment not found")
    att = await chat_assets.get_for_session(attachment_id, session_id)
    asset = await chat_assets.assets.get(att.asset_id) if att is not None else None
    mime = (asset.mime_type if asset else None) or "application/octet-stream"
    filename = (asset.name if asset else None) or "attachment"
    return StreamingResponse(
        _stream(data),
        media_type=mime,
        headers={"Content-Disposition": _disposition(filename), "Content-Length": str(len(data))},
    )


@router.get("/sessions/{session_id}")
async def get_session_messages(
    session_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    before: str | None = Query(None),
    user: AuthUser = Depends(require_user),
):
    """Return a session's title + one page of messages (with ids) for the current user.

    History is paged newest-first: the default page is the latest 50 rows, and ``before``
    (the ``next_before`` of the previous page) walks further back. ``has_more`` says whether
    an older page exists; ``next_before`` is the cursor to request it. Paging happens in the
    database (keyset ``LIMIT``), never by loading the session and slicing it here.
    """
    async with SessionLocal() as session:
        sess = (
            await session.execute(select(SessionModel).where(SessionModel.id == session_id))
        ).scalar_one_or_none()
    if sess is None or sess.user_id != user.user_id:
        raise HTTPException(status_code=404, detail="session not found")
    cursor = None
    if before is not None:
        try:
            cursor = decode_cursor(before)
        except CursorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Recover pre-flag imported state onto the message rows so the response below carries it.
    # First page only: the flags are persisted, and the newest page is always requested
    # first, so every later page already reads correct flags without another backfill pass.
    if before is None:
        await _backfill_imported_rag(SessionLocal, user.user_id, session_id)
    detail = await load_session_detail(SessionLocal, session_id, limit=limit, before=cursor)
    # ``compaction`` is the durable checkpoint inlined for Live-State RECOVERY: a client
    # that lost its in-memory state rebuilds [summary]+[tail] from here, then returns to
    # zero-read turns. Recovery is its only purpose — normal chat never reads it.
    return {
        "session_id": str(session_id),
        "title": detail["title"],
        "type": sess.type,
        # Descriptive scope (WorkSpace sessions); ``{}`` for chat/research sessions.
        "meta": sess.meta or {},
        "compaction": detail["compaction"],
        "messages": detail["messages"],
        "has_more": detail["has_more"],
        "next_before": encode_cursor(detail["next_before"]) if detail["has_more"] else None,
        # The session's first message, sent only when the title is unset (a paged client
        # cannot derive the fallback heading from the newest page alone).
        "first_message": detail["first_message"],
    }


@router.patch("/sessions/{session_id}")
async def rename_session(
    session_id: UUID, body: SessionRenameRequest, user: AuthUser = Depends(require_user)
):
    """Rename a session and/or refresh its descriptive scope metadata.

    An empty title resets it to ``NULL`` so auto-naming kicks in again. ``meta`` is merged
    key-by-key into ``sessions.meta`` — a key sent as ``null`` is stored as ``null`` (e.g.
    "this session is no longer pinned to a folder"), omitted keys are left untouched.
    """
    async with SessionLocal() as session:
        sess = (
            await session.execute(select(SessionModel).where(SessionModel.id == session_id))
        ).scalar_one_or_none()
        if sess is None or sess.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="session not found")
        # ``null`` title keeps its historical meaning ("reset to auto-naming"); only a
        # payload that omits ``title`` leaves the current name alone.
        if "title" in body.model_fields_set:
            title = (body.title or "").strip()
            sess.title = title or None
        if body.meta is not None:
            # Reassign (not in-place mutate) so SQLAlchemy sees a dirty JSONB value.
            sess.meta = {**(sess.meta or {}), **body.meta}
        await session.commit()
        return {"title": sess.title, "meta": sess.meta or {}}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    user: AuthUser = Depends(require_user),
    drive: DriveService = Depends(get_drive_service),
):
    """Delete a session (cascade removes its messages + events) if it belongs to the user.

    The session's chat attachments (``chat_attachments``) are torn down first: an ``owned``
    asset is freed (its ref-count share of the CAS object released), a ``reference`` link to a
    drive file is only dropped — the drive file is never touched. The stable ``RAG/images``
    copy an imported Q&A references is a separate asset row this delete never touches.
    """
    async with SessionLocal() as session:
        sess = (
            await session.execute(select(SessionModel).where(SessionModel.id == session_id))
        ).scalar_one_or_none()
        if sess is None or sess.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="session not found")
    chat_assets = ChatAssetService(SessionLocal, storage=drive.storage)
    await chat_assets.delete_for_session(session_id)
    async with SessionLocal() as session:
        sess = await session.get(SessionModel, session_id)
        if sess is not None:
            await session.delete(sess)
            await session.commit()
    return Response(status_code=204)


@router.delete("/sessions/{session_id}/messages/{message_id}")
async def delete_session_message(
    session_id: UUID,
    message_id: UUID,
    user: AuthUser = Depends(require_user),
    drive: DriveService = Depends(get_drive_service),
    delete_assets: bool = True,
):
    """Delete a single message (no truncation) if its session belongs to the user.

    Its chat attachments are resolved through ``chat_attachments``. When ``delete_assets`` is
    true (default) an ``owned`` asset is freed and the links dropped; when false (the client's
    edit-and-regenerate flow) the links are re-pinned to the session so the screenshot the
    question was built around survives (and dies with the session later), never orphaned. A
    ``reference`` link to a drive file is only ever dropped, never followed.
    """
    async with SessionLocal() as session:
        sess = (
            await session.execute(select(SessionModel).where(SessionModel.id == session_id))
        ).scalar_one_or_none()
        if sess is None or sess.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="session not found")
        message = (
            await session.execute(select(MessageModel).where(MessageModel.id == message_id))
        ).scalar_one_or_none()
        if message is None or message.session_id != session_id:
            raise HTTPException(status_code=404, detail="message not found")
    chat_assets = ChatAssetService(SessionLocal, storage=drive.storage)
    if delete_assets:
        await chat_assets.delete_for_message(message_id)
    else:
        await chat_assets.detach_message(message_id)
    async with SessionLocal() as session:
        message = await session.get(MessageModel, message_id)
        if message is not None:
            await session.delete(message)
            await session.commit()
    return Response(status_code=204)


@router.post("/sessions/{session_id}/reconcile")
async def reconcile_session(
    session_id: UUID,
    body: ReconcileRequest,
    user: AuthUser = Depends(require_user),
):
    """Align SQL to the client's authoritative Live State (RECONNECT/RECOVERY ONLY).

    Compensation for failed async Edit/Delete syncs and writes that never landed while
    the client was offline — not part of any normal turn. Rows after the checkpoint
    boundary (INCLUSIVE watermark) are diffed against the uploaded tail: client-missing
    → delete; SQL-missing (incl. foreign/forged ids — rejected as ids, inserted under
    fresh ones) → insert; text mismatch → client's text wins, ``embedding`` cleared for
    re-embed. Message-id ownership is enforced here (this path reads SQL anyway).
    """
    async with SessionLocal() as session:
        sess = (
            await session.execute(select(SessionModel).where(SessionModel.id == session_id))
        ).scalar_one_or_none()
        if sess is None or sess.user_id != user.user_id:
            raise HTTPException(status_code=404, detail="session not found")
    tail = [
        {
            "message_id": str(t.message_id) if t.message_id else None,
            "role": t.role,
            "content": t.content,
        }
        for t in body.tail
    ]
    counts = await reconcile_messages(SessionLocal, user.user_id, session_id, tail)
    return {"ok": True, **counts}


@router.post("/approvals/{approval_id}")
async def resolve_approval(
    approval_id: str,
    body: ApprovalResolveRequest,
    user: AuthUser = Depends(require_user),
):
    """Resolve a pending human-in-the-loop approval (allow / deny).

    Approvals are registered under the requesting user's id when their chat store is built, so
    ownership is checked before resolving — one user can't approve another's tool call. The
    shared broker resolves the store's pending future (cross-node via Redis pub/sub when
    configured), which unblocks the agent turn awaiting it.
    """
    bridge = get_approval_bridge()
    owner = await bridge.owner(approval_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="approval not found or already resolved")
    if str(user.user_id) != owner:
        raise HTTPException(status_code=403, detail="approval belongs to another user")
    await bridge.resolve(approval_id, body.allow, (body.message or "")[:500] or None)
    return {"ok": True, "allow": body.allow}


@router.post("/checkpoints/{checkpoint_id}/revert")
async def revert_checkpoint(checkpoint_id: str, user: AuthUser = Depends(require_user)):
    """Restore the agent workspace to a prior shadow-git checkpoint.

    Checkpoint ids are recorded before each chat turn (see the agent kernel); this endpoint
    rolls the workspace back to a known-good state after a bad batch of agent file edits.
    """
    agent = get_agent()
    if not settings.checkpoint_enabled:
        # Automatic snapshots are off by default (checkpoint_enabled=false), so no
        # checkpoint id exists. Answer 410 — a retired capability, not a transient outage.
        raise HTTPException(
            status_code=410,
            detail="workspace checkpoints are disabled (checkpoint_enabled=false)",
        )
    if agent.checkpoints is None:
        raise HTTPException(status_code=503, detail="workspace checkpoints are disabled")
    try:
        head = await asyncio.to_thread(agent.checkpoints.revert, checkpoint_id)
    except CheckpointError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "head": head}

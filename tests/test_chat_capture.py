"""Funnel-gated deferred client capture.

The order this module pins end to end:

1. an ordinary turn NEVER captures (no viewer / plain text -> no capture request);
2. the Intent Funnel decides the capability FIRST — only a certified ``vision`` turn
   with no server-readable picture produces a ``capture_request``;
3. the orchestrator pauses the stream, emits ``capture-request``, and resumes the
   SAME plan once the client uploads the page/frame (no re-funnel, no re-extraction);
4. a failed/empty capture is a recoverable ``capture-failed`` state (retry);
5. on success the vision tool makes exactly ONE content-understanding call carrying
   the picture, the locally prepared text and the question together.
"""
from __future__ import annotations

import hashlib
import json
import types
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agent import Context
from agent.engine.context import AgentTurn, _TURN_CTX
from agent.engine.decisions import ToolExecution
from agent.engine.runtime import ToolRuntime
from agent.security.approvals import MemoryApprovalBroker
from api.schemas import ViewerPayload
from api.viewer_context import (
    build_viewer_blocks,
    render_viewer_reference,
    validate_viewer_citations,
    viewer_assembly_with_capture,
    viewer_assembly_with_transcript,
    viewer_citation_map,
    viewer_snapshot,
)
from core.application.chat.capture import (
    STATUS_OK,
    CaptureBroker,
    merge_payload,
    resume_plan,
)
from core.application.chat.context import ChatTurnContext
from core.application.chat.execution_plan import ExecutionPlan, PlanKind
from core.application.chat.executors.base import ViewerDeps
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel.cap_handler import VisionHandler
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    MatchResult,
    TurnFacts,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.turn_orchestrator import TurnOrchestrator
from core.application.chat.understanding import TurnRequirements
from core.config import settings
from core.infrastructure.request_context import request_user
from core.infrastructure.storage import LocalStorage, object_key

from apps.api.tools import vision_tool
from tests._drive_fakes import make_drive

MESSAGE = "这一页的图表讲了什么"
VIEWER_DEPS = ViewerDeps(
    build_blocks=build_viewer_blocks,
    validate_citations=validate_viewer_citations,
    citation_map=viewer_citation_map,
    snapshot=viewer_snapshot,
    render_reference=render_viewer_reference,
    with_capture=viewer_assembly_with_capture,
    with_transcript=viewer_assembly_with_transcript,
)


# ── 1. The capture decision: only a viewer-sourced picture ────────────────────────


def _facts(**kw) -> TurnFacts:
    base = dict(has_viewer=True, viewer_kind="pdf", viewer_provenance="local",
                viewer_name="local.pdf", viewer_current_page=3)
    base.update(kw)
    return TurnFacts(**base)


def test_local_page_needs_a_client_capture():
    cap = VisionHandler().capture_for(query=MESSAGE, facts=_facts())
    assert cap is not None
    assert cap.request()["kind"] == "page"
    assert cap.request()["page"] == 3
    assert cap.request()["question"] == MESSAGE


def test_local_image_needs_a_client_capture():
    cap = VisionHandler().capture_for(
        query=MESSAGE, facts=_facts(viewer_kind="image", viewer_current_page=None))
    assert cap is not None and cap.kind == "image"


def test_video_needs_the_playhead_frame_even_when_cloud_bound():
    cap = VisionHandler().capture_for(
        query=MESSAGE,
        facts=_facts(viewer_kind="video", viewer_provenance="cloud",
                     viewer_asset_id="cloud-video", viewer_t_ms=75_000,
                     viewer_duration_ms=90_000),
    )
    assert cap is not None
    assert (cap.kind, cap.t_ms, cap.duration_ms) == ("frame", 75_000, 90_000)


def test_cloud_document_is_read_server_side_never_captured():
    assert VisionHandler().capture_for(
        query=MESSAGE, facts=_facts(viewer_provenance="cloud")
    ) is None


def test_no_viewer_plain_text_never_captures():
    handler = VisionHandler()
    assert handler.capture_for(query="今天天气怎么样", facts=None) is None
    assert handler.capture_for(
        query="今天天气怎么样", facts=TurnFacts(has_viewer=False)) is None


def test_an_already_uploaded_picture_wins_over_capture():
    # A frame/roi selection is already an owned capture: the tool reads it directly.
    assert VisionHandler().capture_for(
        query=MESSAGE, facts=_facts(viewer_kind="video", viewer_t_ms=1000,
                                    attachment_asset_id="owned-frame")
    ) is None


def test_video_captures_without_a_pre_sent_playhead():
    # Deferred-data client: no timestamp rides the request. The client resolves the live
    # playhead when it answers the capture-request, so the capture is still legal.
    cap = VisionHandler().capture_for(
        query=MESSAGE, facts=_facts(viewer_kind="video", viewer_t_ms=None)
    )
    assert cap is not None and cap.kind == "frame" and cap.t_ms is None


# ── 2. Funnel wiring: certified vision → capture; other turns unchanged ─────────


def _vision_entry() -> CapabilityEntry:
    q = "分析这张图片"
    return CapabilityEntry(
        capability_id="cap-vision", tool_binding="vision",
        description="analyze an attached image",
        standard_queries=(QueryRecord(id="cap-vision-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "asset_id": {"type": "string", "max_len": 64, "required": True},
            "question": {"type": "string", "required": False},
        },
    )


def _ctx(message: str, viewer):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _local_pdf_viewer():
    return SimpleNamespace(
        asset_id=None, page=3, page_from=None, page_to=None, selections=[],
        provenance="local", kind="pdf", name="local.pdf", t_ms=None, duration_ms=None,
    )


def _wire(monkeypatch, *, hit: bool):
    entry = _vision_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return (MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                            matched_literal="分析这张图片")
                if hit else MatchResult(state="MISS"))

    class _NoSelect:
        async def select(self, query, candidates, *, entries_by_id, facts=None):
            raise AssertionError("the selector must never run on this lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: _NoSelect())
    return entry, view


async def test_vision_turn_without_a_server_asset_is_certified_as_deferred_capture(
        monkeypatch):
    _entry, view = _wire(monkeypatch, hit=True)
    from core.application.chat.intent_funnel import funnel
    deps = types.SimpleNamespace(session_factory=None, embedder=lambda: object(),
                                 llm=object(), acquisition_inputs=None)

    out = await funnel.route(
        _ctx(MESSAGE, _local_pdf_viewer()), deps=deps,
        requirements=TurnRequirements())

    assert out.requested_action["tool"] == "vision"
    # capability/tool/kind are already decided; only the picture is still missing
    assert out.requested_action["funnel_stage"] == "capture"
    assert out.requested_action["args"] == {"question": MESSAGE}
    assert out.requested_action["funnel_registry_version"] == view.fingerprint
    assert out.capture_request == {
        "kind": "page", "name": "local.pdf", "page": 3,
        "t_ms": None, "duration_ms": None, "question": MESSAGE,
    }


async def test_ordinary_turn_gets_no_capture_request(monkeypatch):
    _entry, _view = _wire(monkeypatch, hit=False)
    from core.application.chat.intent_funnel import funnel

    async def no_index(**kw):
        return None

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.recall.load_index", no_index)
    deps = types.SimpleNamespace(session_factory=None, embedder=lambda: object(),
                                 llm=object(), acquisition_inputs=None)
    req = TurnRequirements()

    out = await funnel.route(
        _ctx("今天天气怎么样", _local_pdf_viewer()), deps=deps, requirements=req)

    assert out is req and out.capture_request is None


def test_plain_chat_plan_carries_no_capture():
    from core.application.chat.execution_plan import PolicyContext, build_execution_plan

    plan = build_execution_plan(TurnRequirements(), PolicyContext())
    assert plan.kind is PlanKind.AGENT and plan.capture_request is None


# ── 3. Resume is a pure patch of the ALREADY certified plan ──────────────────────


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        kind=PlanKind.ACTION,
        action={"tool": "vision", "args": {"question": MESSAGE},
                "capability_id": "cap-vision", "funnel_registry_version": "fp"},
        capture_request={"kind": "page", "name": "local.pdf", "page": 3,
                         "t_ms": None, "duration_ms": None, "question": MESSAGE},
    )


def test_resume_plan_injects_the_asset_without_re_routing():
    plan = _plan()
    resumed = resume_plan(plan, {"status": "ok", "asset_id": "asset-1",
                                 "text": "page text"})
    assert resumed is not plan
    assert resumed.action["capability_id"] == "cap-vision"      # same certification
    assert resumed.action["args"] == {"question": MESSAGE, "asset_id": "asset-1"}
    assert resumed.capture_request is None                      # cannot pause again
    assert plan.action["args"] == {"question": MESSAGE}          # original untouched


def test_merge_payload_keeps_the_server_spec_and_client_truth():
    merged = merge_payload(_plan().capture_request,
                           {"status": "ok", "asset_id": "a", "t_ms": None})
    assert merged["kind"] == "page" and merged["page"] == 3
    assert merged["asset_id"] == "a" and merged["status"] == "ok"


# ── 4. Assembly: the capture joins the SAME turn's viewer context ────────────────


def _assembly(kind="pdf"):
    viewer = ViewerPayload(name="local.pdf", kind=kind, provenance="local", page=3)
    return build_viewer_blocks(viewer, MESSAGE)


def test_page_capture_adds_image_and_feeds_image_text():
    asm = viewer_assembly_with_capture(_assembly(), {
        "kind": "page", "name": "local.pdf", "page": 3,
        "asset_id": "img-1", "text": "PREPARED PAGE TEXT",
    })
    assert asm["status"] == "injected"
    image = next(b for b in asm["blocks"] if b.image_asset_id == "img-1")
    text = next(b for b in asm["blocks"] if b.feeds_image)
    assert image.kind == "roi" and image.locator == {"page": 3}
    assert text.text == "PREPARED PAGE TEXT" and text.locator == {"page": 3}
    # The prepared text is NOT injected into the main prompt (one request only).
    assert "PREPARED PAGE TEXT" not in render_viewer_reference(asm["blocks"])


def test_frame_capture_carries_the_subtitle_window():
    asm = viewer_assembly_with_capture(_assembly("video"), {
        "kind": "frame", "name": "local.mp4", "t_ms": 75_000,
        "asset_id": "img-2", "text": "subtitles ±5s",
    })
    image = next(b for b in asm["blocks"] if b.image_asset_id == "img-2")
    text = next(b for b in asm["blocks"] if b.feeds_image)
    assert image.kind == "frame" and image.locator == {"t_ms": 75_000}
    assert text.kind == "subtitle_window" and text.text == "subtitles ±5s"


def test_capture_without_a_pre_funnel_assembly_still_binds():
    # The deferred-data client sends no viewer content, so build_viewer_blocks has
    # nothing to assemble. The capture is the turn's only viewer input and must bind.
    asm = viewer_assembly_with_capture(None, {
        "kind": "frame", "name": "local.mp4", "t_ms": 42_000,
        "asset_id": "img-9", "text": "subtitles",
    })
    assert asm is not None and asm["status"] == "injected" and asm["mode"] == "focus"
    assert any(b.image_asset_id == "img-9" for b in asm["blocks"])


def test_snapshot_recovers_identity_from_the_bound_capture():
    # The request carried no name/page/playhead; the persisted snapshot must still show
    # what was actually used (the capture's live identity), never the current viewer.
    asm = viewer_assembly_with_capture(None, {
        "kind": "frame", "name": "local.mp4", "t_ms": 42_000,
        "asset_id": "img-9", "text": "subtitles",
    })
    snap = viewer_snapshot(
        ViewerPayload(kind="video", provenance="local"), asm["blocks"])
    assert snap["name"] == "local.mp4" and snap["t_ms"] == 42_000
    assert any(b["image_asset_id"] == "img-9" for b in snap["injected"])


def test_minimal_viewer_descriptor_binds_no_content():
    # kind + provenance only (the deferred-data client): no blocks, no abort, no content.
    v = ViewerPayload(kind="video", provenance="local")
    asm = build_viewer_blocks(v, "视频中这个图片什么意思")
    assert asm["status"] == "none" and asm["blocks"] == []
    assert "cues" not in v.model_dump(exclude_none=True)


def test_retry_replaces_the_previous_capture_block():
    asm = viewer_assembly_with_capture(_assembly(), {
        "kind": "page", "name": "local.pdf", "page": 3,
        "asset_id": "img-1", "text": "old",
    })
    asm = viewer_assembly_with_capture(asm, {
        "kind": "page", "name": "local.pdf", "page": 3,
        "asset_id": "img-2", "text": "new",
    })
    images = [b for b in asm["blocks"] if b.image_asset_id]
    assert [b.image_asset_id for b in images] == ["img-2"]


# ── 5. Orchestrator: pause / fail / retry / resume ───────────────────────────────


class _Mem:
    def __init__(self):
        self.attachments: list[str] = []
        self.persist_failed = False

    async def flush_writes(self):
        return []

    def take_written(self):
        return []

    def add_attachment_ids(self, ids):
        self.attachments.extend(str(i) for i in ids)


class _Queue:
    def __init__(self):
        self.jobs: list[tuple] = []

    async def enqueue(self, *a, **k):
        self.jobs.append((a, k))


class _Executor:
    kind = PlanKind.ACTION

    def __init__(self):
        self.plans: list[ExecutionPlan] = []

    async def stream(self, req, *, progress_sink):
        self.plans.append(req.plan)
        yield {"type": "content", "data": "CAPTURE ANSWER"}
        yield {"type": "done", "data": {
            "answer": "CAPTURE ANSWER", "messages": [], "usage": {}}}

    async def run(self, req):  # pragma: no cover - stream lane only
        raise AssertionError("run is not used by /chat/stream")


def _capture_ctx() -> ChatTurnContext:
    body = SimpleNamespace(message=MESSAGE, viewer=ViewerPayload(
        name="local.pdf", kind="pdf", provenance="local", page=3))
    assembly = build_viewer_blocks(body.viewer, MESSAGE)
    return ChatTurnContext(
        body=body, user=None, user_id=uuid4(), guest_token=None, log_user=None,
        tier="free", notice=None, model=None, base_url=None, api_key=None,
        business_name=None, credential_id=None,
        user_text=MESSAGE, owned_asset_id=None, inline_image=None,
        viewer_assembly=assembly, agent_context={"viewer": assembly},
        session_id=uuid4(), session_memory=_Mem(),
    )


def _capture_deps(broker) -> SimpleNamespace:
    async def _persist(*a, **k):
        return None

    from core.infrastructure.jobs import SESSION_FINALIZE  # noqa: F401

    return SimpleNamespace(
        capture_broker=broker,
        viewer=VIEWER_DEPS,
        queue=_Queue(),
        new_approval_bridge=lambda: SimpleNamespace(broker=MemoryApprovalBroker()),
        persist_turn_meta=_persist,
        log_usage=_persist,
        agent=SimpleNamespace(runtime=None),
        run_tool=None,
    )


async def _drive(monkeypatch, ctx, deps, on_request):
    """Run stream_turn, answering each capture-request through ``on_request``."""
    orch = TurnOrchestrator(deps)
    plan = _plan()

    async def fake_resolve(self, ctx, deps=None):
        return plan

    executor = _Executor()

    def fake_executor(self, p):
        return executor

    monkeypatch.setattr(TurnOrchestrator, "resolve_plan", fake_resolve)
    monkeypatch.setattr(TurnOrchestrator, "executor_for", fake_executor)

    frames = []
    gen = orch.stream_turn(ctx, t_entry=0.0)
    try:
        async for frame in gen:
            frames.append(json.loads(frame["data"]))
            if frames[-1]["type"] == "capture-request":
                on_request(frames[-1]["data"], deps.capture_broker)
    finally:
        await gen.aclose()
    return frames, executor


async def test_successful_capture_resumes_the_same_turn_once(monkeypatch):
    ctx, broker = _capture_ctx(), CaptureBroker()
    deps = _capture_deps(broker)

    def answer(request, b):
        assert b.submit(request["ticket_id"], user_id=ctx.user_id, payload={
            "status": "ok", "asset_id": "img-1", "attachment_id": str(uuid4()),
            "text": "PREPARED PAGE TEXT"}) == STATUS_OK

    frames, executor = await _drive(monkeypatch, ctx, deps, answer)
    kinds = [f["type"] for f in frames]

    assert kinds == ["capture-request", "content", "done"]
    # the SAME certified turn executed exactly once, with the captured asset
    assert len(executor.plans) == 1
    assert executor.plans[0].action["args"]["asset_id"] == "img-1"
    assert executor.plans[0].capture_request is None
    # the resumed turn's context carries the picture + prepared text
    blocks = ctx.agent_context["viewer"]["blocks"]
    assert any(getattr(b, "image_asset_id", None) == "img-1" for b in blocks)
    assert ctx.session_memory.attachments  # draft bound to this turn's user message
    assert frames[-1]["data"]["answer"] == "CAPTURE ANSWER"


async def test_failed_capture_is_recoverable_and_retried(monkeypatch):
    ctx, broker = _capture_ctx(), CaptureBroker()
    deps = _capture_deps(broker)
    seen: list[dict] = []

    def answer(request, b):
        seen.append(request)
        if len(seen) == 1:
            b.submit(request["ticket_id"], user_id=ctx.user_id,
                     payload={"status": "failed", "reason": "upload_failed"})
        else:
            b.submit(request["ticket_id"], user_id=ctx.user_id, payload={
                "status": "ok", "asset_id": "img-ok", "text": "text"})

    frames, executor = await _drive(monkeypatch, ctx, deps, answer)
    kinds = [f["type"] for f in frames]

    assert kinds == ["capture-request", "capture-failed", "capture-request",
                     "content", "done"]
    failed = frames[1]["data"]
    assert failed["retryable"] is True and failed["attempt"] == 1
    assert seen[0]["ticket_id"] != seen[1]["ticket_id"]  # fresh ticket per attempt
    assert len(executor.plans) == 1
    assert executor.plans[0].action["args"]["asset_id"] == "img-ok"


async def test_capture_timeout_never_calls_the_tool(monkeypatch):
    monkeypatch.setattr(settings, "chat_capture_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "chat_capture_max_attempts", 1)
    ctx, broker = _capture_ctx(), CaptureBroker()
    deps = _capture_deps(broker)

    frames, executor = await _drive(monkeypatch, ctx, deps, lambda *a: None)
    kinds = [f["type"] for f in frames]

    assert kinds == ["capture-request", "capture-failed", "done"]
    failed = frames[1]["data"]
    assert failed["reason"] == "capture_timeout" and failed["retryable"] is False
    done = frames[-1]["data"]
    assert done["answer"] is None and done["capture"] == {
        "status": "failed", "retryable": True}
    assert executor.plans == []          # no tool, no model, no execution


async def test_plan_without_capture_never_pauses(monkeypatch):
    ctx = _capture_ctx()
    deps = _capture_deps(CaptureBroker())
    monkeypatch.setattr(TurnOrchestrator, "resolve_plan",
                        _no_capture_resolve)
    monkeypatch.setattr(TurnOrchestrator, "executor_for",
                        lambda self, p: _Executor())

    frames = [json.loads(f["data"]) async for f in
              TurnOrchestrator(deps).stream_turn(ctx, t_entry=0.0)]
    assert [f["type"] for f in frames] == ["content", "done"]


async def _no_capture_resolve(self, ctx, deps=None):
    return replace(_plan(), capture_request=None)


# ── 5b. Deferred full transcript (video FULL, Agent lane) ───────────────────────

FULL_Q = "视频中讲了什么,这个字幕什么意思"


def _transcript_ctx() -> ChatTurnContext:
    body = SimpleNamespace(message=FULL_Q, viewer=ViewerPayload(
        kind="video", provenance="local"))
    assembly = build_viewer_blocks(body.viewer, body.message)
    assert assembly["status"] == "transcript_deferred"
    return ChatTurnContext(
        body=body, user=None, user_id=uuid4(), guest_token=None, log_user=None,
        tier="free", notice=None, model=None, base_url=None, api_key=None,
        business_name=None, credential_id=None,
        user_text=FULL_Q, owned_asset_id=None, inline_image=None,
        viewer_assembly=assembly, agent_context=None,
        session_id=uuid4(), session_memory=_Mem(),
    )


class _AgentStream:
    kind = PlanKind.AGENT

    def __init__(self):
        self.plans: list[ExecutionPlan] = []

    async def stream(self, req, *, progress_sink):
        self.plans.append(req.plan)
        yield {"type": "content", "data": "VIDEO SUMMARY"}
        yield {"type": "done", "data": {
            "answer": "VIDEO SUMMARY", "messages": [], "usage": {}}}


async def _drive_agent(monkeypatch, ctx, deps, on_request):
    """Run stream_turn on an AGENT plan, answering transcript-request frames."""
    orch = TurnOrchestrator(deps)
    plan = ExecutionPlan(kind=PlanKind.AGENT)

    async def fake_resolve(self, ctx, deps=None):
        return plan

    executor = _AgentStream()
    monkeypatch.setattr(TurnOrchestrator, "resolve_plan", fake_resolve)
    monkeypatch.setattr(TurnOrchestrator, "executor_for",
                        lambda self, p: executor)

    frames = []
    gen = orch.stream_turn(ctx, t_entry=0.0)
    try:
        async for frame in gen:
            frames.append(json.loads(frame["data"]))
            if frames[-1]["type"] == "transcript-request":
                on_request(frames[-1]["data"], deps.capture_broker)
    finally:
        await gen.aclose()
    return frames, executor


async def test_full_video_asks_for_the_transcript_then_resumes_the_same_turn(monkeypatch):
    ctx, broker = _transcript_ctx(), CaptureBroker()
    deps = _capture_deps(broker)

    def answer(request, b):
        assert request["kind"] == "transcript" and request["question"] == FULL_Q
        assert b.submit(request["ticket_id"], user_id=ctx.user_id, payload={
            "status": "ok", "full_text": "line one\nline two",
            "full_trusted": True}) == STATUS_OK

    frames, executor = await _drive_agent(monkeypatch, ctx, deps, answer)
    kinds = [f["type"] for f in frames]

    assert kinds == ["transcript-request", "content", "done"]
    assert len(executor.plans) == 1                    # one Agent run, same turn
    viewer = ctx.agent_context["viewer"]
    assert viewer["status"] == "injected"
    full = next(b for b in viewer["blocks"] if b.kind == "full_subtitles")
    assert full.text == "line one\nline two"
    assert frames[-1]["data"]["answer"] == "VIDEO SUMMARY"


async def test_missing_transcript_ends_honestly_without_a_model_call(monkeypatch):
    monkeypatch.setattr(settings, "chat_capture_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "chat_capture_max_attempts", 1)
    ctx, broker = _transcript_ctx(), CaptureBroker()
    deps = _capture_deps(broker)

    frames, executor = await _drive_agent(monkeypatch, ctx, deps, lambda *a: None)
    kinds = [f["type"] for f in frames]

    assert kinds == ["transcript-request", "transcript-failed", "viewer", "done"]
    assert frames[1]["data"]["reason"] == "transcript_timeout"
    assert frames[2]["data"]["status"] == "unavailable"
    assert frames[-1]["data"]["answer"] is None
    assert frames[-1]["data"]["transcript"] == {"status": "failed", "retryable": True}
    assert executor.plans == []                        # no model, no session write


async def test_non_agent_plan_never_asks_for_a_transcript(monkeypatch):
    # An ACTION plan (e.g. vision) executes without needing the reference transcript.
    ctx = _transcript_ctx()
    deps = _capture_deps(CaptureBroker())
    executor = _Executor()
    monkeypatch.setattr(TurnOrchestrator, "resolve_plan", _no_capture_resolve)
    monkeypatch.setattr(TurnOrchestrator, "executor_for",
                        lambda self, p: executor)

    frames = [json.loads(f["data"]) async for f in
              TurnOrchestrator(deps).stream_turn(ctx, t_entry=0.0)]
    assert [f["type"] for f in frames] == ["content", "done"]
    assert len(executor.plans) == 1


# ── 6. One content-understanding call: picture + text + question together ───────


async def test_vision_makes_one_call_with_picture_text_and_question(monkeypatch, tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"\x55" * 32
    storage = LocalStorage(tmp_path)
    sha = hashlib.sha256(png).hexdigest()
    await storage.put(object_key(sha), png)
    drive = make_drive(tmp_path)
    uid = uuid4()
    asset = await drive.assets.create(uid, "page_3.png", object_sha256=sha, scope="chat")
    monkeypatch.setattr(vision_tool, "DriveService", lambda sf: drive)
    request_user.set(uid)

    calls: list[dict] = []

    async def fake_describe(data, mime="", *, llm, session_factory, prompt=None,
                            user_id=None, role_id=None):
        calls.append({"data": data, "prompt": prompt})
        return "FINAL ANSWER"

    monkeypatch.setattr(vision_tool, "describe_image", fake_describe)
    ctx = Context()
    ctx.provide("storage", storage)
    ctx.provide("session_factory", object)
    runtime = ToolRuntime()
    vision_tool.register(runtime, ctx, llm=None)

    assembly = viewer_assembly_with_capture(_assembly(), {
        "kind": "page", "name": "local.pdf", "page": 3,
        "asset_id": str(asset.id), "text": "PREPARED PAGE TEXT",
    })
    token = _TURN_CTX.set(AgentTurn(user_msg=MESSAGE,
                                    context={"viewer": assembly}))
    try:
        res = await runtime.execute(ToolExecution(
            "c1", "vision", {"asset_id": str(asset.id), "question": MESSAGE}))
    finally:
        _TURN_CTX.reset(token)

    assert res.is_error is False and str(res.value) == "FINAL ANSWER"
    assert len(calls) == 1                                  # exactly ONE call
    assert calls[0]["data"] == png                          # the captured picture
    assert "PREPARED PAGE TEXT" in calls[0]["prompt"]       # locally prepared text
    assert MESSAGE in calls[0]["prompt"]                    # the original question

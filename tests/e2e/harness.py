"""Real-chain E2E harness (Phase 4-A / A-1).

Assembles ONE stack in which a certified ACTION rides the REAL control plane all
the way into a real tool body:

    httpx/ASGI -> /chat/stream -> TurnOrchestrator.resolve_plan -> intent_funnel.route
      -> REAL Matcher (HIT on an injected Registry view) -> REAL _acquisition_hop
         (per-capability handler) -> REAL Binder -> REAL ActionExecutor._dispatch
      -> REAL chat._run_tool -> REAL ToolRuntime (sandbox / approval) -> REAL tool body

Only the OUTER world is faked: the Registry *read* (a hand-built
``RegistryLiveView`` — a DB-I/O seam, the same class as faking the DB), the LLM
port (``ScriptedPort`` for the funnel/Agent; a SEPARATE ``ToolLLM`` for a tool's own
external-LLM call), the retrieval / web / vision / drive / vocabulary / storage
seams, and the DB (memory fake, inherited from the p5 harness).

Every effect is observed through ``EffectRecorder`` on the runtime's
``tools/result`` event — a channel DECOUPLED from the funnel trace, so a turn that
"reports success" but never reaches a real tool body fails the anti-fake guard.

Terminology (locked): this asserts a real, DETERMINISTIC EFFECT at the tool
boundary (真实 Tool Body 执行 + 确定性效果验证). It does NOT persist to a real
database — that is A-2 Live's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from api.tools import pdf_tools as pdf_mod
from api.tools import rag_search_tool as rag_mod
from api.tools import read_document_tool as read_document_mod
from api.tools import translate_tool as translate_mod
from api.tools import vision_tool as vision_mod
from api.tools import web_search_tool as web_search_mod
from core.application.chat.intent_funnel import registry as registry_pkg
from core.application.chat.intent_funnel.registry import (
    STATUS_ACTIVE,
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    content_fingerprint,
    derive_language,
)
from core.config import settings

from tests.p5_validation._p5_harness import (
    USER,
    FakeSeam,
    ScriptedPort,
    Spy,
    build_app,
    build_kernel,
    domains_named,
    sse,
)

# One terminal Agent step: it emits content and NO tool call, so the ReAct loop
# breaks after exactly one LLM step (the "escalation happened" signal).
STEP = {"content": ["unused"], "tool_calls": None}


# ── decoupled tool-internal LLM double ─────────────────────────────────────────────
class ToolLLM:
    """The fake ``llm`` handed to tool REGISTRATION (translate's `complete`).

    Deliberately SEPARATE from ``ScriptedPort``: a tool's own external-LLM call is
    a legitimate part of the real tool body and must NOT be visible to the funnel /
    Agent zero-LLM counters. Keeping the two objects distinct is what lets a test
    assert "the funnel/Agent spent zero model calls" AND "the translate tool really
    called its model" at the same time, with no contradiction.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system_prompt: str = "") -> str:
        self.calls.append((prompt, system_prompt))
        return "已翻译"


# ── storage / asset seams ──────────────────────────────────────────────────────────
class FakeStorage:
    """Object-storage seam: returns canned bytes and records every key read."""

    def __init__(self, data: bytes = b"e2e-bytes") -> None:
        self.data = data
        self.keys: list[str] = []

    async def get(self, key: str) -> bytes:
        self.keys.append(key)
        return self.data


@dataclass
class FakeAsset:
    """A drive-asset row double (only the fields the tool bodies read)."""

    id: str
    name: str
    mime_type: str
    object_sha256: str = "e2e-sha"
    user_id: Any = USER
    workspace_id: Any = None


def _asset_repo(asset: FakeAsset):
    """Stand-in for ``SqlAssetRepository`` — ``get(asset_id)`` returns the canned row."""

    class _Repo:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def get(self, asset_id):
            return asset

    return _Repo


def _read_drive(asset: FakeAsset):
    """Stand-in for the read-side ``DriveService`` — readability always passes."""

    class _Drive:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def ensure_asset_readable(self, user_id, asset_id):
            return asset

    return _Drive


class VisionSeam:
    """Stand-in for ``vision_caption.describe_image`` — records the call, no model."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def describe(self, data, mime, *, llm=None, session_factory=None,
                       prompt="", user_id=None):
        self.calls.append({"mime": mime, "prompt": prompt, "user_id": str(user_id)})
        return "[vision analysis]"


class ReadDocSeam:
    """Stand-in for ``ingest.extract_document_text`` — records the name, no parse."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def extract(self, data, name, llm=None):
        self.calls.append(name)
        return "[document text]"


class FakePdfLib:
    """Stand-in for ``core.infrastructure.pdf`` — deterministic, no PyMuPDF."""

    def __init__(self) -> None:
        self.extract_calls: list[Any] = []
        self.table_calls: list[Any] = []
        self.transcribe_calls: list[str] = []

    def extract_pdf_text(self, data, pages=None):
        self.extract_calls.append(pages)
        return "[pdf text]"

    def detect_tables(self, data, pages=None):
        self.table_calls.append(pages)
        return [b"png-table"]

    async def pdf_table_to_text(self, png, llm, prompt):
        self.transcribe_calls.append(prompt)
        return "[table text]"


# ── effect recorder (anti-fake guard) ──────────────────────────────────────────────
class EffectRecorder:
    """Observes the runtime's ``tools/result`` event — the REAL tool-body boundary.

    Fires for every executed (or failed/denied) tool, independent of the funnel
    trace. A test asserts ``call_count`` / ``recorded_tool`` / ``recorded_args`` so a
    turn that is merely *reported* certified but never reaches a tool body fails.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def observe(self, runtime) -> None:
        runtime.events.observe("tools/result", self._on_result)

    def _on_result(self, payload: dict) -> None:
        ex = payload["exec"]
        result = payload["result"]
        is_error = bool(result.is_error)
        self.calls.append({
            "tool": ex.name,
            "args": dict(ex.arguments or {}),
            "is_error": is_error,
            "value": None if is_error else getattr(result, "value", None),
        })

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def recorded_tool(self) -> str | None:
        return self.calls[-1]["tool"] if self.calls else None

    @property
    def recorded_args(self) -> dict | None:
        return self.calls[-1]["args"] if self.calls else None

    @property
    def recorded_value(self) -> Any:
        return self.calls[-1]["value"] if self.calls else None


# ── capability matrix fixture ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Cap:
    """One capability row of the E2E matrix.

    ``source`` says how the turn supplies the asset the handler needs:
    ``none`` (query/name only), ``attach_image`` (vision), ``attach_pdf``
    (read_document), ``viewer_pdf`` (the pdf_* caps). ``extra`` are the fixed
    non-asset args the certified ACTION must carry.
    """

    cap_id: str
    tool: str
    message: str
    write: bool
    source: str = "none"
    needs_asset: bool = False
    query_arg: bool = False
    extra: tuple[tuple[str, str], ...] = ()


CAPABILITIES: tuple[Cap, ...] = (
    Cap("cap-create-folder", "create_folder",
        "please make a folder named gamma", write=True,
        extra=(("name", "gamma"),)),
    Cap("cap-add-term", "add_term",
        "add quantum to my physics terms", write=True,
        extra=(("term", "quantum"), ("domain", "physics terms"))),
    Cap("cap-web-search", "web_search",
        "search the web for the latest quantum error correction news", write=False,
        query_arg=True),
    Cap("cap-rag-search", "rag_search",
        "find learning material about gradient descent", write=False,
        query_arg=True),
    Cap("cap-translate", "translate",
        'translate "hello world" into Chinese', write=False,
        extra=(("text", "hello world"),)),
    Cap("cap-vision", "vision",
        "describe this image", write=False,
        source="attach_image", needs_asset=True,
        extra=(("question", "describe this image"),)),
    Cap("cap-read-document", "read_document",
        "summarize the attached document", write=False,
        source="attach_pdf", needs_asset=True),
    Cap("cap-pdf-extract-text", "pdf_extract_text",
        "pull the text out of this pdf", write=False,
        source="viewer_pdf", needs_asset=True),
    Cap("cap-pdf-table-to-text", "pdf_table_to_text",
        "pull the tables out of this pdf", write=False,
        source="viewer_pdf", needs_asset=True),
)

CAP_BY_ID: dict[str, Cap] = {c.cap_id: c for c in CAPABILITIES}


def expected_args(cap: Cap, asset_id: Any) -> dict:
    """The exact schema-filtered args the tool body must receive."""
    args: dict = {}
    if cap.needs_asset:
        args["asset_id"] = str(asset_id)
    if cap.query_arg:
        args["query"] = cap.message
    for key, value in cap.extra:
        args[key] = value
    return args


def turn_fields(cap: Cap, *, asset_id: Any = None,
                page_from: int | None = None, page_to: int | None = None) -> dict:
    """The ``/chat/stream`` body extras that supply this capability's context."""
    if cap.source == "attach_image":
        return {"attach": {"kind": "asset", "asset_id": str(asset_id),
                           "name": "shot.png", "mime_type": "image/png", "owned": True}}
    if cap.source == "attach_pdf":
        return {"attach": {"kind": "asset", "asset_id": str(asset_id),
                           "name": "report.pdf", "mime_type": "application/pdf",
                           "owned": True}}
    if cap.source == "viewer_pdf":
        viewer: dict = {"kind": "pdf", "name": "report.pdf",
                        "asset_id": str(asset_id), "follow": True}
        if page_from is not None:
            viewer["page_from"] = page_from
        if page_to is not None:
            viewer["page_to"] = page_to
        return {"viewer": viewer}
    return {}


# ── gate configuration ─────────────────────────────────────────────────────────────
def _set_gates(monkeypatch, *, backend: str = "stub") -> None:
    """Pin the routing gates: MATCH_HIT direct lane on, every fast path off."""
    monkeypatch.setattr(settings, "chat_direct_fast_path_enabled", False, raising=False)
    monkeypatch.setattr(settings, "chat_viewer_fast_path_enabled", False, raising=False)
    monkeypatch.setattr(settings, "chat_retrieval_fast_path_enabled", False, raising=False)
    monkeypatch.setattr(settings, "chat_composite_fast_path_enabled", False, raising=False)
    # "off" would keep the legacy HIT hop (ToolIntentModel); "stub" gives the
    # MATCH_HIT -> acquisition-hop direct lane the A-1 matrix asserts.
    monkeypatch.setattr(settings, "chat_cap_router_backend", backend, raising=False)
    monkeypatch.setattr(settings, "chat_funnel_trace_capture", False, raising=False)


# ── the assembled stack ────────────────────────────────────────────────────────────
@dataclass
class Stack:
    app: Any
    kernel: Any
    runtime: Any
    ctx: Any
    broker: Any
    port: ScriptedPort
    spy: Spy
    recorder: EffectRecorder
    tool_llm: ToolLLM
    storage: FakeStorage
    retrieval: FakeSeam
    vision_seam: VisionSeam
    read_doc_seam: ReadDocSeam
    pdf_seam: FakePdfLib
    view: RegistryLiveView
    entries: tuple
    image_asset: FakeAsset
    pdf_asset: FakeAsset


def _make_fake_active_view(view: RegistryLiveView):
    async def _fake_active_view(*, session_factory=None):
        return view

    return _fake_active_view


def build_stack(
    monkeypatch,
    *,
    broker_mode: str = "allow",
    drive_mode: str = "ok",
    web_status: str = "ok",
    grant=(),
    rules=(),
    query_overrides: dict[str, str] | None = None,
) -> Stack:
    """Compose the in-process real-chain stack (see module docstring)."""
    port = ScriptedPort(steps=[dict(STEP)])
    spy = Spy()
    tool_llm = ToolLLM()
    storage = FakeStorage()
    retrieval = FakeSeam(hits=[{"text": "learning material chunk"}])

    kernel, runtime, ctx, broker = build_kernel(
        monkeypatch, port, spy, broker_mode=broker_mode, drive_mode=drive_mode,
        domains=domains_named("physics terms"), web_status=web_status,
        grant=grant, rules=rules, audit_dir=".output/e2e",
    )
    # The real tool bodies resolve their seams from the kernel Context.
    ctx.provide("storage", storage)
    ctx.provide("retrieval", retrieval)

    # web_search's keyless SERP/crawl paths would hit the network — pin them off.
    async def _no_serps(query, top_k):
        return []

    async def _no_pages(results):
        return []

    monkeypatch.setattr(web_search_mod, "_scrape_serps", _no_serps)
    monkeypatch.setattr(web_search_mod, "_crawl_pages", _no_pages)

    # Asset-bearing tools resolve their asset row + bytes from these seams.
    image_asset = FakeAsset(id=str(uuid4()), name="shot.png", mime_type="image/png")
    pdf_asset = FakeAsset(id=str(uuid4()), name="report.pdf", mime_type="application/pdf")
    vision_seam, read_doc_seam, pdf_seam = VisionSeam(), ReadDocSeam(), FakePdfLib()

    monkeypatch.setattr(vision_mod, "SqlAssetRepository", _asset_repo(image_asset))
    monkeypatch.setattr(vision_mod, "_mime_for", lambda asset: "image/png")
    monkeypatch.setattr(vision_mod, "describe_image", vision_seam.describe)
    monkeypatch.setattr(read_document_mod, "DriveService", _read_drive(pdf_asset))
    monkeypatch.setattr(read_document_mod, "extract_document_text", read_doc_seam.extract)
    monkeypatch.setattr(pdf_mod, "SqlAssetRepository", _asset_repo(pdf_asset))
    monkeypatch.setattr(pdf_mod, "pdf_lib", pdf_seam)

    # Register the REAL tool bodies (create_folder/add_term/web_search were already
    # registered by build_kernel).
    rag_mod.register(runtime, ctx, None)
    translate_mod.register(runtime, ctx, tool_llm)
    vision_mod.register(runtime, ctx, None)
    read_document_mod.register(runtime, ctx, None)
    pdf_mod.register(runtime, ctx, None)

    recorder = EffectRecorder()
    recorder.observe(runtime)

    overrides = query_overrides or {}
    entries = tuple(
        CapabilityEntry(
            capability_id=cap.cap_id,
            tool_binding=cap.tool,
            description=f"e2e fixture for {cap.cap_id}",
            parameters=runtime.get(cap.tool).parameters,
            standard_queries=(
                QueryRecord(
                    id=f"q-{cap.cap_id}",
                    query=overrides.get(cap.cap_id, cap.message),
                    language=derive_language(overrides.get(cap.cap_id, cap.message)),
                ),
            ),
            enabled=True,
            status=STATUS_ACTIVE,
            intent_kind="action",
        )
        for cap in CAPABILITIES
    )
    view = RegistryLiveView(fingerprint=content_fingerprint(entries), entries=entries)

    # THE seam that forces every fixture cap through the MATCH_HIT direct lane:
    # both the cascade (orchestrator) and the executor's TOCTOU re-read import
    # ``active_view`` from the registry PACKAGE at call time, so one patch covers
    # both and returns the SAME fingerprint-stable view.
    monkeypatch.setattr(registry_pkg, "active_view", _make_fake_active_view(view))

    _set_gates(monkeypatch)
    app = build_app(monkeypatch, port, retrieval, kernel, broker)

    return Stack(
        app=app, kernel=kernel, runtime=runtime, ctx=ctx, broker=broker, port=port,
        spy=spy, recorder=recorder, tool_llm=tool_llm, storage=storage,
        retrieval=retrieval, vision_seam=vision_seam, read_doc_seam=read_doc_seam,
        pdf_seam=pdf_seam, view=view, entries=entries, image_asset=image_asset,
        pdf_asset=pdf_asset,
    )


async def sse_for(stack: Stack, message: str, **fields):
    """Drive one turn through the real /chat/stream route."""
    return await sse(stack.app, message, user=USER, **fields)

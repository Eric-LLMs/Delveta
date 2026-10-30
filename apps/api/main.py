"""FastAPI app: expose core use cases as REST/SSE.

Start: uvicorn api.main:app --reload
"""
import asyncio
import contextlib
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from agent.security.approvals import configure_approval_broker
from api.routers.admin import router as admin_router
from api.routers.auth import AVATAR_DIR
from api.routers.auth import router as auth_router
from api.routers.chat import router as chat_router
from api.routers.config import _bootstrap_config
from api.routers.config import router as config_router
from api.routers.drive import files as drive_files_router
from api.routers.drive import folders as drive_folders_router
from api.routers.drive import trash as drive_trash_router
from api.routers.drive import users as drive_users_router
from api.routers.drive import workspaces as drive_workspaces_router
from api.routers.jobs import router as jobs_router
from api.routers.observability import router as observability_router
from api.routers.rag_admin import router as rag_admin_router
from api.routers.registry_admin import router as registry_admin_router
from api.routers.research import research_validation_handler
from api.routers.research import router as research_router
from api.routers.sessions import router as sessions_router
from api.routers.vocab import router as vocab_router
from arq import create_pool
from arq.connections import RedisSettings
from core.application.drive_service import DriveError
from core.application.services import VocabError
from core.config import settings
from core.infrastructure.db import SessionLocal, init_db
from core.infrastructure.llm import aclose_llm_clients
from core.infrastructure.security import ensure_admin_user, ensure_default_admin
from core.logger import configure_logging, reset_log_context, set_log_context
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from rag.query_cache import bump_corpus_version, configure_query_cache

logger = logging.getLogger("uvicorn.error")


class RequestLogContext:
    """Pure-ASGI middleware: tag every HTTP request with a fresh ``request_id``.

    A pure ASGI wrapper (not BaseHTTPMiddleware) so the ``ContextVar`` stays set while a
    streaming response body is sent — the chat / research SSE generators and the approval
    tool handlers run inside this ``await``, so their log lines carry the request id. The
    context is reset only when the whole response (including the stream) has finished.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        tokens = set_log_context(request_id=uuid.uuid4().hex[:12])
        try:
            await self.app(scope, receive, send)
        finally:
            reset_log_context(tokens)


# ── tool-intent warm-up (O2) ───────────────────────────────────────────────────
# The extractor sidecar's health check proves the model is REGISTERED, not loaded.
# Loading happens on the first inference and is far slower than the per-call
# extraction timeout, so a cold process can time out a correct first turn. Warm it
# in the background at boot: the same endpoint, model tag and httpx mechanism the
# extractor itself uses, with its own generous timeout (the 4s extraction guardrail
# does not apply to loading). Ollama keeps the model resident via the sidecar's
# OLLAMA_KEEP_ALIVE=24h; no client, connection pool or config is added.
_WARMUP_TIMEOUT_SECONDS = 600.0
_warmup_task: asyncio.Task | None = None


async def _warm_tool_intent_model() -> None:
    url = (settings.chat_tool_intent_local_url or "").strip()
    model = (settings.chat_tool_intent_local_model or "").strip()
    if not url or not model:
        logger.info("tool-intent warmup skipped: no extractor endpoint configured")
        return
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=_WARMUP_TIMEOUT_SECONDS) as client:
            resp = await client.post(url.rstrip("/") + "/chat/completions", json=payload)
            resp.raise_for_status()
        logger.info("tool-intent warmup complete model=%s", model)
    except Exception:  # noqa: BLE001 — a failed warm-up must never affect startup
        logger.warning("tool-intent warmup failed model=%s", model, exc_info=True)


def _schedule_tool_intent_warmup() -> None:
    """Start ONE background warm-up task per process (never blocks startup)."""
    global _warmup_task
    if _warmup_task is not None and not _warmup_task.done():
        return
    _warmup_task = asyncio.create_task(_warm_tool_intent_model())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Route root loggers → logs/api.log (rotating) before anything logs.
    configure_logging(settings, app="api")
    if settings.bash_sandbox == "host":
        logger.warning(
            "bash_sandbox=\"host\": the agent bash tool runs directly on the host process. "
            "This is NOT a security boundary — use it for local dev only; the production "
            "default is \"docker\" (per-command container sandbox)."
        )
    if settings.jwt_secret == "change-me":
        if settings.enforce_secure_secrets:
            raise RuntimeError(
                "enforce_secure_secrets is enabled but JWT_SECRET is still the default "
                "'change-me'. Set a strong secret before starting in a secure environment."
            )
        logger.warning("JWT_SECRET is the default 'change-me' — set it for production.")
    await init_db()
    async with SessionLocal() as session:
        await ensure_default_admin(session)
        await ensure_admin_user(session)
        await _bootstrap_config(session)
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.redis = redis
    configure_approval_broker(redis)  # distributed approval wakeup across API nodes
    configure_query_cache(redis)      # RAG query cache (keyed by config + corpus version)
    # Built-in product manual → SQL chunks + pgvector (public, version-gated). Fail-soft:
    # an unreachable embedding service simply retries on the next boot (the marker in
    # app_settings is only written after a full success).
    try:
        from core.infrastructure.manual_seed import seed_product_manual
        from core.infrastructure.vector import TEIEmbedder

        if await seed_product_manual(SessionLocal, TEIEmbedder(timeout=120.0)):
            await bump_corpus_version(redis)
    except Exception:
        logger.warning("product-manual seed failed — retrying next boot", exc_info=True)
    # Load the local extractor while the API starts serving; failures are logged only.
    _schedule_tool_intent_warmup()
    yield
    # Stop the background warm-up cleanly if it is still loading.
    global _warmup_task
    if _warmup_task is not None and not _warmup_task.done():
        _warmup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _warmup_task
    _warmup_task = None
    await aclose_llm_clients()
    await redis.aclose()


app = FastAPI(title="Delveta API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(VocabError)
async def _vocab_error_handler(request: Request, exc: VocabError):
    """Map vocabulary domain errors (403/404/etc.) to JSON responses."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(DriveError)
async def _drive_error_handler(request: Request, exc: DriveError):
    """Map cloud-drive domain errors (403/404/409/etc.) to JSON responses."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# Research ``execution_mode`` is a Literal that must read as a malformed request (400) rather
# than Pydantic's default 422; the handler delegates every non-execution_mode body error to the
# stock 422 response, so nothing else changes.
app.add_exception_handler(RequestValidationError, research_validation_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # open during dev; tighten for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Per-request request_id log context (pure ASGI, so it covers SSE bodies too).
app.add_middleware(RequestLogContext)

# Serve cached TTS audio / images (paths produced by TTS and image scraping).
for _dir in (settings.audio_cache_path, settings.image_cache_path):
    _dir.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=Path(settings.audio_cache_path).resolve()), name="audio")
app.mount("/images", StaticFiles(directory=Path(settings.image_cache_path).resolve()), name="images")

# User avatar uploads (self-service profile).
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/avatars", StaticFiles(directory=AVATAR_DIR.resolve()), name="avatars")

# Cloud drive: workspaces + file upload lifecycle / download / sharing / RAG status.
app.include_router(drive_workspaces_router)
app.include_router(drive_files_router)
app.include_router(drive_folders_router)
app.include_router(drive_trash_router)
app.include_router(drive_users_router)

# Functional groups (each router owns one domain; paths are unchanged from the monolith).
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(config_router)
app.include_router(vocab_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(jobs_router)
app.include_router(rag_admin_router)
app.include_router(research_router)
app.include_router(registry_admin_router)
app.include_router(observability_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

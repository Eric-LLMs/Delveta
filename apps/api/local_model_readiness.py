"""Startup load, real-inference warm-up, residency and Ready for local models.

HTTP 200 from a sidecar is not model readiness: an Ollama process can answer
``/health`` before its weights are loaded, and a plain TCP accept says nothing
about the model at all. This module owns the ONE readiness sequence used by the
API process:

    service reachable -> weights loaded -> real lightweight inference ->
    result validated -> residency confirmed -> Ready

The three models the chat path actually depends on are REQUIRED:

* ``embedding``   — Delveta-BGE-M3 (INT8 ONNX) at ``embedding_base_url``
* ``cap_router``  — LayaChoice behind ``/v1/systemone`` at ``cap_router_url``
* ``tool_intent`` — the Qwen extractor behind Ollama at the OpenAI-compatible
                    base URL (the native ``/api/generate`` + ``/api/ps`` wire is
                    used for the probe only; the business call is unchanged)

TTS and STT are optional: they are reported as ``reachable``/``failed`` but
never gate the API's readiness. A model whose endpoint is not configured is
reported as ``not_configured`` and is not gated either — it does not take part
in a request on this deployment.

The probes are read-only: one safe embedding, one synthetic choice question, one
one-token generation. No tool, file, database row or user data is touched.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import types
from dataclasses import dataclass

import httpx
from core.config import settings

logger = logging.getLogger(__name__)

REQUIRED = "required"
OPTIONAL = "optional"

NOT_CONFIGURED = "not_configured"
LOADING = "loading"
WARMING = "warming"
READY = "ready"
REACHABLE = "reachable"
FAILED = "failed"

# Per-model bounds. The API runs on the host and may start before the sidecars
# finish loading, so these are LOAD bounds, not per-request guardrails; they are
# never used for business calls.
_WARM_TIMEOUTS = {
    "embedding": 60.0,
    "cap_router": 300.0,
    "tool_intent": 300.0,
    "tts": 30.0,
    "stt": 60.0,
}
_QUICK_TIMEOUT = 15.0
# How often the already-Ready state is re-proven. A restarted sidecar must stop
# being reported Ready promptly, so the window is bounded well below the minutes
# a stale "ready" would mislead a deploy gate; each quick probe is a real (but
# tiny) inference or a residency lookup, not a bare TCP check.
_RECHECK_INTERVAL_SECONDS = 15.0
_EMBED_PROBE_TEXT = "Delveta local model readiness probe"
_CAP_PROBE_ID = "cap-warmup"


class ProbeError(RuntimeError):
    """A probe reached the service but did not prove the model is usable."""


@dataclass
class ModelState:
    """One model's readiness record (the shape both logs and /health/ready use)."""

    name: str
    endpoint: str
    model_id: str
    kind: str
    state: str = NOT_CONFIGURED
    load_ms: int | None = None
    warmup_ms: int | None = None
    detail: str = ""
    error: str = ""

    @property
    def gating(self) -> bool:
        return self.kind == REQUIRED and self.state != NOT_CONFIGURED

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "state": self.state,
            "endpoint": self.endpoint,
            "model": self.model_id,
            "load_ms": self.load_ms,
            "warmup_ms": self.warmup_ms,
            "detail": self.detail,
            "error": self.error,
        }


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _short(exc: BaseException, limit: int = 240) -> str:
    return " ".join(str(exc).split())[:limit]


def _model_endpoint(base: str) -> str:
    """The SERVICE ROOT for a configured base URL (drops an OpenAI ``/v1`` tail)."""
    url = (base or "").strip().rstrip("/")
    return url.removesuffix("/v1")


def _validate_embedding(vectors: object) -> None:
    if not isinstance(vectors, list) or len(vectors) != 1:
        raise ProbeError("embedding reply is not a one-item list")
    vector = vectors[0]
    if not isinstance(vector, list) or len(vector) != settings.embedding_dim:
        got = len(vector) if isinstance(vector, list) else type(vector).__name__
        raise ProbeError(
            f"embedding dimension is {got}, expected {settings.embedding_dim}")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
        raise ProbeError("embedding carries a non-finite or non-numeric value")
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if not 0.98 <= norm <= 1.02:
        raise ProbeError(f"embedding is not L2-normalized (norm={norm:.4f})")


def _cap_probe_question() -> dict:
    """The REAL production choice-question wire, with a synthetic safe card."""
    from core.application.chat.intent_funnel.cap_router.backends.service import QID
    from core.application.chat.intent_funnel.cap_router.card_renderer import (
        render_choice_question,
    )

    entry = types.SimpleNamespace(
        capability_id=_CAP_PROBE_ID,
        tool_binding="warmup",
        description="Internal model readiness probe. This is not a user action.",
        negatives=(),
        parameters={},
    )
    candidate = types.SimpleNamespace(capability_id=_CAP_PROBE_ID)
    return {"state": "warmup",
            "questions": {QID: render_choice_question([candidate],
                                                      {_CAP_PROBE_ID: entry})}}


def _validate_choice(body: object) -> None:
    from core.application.chat.intent_funnel.cap_router.backends.service import QID
    from core.application.chat.intent_funnel.cap_router.card_renderer import (
        REJECT_LABEL,
    )

    if not isinstance(body, dict):
        raise ProbeError("cap_router reply is not a JSON object")
    answers = body.get("answers")
    answer = answers.get(QID) if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ProbeError("cap_router did not answer the choice question")
    if answer.get("choice") not in {_CAP_PROBE_ID, REJECT_LABEL}:
        raise ProbeError(f"cap_router answered an off-card choice "
                         f"{answer.get('choice')!r}")


def _validate_generate(body: object) -> None:
    if not isinstance(body, dict):
        raise ProbeError("generate reply is not a JSON object")
    if not isinstance(body.get("response"), str):
        raise ProbeError("generate reply carries no response text")
    if body.get("done") not in (True, None):
        raise ProbeError("generate did not finish")


def _model_entry(ps: object, model: str) -> dict | None:
    """The ``/api/ps`` row for ``model`` (exact tag, else base-name match)."""
    rows = ps.get("models") if isinstance(ps, dict) else None
    if not isinstance(rows, list):
        return None
    base = model.split(":")[0]
    for row in rows:
        if not isinstance(row, dict):
            continue
        names = {str(row.get("name") or ""), str(row.get("model") or "")}
        if model in names:
            return row
        if any(name.split(":")[0] == base and name for name in names):
            return row
    return None


def _client(timeout: float) -> httpx.AsyncClient:
    """One probe client. Tests replace this seam, never the module's transport."""
    return httpx.AsyncClient(timeout=timeout)


class LocalModelReadiness:
    """The API process's single readiness manager for local model services."""

    def __init__(self, *, interval: float = _RECHECK_INTERVAL_SECONDS):
        self._interval = interval
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._states = self._build_states()

    @staticmethod
    def _build_states() -> dict[str, ModelState]:
        states = {
            "embedding": ModelState(
                "embedding", (settings.embedding_base_url or "").rstrip("/"),
                settings.embedding_model, REQUIRED),
            "cap_router": ModelState(
                "cap_router", (settings.chat_cap_router_url or "").rstrip("/"),
                "cap_router", REQUIRED),
            "tool_intent": ModelState(
                "tool_intent",
                _model_endpoint(settings.chat_tool_intent_local_url),
                settings.chat_tool_intent_local_model, REQUIRED),
            "tts": ModelState(
                "tts", _model_endpoint(settings.tts_base_url),
                settings.tts_model, OPTIONAL),
            "stt": ModelState(
                "stt", _model_endpoint(settings.stt_base_url),
                settings.stt_model, OPTIONAL),
        }
        for state in states.values():
            if state.endpoint:
                state.state = LOADING
        return states

    # ── public surface ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start ONE background readiness task (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="local-model-readiness")

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning("local-model readiness loop stopped with an error",
                               exc_info=True)

    @property
    def ready(self) -> bool:
        return all(st.state == READY for st in self._states.values() if st.gating)

    def snapshot(self) -> dict:
        models = {name: st.as_dict() for name, st in self._states.items()}
        return {
            "status": "ready" if self.ready else "not_ready",
            "models": models,
        }

    async def warm_all(self) -> None:
        """Run every model's full load+warm sequence once (startup and retry)."""
        await asyncio.gather(*(self._warm(name) for name in self._states))

    # ── loop ─────────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        await self.warm_all()
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return
            except TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            await self._recheck_all()

    async def _recheck_all(self) -> None:
        for name, state in self._states.items():
            if state.state == NOT_CONFIGURED:
                continue
            # A model that was not Ready (first pass, a failed load, or a restart
            # detected here) gets the FULL sequence again: health alone must not
            # promote it, because a restarted sidecar is loaded but not warmed.
            # ``reachable`` is the terminal state for an optional model.
            if state.state not in (READY, REACHABLE):
                await self._warm(name)
                continue
            try:
                await self._quick_probe(name)
            except Exception as exc:  # noqa: BLE001 - a lost model is Not Ready
                logger.warning(
                    "local-model not-ready name=%s reason=%s detail=%s",
                    name, type(exc).__name__, _short(exc))
                await self._warm(name)
                continue
            if state.kind == OPTIONAL:
                state.state = REACHABLE
                continue
            state.state = READY

    # ── one model ────────────────────────────────────────────────────────────

    async def _warm(self, name: str) -> None:
        state = self._states[name]
        if state.state == NOT_CONFIGURED:
            logger.info("local-model not-configured name=%s", name)
            return
        state.state = LOADING
        state.error = ""
        logger.info("local-model service-started name=%s endpoint=%s model=%s",
                    name, state.endpoint, state.model_id)
        try:
            load_ms, warmup_ms, detail = await self._probe(name, state)
        except Exception as exc:  # noqa: BLE001 - a failed model is Not Ready
            state.state = FAILED
            state.error = _short(exc)
            logger.warning("local-model not-ready name=%s reason=%s detail=%s",
                           name, type(exc).__name__, state.error)
            return

        state.load_ms = load_ms
        state.warmup_ms = warmup_ms
        state.detail = detail
        if state.kind == OPTIONAL:
            state.state = REACHABLE
            logger.info("local-model reachable name=%s endpoint=%s",
                        name, state.endpoint)
            return
        state.state = WARMING
        logger.info("local-model model-loaded name=%s model=%s load_ms=%s",
                    name, state.model_id, load_ms)
        logger.info("local-model warmup-completed name=%s warmup_ms=%s valid=true",
                    name, warmup_ms)
        state.state = READY
        logger.info("local-model ready name=%s detail=%s", name, detail)

    async def _probe(self, name: str, state: ModelState) -> tuple[int, int, str]:
        timeout = _WARM_TIMEOUTS.get(name, 60.0)
        if name == "embedding":
            return await self._probe_embedding(timeout)
        if name == "cap_router":
            return await self._probe_cap_router(timeout)
        if name == "tool_intent":
            return await self._probe_tool_intent(timeout)
        return await self._probe_health_only(timeout, state.endpoint)

    async def _quick_probe(self, name: str) -> None:
        state = self._states[name]
        timeout = _QUICK_TIMEOUT
        if name == "embedding":
            async with _client(timeout) as client:
                await self._embed(client, state.endpoint)
            return
        if name == "cap_router":
            async with _client(timeout) as client:
                body = await self._cap_health(client, state.endpoint)
                if body.get("status") != "ok":
                    raise ProbeError(f"cap_router is {body.get('status')!r}")
            return
        if name == "tool_intent":
            async with _client(timeout) as client:
                ps = await self._ollama_ps(client, state.endpoint)
                if _model_entry(ps, state.model_id) is None:
                    raise ProbeError("model is not resident")
            return
        async with _client(timeout) as client:
            resp = await client.get(state.endpoint + "/health")
            resp.raise_for_status()

    # ── probes ───────────────────────────────────────────────────────────────

    async def _probe_embedding(self, timeout: float) -> tuple[int, int, str]:
        started = time.monotonic()
        async with _client(timeout) as client:
            resp = await client.get(self._states["embedding"].endpoint + "/health")
            resp.raise_for_status()
            load_ms = _ms(started)
            warm = time.monotonic()
            await self._embed(client, self._states["embedding"].endpoint)
            return load_ms, _ms(warm), f"dim={settings.embedding_dim}"

    @staticmethod
    async def _embed(client: httpx.AsyncClient, endpoint: str) -> None:
        resp = await client.post(endpoint + "/embed",
                                 json={"inputs": [_EMBED_PROBE_TEXT],
                                       "normalize": True})
        resp.raise_for_status()
        _validate_embedding(resp.json())

    @staticmethod
    async def _cap_health(client: httpx.AsyncClient, endpoint: str) -> dict:
        resp = await client.get(endpoint + "/health")
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise ProbeError("cap_router health reply is not a JSON object")
        return body

    async def _probe_cap_router(self, timeout: float) -> tuple[int, int, str]:
        endpoint = self._states["cap_router"].endpoint
        started = time.monotonic()
        deadline = started + timeout
        async with _client(timeout) as client:
            body: dict = {}
            while True:
                try:
                    body = await self._cap_health(client, endpoint)
                    if body.get("status") == "ok":
                        break
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise ProbeError(
                        f"cap_router did not finish loading within {timeout:.0f}s")
                await asyncio.sleep(0.5)
            load_ms = _ms(started)
            warm = time.monotonic()
            resp = await client.post(endpoint + "/v1/systemone",
                                     json=_cap_probe_question())
            resp.raise_for_status()
            _validate_choice(resp.json())
            model = str(body.get("model") or "cap_router")
            return load_ms, _ms(warm), f"model={model}"

    async def _probe_tool_intent(self, timeout: float) -> tuple[int, int, str]:
        state = self._states["tool_intent"]
        model = state.model_id
        started = time.monotonic()
        async with _client(timeout) as client:
            ps = await self._ollama_ps(client, state.endpoint)
            load_ms = 0 if _model_entry(ps, model) is not None else None
            generate = asyncio.create_task(
                self._ollama_generate(client, state.endpoint, model))
            if load_ms is None:
                while not generate.done():
                    await asyncio.sleep(0.25)
                    ps = await self._ollama_ps(client, state.endpoint)
                    if _model_entry(ps, model) is not None:
                        load_ms = _ms(started)
                        break
            _validate_generate(await generate)
            warmup_ms = _ms(started) - (load_ms or 0)
            ps = await self._ollama_ps(client, state.endpoint)
            row = _model_entry(ps, model)
            if row is None:
                raise ProbeError("model is not resident after the warm-up")
            logger.info(
                "local-model residency-confirmed name=tool_intent model=%s "
                "size=%s size_vram=%s expires_at=%s",
                model, row.get("size"), row.get("size_vram"),
                row.get("expires_at"))
            return _ms(started) if load_ms is None else load_ms, warmup_ms, \
                f"expires_at={row.get('expires_at')}"

    @staticmethod
    async def _ollama_ps(client: httpx.AsyncClient, root: str) -> dict:
        resp = await client.get(root + "/api/ps")
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise ProbeError("ollama ps reply is not a JSON object")
        return body

    @staticmethod
    async def _ollama_generate(client: httpx.AsyncClient, root: str,
                               model: str) -> dict:
        resp = await client.post(root + "/api/generate", json={
            "model": model,
            "prompt": "warmup",
            "stream": False,
            "options": {"num_predict": 1},
            # Residency request: the native wire is the ONLY one that accepts it
            # (the OpenAI-compatible /v1 dialect ignores it silently).
            "keep_alive": -1,
        })
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise ProbeError("generate reply is not a JSON object")
        return body

    async def _probe_health_only(self, timeout: float,
                                 endpoint: str) -> tuple[int, int, str]:
        started = time.monotonic()
        async with _client(timeout) as client:
            resp = await client.get(endpoint + "/health")
            resp.raise_for_status()
        return _ms(started), 0, "health-only (optional model)"

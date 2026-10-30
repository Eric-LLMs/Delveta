"""LLM client for a chat-completions HTTP API, with real streaming.

The endpoint, key, and model are read from configuration — unless the client was built
as a *shell* (``require_channel=True``, the worker posture): then it holds no key at all
and every call must carry (or find in the request context) a channel resolved by the
LLM dispatch gateway (:mod:`core.infrastructure.llm_routing`), else it raises
:class:`NoActiveChannelError`. Global-key fallback is forbidden by platform doctrine.
"""
import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from agent.llm.llm_errors import raise_classified
from openai import AsyncOpenAI

from core.config import settings
from core.infrastructure.request_context import get_request_llm_channel

logger = logging.getLogger(__name__)

# The OpenAI SDK refuses to build a client without credentials, but the real key is loaded
# from the DB (admin panel) at startup. Use a placeholder until ``configure`` supplies it.
_PLACEHOLDER_KEY = "sk-placeholder"


# ── Connection pooling ────────────────────────────────────────────────────────────────
# Every turn resolves its channel from the DB, so a naive per-call ``AsyncOpenAI(...)``
# opens a fresh TCP+TLS connection on every call (~1s to this provider). Two caches fix
# that without ever letting a credential outlive the request that resolved it:
#
# * TRANSPORTS — one ``httpx.AsyncClient`` per ORIGIN (scheme/host/port), shared by every
#   credential of that provider. A transport is never evicted while the process lives; it
#   is closed once, at app shutdown, by :func:`aclose_llm_clients`.
# * WRAPPERS — thin ``AsyncOpenAI`` objects keyed by ``(base_url, sha256(api_key))``,
#   sharing the origin's transport. LRU + TTL bounded; eviction is a pure dict removal
#   and NEVER calls ``close()`` (``AsyncOpenAI.close()`` closes the shared transport).
#
# Eviction is not revocation: correctness comes from resolving the channel per request.
# Dropping a cached wrapper only frees memory; the next call re-reads the DB channel and
# builds a new wrapper. In-flight requests hold their own references, so eviction while
# a request is running cannot affect it.
_WRAPPER_CACHE_MAX = 256
_WRAPPER_CACHE_TTL_SECONDS = 30 * 60


@dataclass
class _TransportEntry:
    client: httpx.AsyncClient
    # An httpx.AsyncClient is bound to the event loop that opened its sockets. Production
    # runs one loop per process; pytest gives every test a fresh loop, so a mismatch
    # (only ever seen in tests) drops the stale transport instead of reusing it.
    loop: asyncio.AbstractEventLoop | None


@dataclass
class _WrapperEntry:
    client: AsyncOpenAI
    created_at: float


_TRANSPORTS: dict[tuple[str, str, int], _TransportEntry] = {}
_WRAPPERS: OrderedDict[tuple[str, str], _WrapperEntry] = OrderedDict()
_POOL_LOCK = threading.Lock()


def _origin_of(base_url: str) -> tuple[str, str, int]:
    """Return ``(scheme, host, port)`` — the identity that owns one TCP/TLS pool."""
    url = httpx.URL(base_url)
    port = url.port or (443 if url.scheme == "https" else 80)
    return url.scheme, url.host, port


def _key_fingerprint(api_key: str) -> str:
    """Cache-key form of a credential: never keep or log the plaintext key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _get_transport(base_url: str) -> httpx.AsyncClient:
    origin = _origin_of(base_url)
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    with _POOL_LOCK:
        entry = _TRANSPORTS.get(origin)
        if entry is not None and entry.loop is loop:
            return entry.client
        # Keep the SDK's own transport defaults (5/600/600/600, 1000/100) and change ONE
        # thing: keepalive_expiry 5s → 300s. At 5s an idle socket is dropped between two
        # turns of the same user, which is exactly the latency this cache exists to save.
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0),
            limits=httpx.Limits(
                max_connections=settings.llm_pool_max_connections,
                max_keepalive_connections=settings.llm_pool_max_keepalive_connections,
                keepalive_expiry=settings.llm_pool_keepalive_expiry_seconds,
            ),
            follow_redirects=True,
            trust_env=True,
        )
        _TRANSPORTS[origin] = _TransportEntry(client=client, loop=loop)
        return client


def _get_wrapper(base_url: str, api_key: str) -> AsyncOpenAI:
    """Return the cached ``AsyncOpenAI`` for one (endpoint, credential) pair."""
    key = (base_url, _key_fingerprint(api_key))
    transport = _get_transport(base_url)
    now = time.monotonic()
    with _POOL_LOCK:
        entry = _WRAPPERS.get(key)
        if entry is not None:
            if now - entry.created_at < _WRAPPER_CACHE_TTL_SECONDS:
                _WRAPPERS.move_to_end(key)
                return entry.client
            del _WRAPPERS[key]  # expired: drop the reference, never close()
        while len(_WRAPPERS) >= _WRAPPER_CACHE_MAX:
            _WRAPPERS.popitem(last=False)  # LRU: drop the reference, never close()
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=settings.llm_timeout_seconds,
            http_client=transport,
        )
        _WRAPPERS[key] = _WrapperEntry(client=client, created_at=now)
        return client


def invalidate_channel(base_url: str | None = None, api_key: str | None = None) -> int:
    """Drop cached credential wrappers, optionally filtered by endpoint and key.

    This is memory hygiene after a credential is rotated or removed from the DB. It is
    NOT the revocation mechanism and does not need to be: the channel is resolved from
    the DB on every call, so a revoked key is never selected again regardless of what
    this cache holds. No transport is closed and no in-flight request is affected.
    """
    fingerprint = _key_fingerprint(api_key) if api_key else None
    with _POOL_LOCK:
        if base_url is None and fingerprint is None:
            dropped = len(_WRAPPERS)
            _WRAPPERS.clear()
            return dropped
        doomed = [
            key for key in _WRAPPERS
            if (base_url is None or key[0] == base_url)
            and (fingerprint is None or key[1] == fingerprint)
        ]
        for key in doomed:
            del _WRAPPERS[key]
        return len(doomed)


async def aclose_llm_clients() -> None:
    """Close every pooled transport. Idempotent; called once from the API lifespan."""
    with _POOL_LOCK:
        entries = list(_TRANSPORTS.values())
        _TRANSPORTS.clear()
        _WRAPPERS.clear()
    for entry in entries:
        try:
            await entry.client.aclose()
        except Exception:
            logger.warning("failed to close a pooled LLM transport", exc_info=True)


class NoActiveChannelError(RuntimeError):
    """A shell client (``require_channel=True``) was called with no gateway-resolved channel.

    Fail-Fast by design: the worker holds no resident commercial key, so this means the
    dispatch gateway found no active channel for the job's owner at job start. The caller
    must surface it as the job's error — never retry with a global key.
    """


def _wire_tool_call(tc: dict) -> dict:
    """Normalize one tool_call to the OpenAI wire format ``{id, type, function:{name, arguments}}``.

    Strict providers (e.g. DeepSeek) deserialize each tool call with a required ``type``
    discriminator and a ``function`` wrapper; the agent loop stores a compact
    ``{id, name, arguments}`` shape, which they reject with ``missing field 'type'``.
    """
    if "function" in tc:
        return {**tc, "type": tc.get("type") or "function"}
    return {
        "id": tc.get("id"),
        "type": "function",
        "function": {
            "name": tc.get("name"),
            "arguments": tc.get("arguments") or "{}",
        },
    }


def _wire_messages(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with assistant ``tool_calls`` in OpenAI wire format.

    Only assistant messages carrying ``tool_calls`` are rewritten (into a fresh dict, so the
    caller's list is untouched); everything else passes through as-is. This covers both the
    in-turn replay (step N+1 echoes step N's calls) and history replayed from ``body.history``.
    """
    out = []
    for m in messages:
        m = dict(m)
        tcs = m.get("tool_calls")
        if m.get("role") == "assistant" and tcs:
            m["tool_calls"] = [_wire_tool_call(tc) for tc in tcs]
        out.append(m)
    return out


class OpenAILLM:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        require_channel: bool = False,
    ) -> None:
        # ``require_channel=True`` = shell posture (worker / retrieval): NO usable key is
        # stored on the client; every call must carry a gateway-resolved channel.
        self.require_channel = require_channel
        self.client = AsyncOpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key or _PLACEHOLDER_KEY,
            timeout=settings.llm_timeout_seconds,
        )
        self.model = model or settings.llm_model

    def configure(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None) -> None:
        """Rebuild the client in place so a runtime config change takes effect without a restart."""
        self.client = AsyncOpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key or _PLACEHOLDER_KEY,
            timeout=settings.llm_timeout_seconds,
        )
        if model:
            self.model = model

    def _call_channel(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> tuple[AsyncOpenAI, str]:
        """Resolve the client + model for ONE call: explicit args > request contextvar > self.

        Priority per field, so a caller that only overrides the model still rides the turn's
        pinned channel. A shell client (``require_channel``) raises
        :class:`NoActiveChannelError` unless BOTH base_url and api_key come from the
        explicit args or the gateway-pinned contextvar — the embedded/global config is
        never consulted, which is what "worker holds no resident key" means in code.

        A resolved channel rides the shared per-origin transport (see ``_get_wrapper``);
        per-call ``timeout``/``max_retries`` are applied to that cached wrapper through
        ``with_options``, which reuses its ``http_client`` instead of opening a new pool.
        Calls with NO resolved channel keep the historical behaviour exactly: the
        embedded-config client for plain calls, a throwaway client when the caller pins a
        timeout/retry budget.
        """
        ch = get_request_llm_channel()
        if ch is not None:
            c_model, c_base, c_key = ch
            model = model or c_model
            base_url = base_url or c_base
            api_key = api_key or c_key
        if self.require_channel and not (base_url and api_key):
            raise NoActiveChannelError(
                "no gateway-resolved LLM channel for this call — the shell client holds no "
                "key and global-key fallback is forbidden (Fail-Fast at call site)"
            )
        if base_url or api_key:
            client = _get_wrapper(
                base_url=base_url or str(self.client.base_url),
                api_key=api_key or self.client.api_key or _PLACEHOLDER_KEY,
            )
            options: dict = {}
            if timeout is not None:
                options["timeout"] = timeout
            if max_retries is not None:
                options["max_retries"] = max_retries
            if options:
                client = client.with_options(**options)
            return client, model or self.model
        if timeout is not None or max_retries is not None:
            kwargs: dict = {
                "base_url": str(self.client.base_url),
                "api_key": self.client.api_key or _PLACEHOLDER_KEY,
                "timeout": timeout if timeout is not None else settings.llm_timeout_seconds,
            }
            if max_retries is not None:
                kwargs["max_retries"] = max_retries
            return AsyncOpenAI(**kwargs), model or self.model
        return self.client, model or self.model

    @staticmethod
    def _messages(prompt: str, system_prompt: str,
                  images: list[str] | None = None) -> list[dict]:
        if images:
            # Multimodal wire (same content-part shape as pdf.py table OCR and
            # vision_tool): user content = text first, then image_url data URLs.
            content: list[dict] = [{"type": "text", "text": prompt}]
            content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    async def _stream_accumulate(
        self, client: AsyncOpenAI, mdl: str, messages: list[dict],
        response_format: dict | None = None,
        usage_out: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """One streamed completion, accumulated to the full text.

        Batch generation never needs a whole-response wall time: providers such as
        dashscope enforce a ~300s cutoff on NON-streaming requests, so a full-context
        Pass A that legitimately needs 4-5 minutes dies at the gateway while the model
        is still generating. Streaming moves the per-call timeout to an idle-between-
        chunks deadline, and ``enable_thinking: false`` (Qwen-compatible flag,
        ``llm_disable_thinking``) removes reasoning tokens — pure latency for
        schema-validated JSON output. Interactive chat paths are untouched.

        ``max_tokens`` bounds the output length for callers whose reply shape is
        known small (tool-intent verdicts); ``disable_thinking`` makes the thinking-off
        request EXPLICIT at that call site instead of riding the global knob —
        the experiments pin both per-call.

        When ``usage_out`` is given, the provider's real token counts are merged into
        it (``stream_options.include_usage``): the usage chunk arrives last, with no
        choices — reading it off the stream keeps instrumentation honest (no estimates).
        """
        kwargs: dict = {
            "model": mdl, "messages": messages, "stream": True,
            # None keeps the historical 0.3; per-call overrides may ask for 0.0.
            "temperature": 0.3 if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        if usage_out is not None:
            kwargs["stream_options"] = {"include_usage": True}
        if settings.llm_disable_thinking or disable_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}
        stream = await client.chat.completions.create(**kwargs)
        parts: list[str] = []
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            # NOTE the ``usage_out is not None`` guard: some providers (deepseek)
            # send a usage chunk even when include_usage was NOT requested —
            # writing through a None sink used to raise TypeError, which made
            # every usage_out-less complete_json caller would fail-open
            # on exactly those channels.
            if usage is not None and usage_out is not None:
                usage_out["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                usage_out["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        return "".join(parts).strip()

    async def complete(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> str:
        # A per-call ``timeout`` (e.g. toolkit full-context generation) forces a fresh
        # client; without it the shared client's global wall time applies. Under the
        # streaming wire this bounds IDLE time between chunks, not total generation.
        client, mdl = self._call_channel(model, base_url, api_key, timeout=timeout)
        return await self._stream_accumulate(
            client, mdl, self._messages(prompt, system_prompt),
            temperature=temperature)

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        usage_out: dict | None = None,
        images: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> dict:
        """Structured completion: ask the provider for a JSON object.

        Uses JSON mode (``response_format={"type": "json_object"}``) — the prompt must
        contain the word "json" for some providers to honour the mode. Returns the parsed
        JSON object. A provider that rejects JSON mode raises (the caller can fall back to
        ``complete`` + a tolerant JSON parse). A per-call ``timeout`` bounds long
        full-context generations (toolkit); under the streaming wire it is an idle-between-
        chunks deadline, not a total-generation cutoff. ``images`` (data-URL list) makes
        the user turn multimodal — used by the deck visual-understanding pass.
        ``max_tokens``/``disable_thinking`` are pinned per-call by structured-output
        callers; every other caller keeps the old behavior.
        """
        client, mdl = self._call_channel(model, base_url, api_key, timeout=timeout)
        try:
            content = await self._stream_accumulate(
                client, mdl, self._messages(prompt, system_prompt, images=images),
                response_format={"type": "json_object"},
                usage_out=usage_out,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
        except Exception as exc:
            raise raise_classified(exc) from exc
        try:
            return json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"model did not return valid JSON: {exc}") from exc

    async def complete_json_once(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 256,
        disable_thinking: bool = False,
        usage_out: dict | None = None,
    ) -> dict:
        """One non-streaming structured completion (``stream=False``).

        The chat path intentionally never reads a whole reply off a stream: some
        providers cut long NON-streaming requests at the gateway, which is why
        :meth:`complete_json` streams. A small, bounded extraction call is the
        opposite case — one request, one response, no partial-JSON assembly.

        Thinking is disabled with both provider dialects this repository uses:
        the Qwen-compatible ``enable_thinking`` body switch and
        ``reasoning_effort="none"`` for reasoning-first providers such as
        DeepSeek (which ignores ``enable_thinking``). Both are gated by
        ``settings.llm_disable_thinking`` unless the caller pins them with
        ``disable_thinking=True``.

        Raises the classified provider error on transport failure and
        :class:`TypeError` when the reply is not a JSON object — neither is
        repaired here.
        """
        client, mdl = self._call_channel(model, base_url, api_key, timeout=timeout)
        kwargs: dict = {
            "model": mdl,
            "messages": self._messages(prompt, system_prompt),
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if settings.llm_disable_thinking or disable_thinking:
            # Two dialects for one intent: Qwen reads ``enable_thinking``, while
            # reasoning-first providers (DeepSeek) ignore it and expose
            # ``reasoning_effort``. Send both so a structured call never spends
            # its bounded ``max_tokens`` budget on reasoning, whichever channel
            # the request resolved to.
            kwargs["extra_body"] = {"enable_thinking": False}
            kwargs["reasoning_effort"] = "none"
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise raise_classified(exc) from exc
        if usage_out is not None:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                usage_out["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                usage_out["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"model did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TypeError("model did not return a JSON object")
        return parsed

    async def complete_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: dict | None = None,
    ) -> AsyncIterator[str]:
        # Per-call timeout/max_retries overrides force a fresh client (the shared
        # self.client carries process-wide settings); otherwise reuse it.
        client, mdl = self._call_channel(model, base_url, api_key, timeout=timeout, max_retries=max_retries)
        # Same thinking switch as ``_stream_accumulate``: a reasoning pass would be
        # generated (and paid for) before the first visible token, so a streamed
        # chat answer would sit silent instead of streaming.
        body = dict(extra_body or {})
        if settings.llm_disable_thinking:
            body.setdefault("enable_thinking", False)
        stream = await client.chat.completions.create(
            model=mdl,
            messages=self._messages(prompt, system_prompt),
            temperature=0.3,
            stream=True,
            **({"extra_body": body} if body else {}),
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream a chat completion, yielding per-chunk event dicts.

        Events (``{"type": ...}``):
        - ``thinking``: a ``reasoning_content`` delta (provider reasoning, when present);
        - ``content``: a content delta;
        - ``tool_calls``: the fully accumulated tool calls ``[{id, name, arguments}]``
          (empty list when the model made no calls), emitted once at the end of the stream;
        - ``usage``: the provider token counts ``{prompt_tokens, completion_tokens,
          total_tokens}`` (all 0 when the provider omits them).

        ``base_url`` / ``api_key`` optionally route this call through a specific LLM
        channel without mutating the shared client. ``disable_thinking`` sends the
        Qwen-compatible ``enable_thinking: false`` flag — the live voice-call path uses
        it because reasoning tokens are pure time-to-first-sentence there; interactive
        typed chat keeps thinking on.
        """
        client, mdl = self._call_channel(model, base_url, api_key)
        kwargs = {
            "model": mdl,
            "messages": _wire_messages(messages),
            "temperature": 0.3,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        if disable_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}
        stream = await client.chat.completions.create(**kwargs)

        # Tool-call arguments/name arrive as fragmented deltas keyed by index; accumulate
        # them across chunks, then emit the assembled calls once the stream ends.
        acc: dict[int, dict] = {}
        async for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    u = chunk.usage
                    yield {
                        "type": "usage",
                        "data": {
                            "prompt_tokens": u.prompt_tokens or 0,
                            "completion_tokens": u.completion_tokens or 0,
                            "total_tokens": u.total_tokens or 0,
                        },
                    }
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "thinking", "data": reasoning}
            if delta.content:
                yield {"type": "content", "data": delta.content}
            for tc in (delta.tool_calls or []):
                entry = acc.setdefault(tc.index, {"id": tc.id or "", "name": "", "arguments": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        entry["name"] += tc.function.name
                    if tc.function.arguments:
                        entry["arguments"] += tc.function.arguments

        if acc:
            ordered = [acc[i] for i in sorted(acc)]
            yield {"type": "tool_calls", "data": ordered}
        else:
            yield {"type": "tool_calls", "data": []}

    async def generate_definition(self, term: str) -> str:
        """Generate an English definition + Chinese translation for a term."""
        prompt = (
            f"Provide a clear, concise English definition and its Chinese translation "
            f"for the term '{term}'."
        )
        return await self.complete(prompt, "You are a helpful dictionary assistant. Output only the definition.")

    async def analyze_syntax(self, sentence: str) -> str:
        """Syntactic/semantic analysis, returns Markdown."""
        prompt = (
            "Please perform a professional syntactic and semantic analysis for the following "
            "sentence, specifically tailored for an industry/technical context.\n"
            f'Sentence: "{sentence}"\n\n'
            "You MUST format your response EXACTLY following this Markdown template "
            "(Do not output markdown codeblock backticks ```):\n\n"
            "### 📖 句子意译\n"
            "(Provide a clear, fluent, and professional Chinese translation here)\n\n"
            "### 🔍 句法结构\n"
            "* **主干结构**: (Extract the core Subject-Verb-Object)\n"
            "* **深度解析**: (Explain clauses, modifiers, long dependencies, or specific grammatical structures clearly)\n\n"
            "### 🔑 行业核心词汇与词组\n"
            "* **[Key Term 1]**: (Explain its specific meaning and role in this technical context)\n"
            "* **[Key Term 2]**: (Explain its specific meaning and role in this technical context)"
        )
        return await self.complete(prompt, "You are an expert English linguist and tech-domain specialist.")

    async def explain_term(self, term: str, context: str) -> dict:
        """Explain a term's meaning in context (full-sentence translation + English definition of the term)."""
        system_prompt = (
            "You are a linguistic expert helper. "
            "Please perform two tasks:\n"
            "1. **Translate the entire context sentence** into natural, fluent Chinese.\n"
            "2. Provide a concise explanation of the **target term's** specific "
            "meaning/usage within this context (in English).\n\n"
            "Output strictly in JSON format with keys:\n"
            "- 'translation': The full Chinese translation of the sentence.\n"
            "- 'explanation': The explanation of the term."
        )
        user_prompt = f"Target Term: {term}\nContext Sentence: {context}"
        client, mdl = self._call_channel()
        try:
            resp = await client.chat.completions.create(
                model=mdl,
                messages=self._messages(user_prompt, system_prompt),
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            raise raise_classified(exc) from exc

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> dict:
        """Conversation with tool calls (used by the Agent loop).

        ``base_url`` / ``api_key`` optionally route this single call through a specific LLM
        channel (e.g. the credential pinned on a user's access token); the shared client is
        never mutated, so concurrent requests can each use their own channel.

        Returns ``{content, tool_calls, usage}`` where ``usage`` is the token counts from the
        provider (``prompt_tokens``/``completion_tokens``/``total_tokens``, all 0 if absent).
        """
        client, mdl = self._call_channel(model, base_url, api_key)
        kwargs = {"model": mdl, "messages": _wire_messages(messages), "temperature": 0.3}
        if tools:
            kwargs["tools"] = tools
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = resp.usage
        return {
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }

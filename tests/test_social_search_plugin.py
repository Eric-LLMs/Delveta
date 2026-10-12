"""Mock-network unit tests for ``plugins/social_search/plugin.py``.

The ``search_social`` tool's platform adapters all follow the same shape
(``httpx.AsyncClient`` GET → normalize), so the whole network layer is stubbed:
no real sockets, no real Reddit/X. These tests pin down the plugin *contract* —
the unified item schema, platform error mapping, and the ``auto`` merge policy —
without depending on external services.
"""
import os
import unittest.mock as mock

import httpx
import pytest

from agent.tools.tool_permissions import ToolPermission
from core.config import export_secret_env, settings
from plugins.social_search import plugin as mod


# ── network doubles ──────────────────────────────────────────────────────────
_REDDIT_PAYLOAD = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "Great thread",
                    "selftext": "x" * 2000,  # longer than the snippet cap
                    "author": "alice",
                    "score": 42,
                    "num_comments": 7,
                    "url": "/r/test/comments/abc/great_thread/",
                    "created_utc": 1234567890,
                }
            }
        ]
    }
}

_X_PAYLOAD = {
    "data": [
        {
            "id": "1",
            "text": "hello",
            "author_id": "bob",
            "public_metrics": {"like_count": 3, "retweet_count": 1, "reply_count": 0},
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]
}


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://www.reddit.com/search.json")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status_code, request=req))


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status_code = status_code

    @property
    def status_code(self):
        return self._status_code

    def raise_for_status(self):
        if self._status_code >= 400:
            raise _http_error(self._status_code)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Async-context-manager ``httpx.AsyncClient`` double that records GETs/POSTs."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self._status_code = status_code
        self.calls: list[tuple[str, dict, dict]] = []  # (url, params, kwargs)
        self.post_calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, **kwargs):
        self.calls.append((url, params, kwargs))
        return _FakeResponse(self._payload, self._status_code)

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return _FakeResponse(self._payload, self._status_code)


async def _run(payload, *, platform="reddit", query="q", status_code=200, **extra):
    """Run the raw ``_execute`` body with the network layer stubbed out."""
    fake = _FakeAsyncClient(payload, status_code)
    with mock.patch.object(mod.httpx, "AsyncClient", lambda **kw: fake):
        args = {"query": query, "platform": platform, **extra}
        result = await mod._execute(args, exec=None)
    return result, fake


# ── platform-selection doubles ────────────────────────────────────────────────
# These swap every adapter for a recording stub, so a test can assert exactly which
# platforms ``_execute`` *called* (not just which ``platform`` value was passed). The
# real adapters would otherwise hit the network (x/zhihu degrade via ``_domain_search``).
_OMITTED = object()


def _spy_adapters(record, *, raise_map=None):
    raise_map = raise_map or {}

    def _make(name):
        async def _adapter(query, limit, subreddit):
            record.append(name)
            if name in raise_map:
                raise raise_map[name]
            return [{"platform": name}]

        return _adapter

    return {name: _make(name) for name in mod._ADAPTERS}


async def _run_spy(*, platform=_OMITTED, subreddit=_OMITTED, raise_map=None):
    """Run ``_execute`` with all adapters stubbed; return (called_platforms, result)."""
    record: list[str] = []
    spies = _spy_adapters(record, raise_map=raise_map)
    args = {"query": "q"}
    if platform is not _OMITTED:
        args["platform"] = platform
    if subreddit is not _OMITTED:
        args["subreddit"] = subreddit
    with mock.patch.object(mod, "_ADAPTERS", spies):
        result = await mod._execute(args, exec=None)
    return record, result


@pytest.fixture(autouse=True)
def _no_reddit_creds():
    """Keep the anonymous-reddit tests deterministic: no OAuth creds, cold token cache."""
    with mock.patch.dict(
        os.environ,
        {
            "REDDIT_CLIENT_ID": "",
            "REDDIT_CLIENT_SECRET": "",
            "REDDIT_USERNAME": "",
            "REDDIT_PASSWORD": "",
        },
        clear=False,
    ):
        mod._TOKEN_CACHE.clear()
        yield
        mod._TOKEN_CACHE.clear()


# ── plugin contract ──────────────────────────────────────────────────────────
class TestPluginContract:
    def test_tool_name_and_permissions(self):
        tool = mod.PLUGIN.tools[0]
        assert tool.name == "search_social"
        assert tool.permission == {ToolPermission.READ, ToolPermission.NETWORK}
        assert tool.is_concurrency_safe is True

    def test_platform_enum(self):
        props = mod.PLUGIN.tools[0].parameters["properties"]
        assert props["platform"]["enum"] == ["reddit", "x", "zhihu", "auto"]


# ── reddit adapter ───────────────────────────────────────────────────────────
class TestReddit:
    async def test_normalizes_to_unified_schema(self):
        result, fake = await _run(_REDDIT_PAYLOAD)
        assert len(fake.calls) == 1
        assert result == [
            {
                "platform": "reddit",
                "title": "Great thread",
                "content": "x" * mod._SNIPPET_CHARS,  # truncated to the snippet cap
                "author": "alice",
                "metrics": {"score": 42, "comments": 7},
                "url": "https://www.reddit.com/r/test/comments/abc/great_thread/",
                "published_utc": 1234567890,
            }
        ]

    async def test_subreddit_scopes_the_url(self):
        _, fake = await _run(_REDDIT_PAYLOAD, subreddit="MachineLearning")
        url, params, _ = fake.calls[0]
        assert "/r/MachineLearning/search.json" in url
        assert params["q"] == "q"
        assert params["limit"] == 5   # the tool default (was 10 before the §F10 change)
        assert params["sort"] == "relevance"

    async def test_absolute_url_left_untouched(self):
        payload = {
            "data": {
                "children": [{"data": {"title": "t", "url": "https://old.reddit.com/x"}}]
            }
        }
        result, _ = await _run(payload)
        assert result[0]["url"] == "https://old.reddit.com/x"

    async def test_limit_is_clamped_to_the_adapter_cap(self):
        _, fake = await _run(_REDDIT_PAYLOAD, limit=999)
        assert fake.calls[0][1]["limit"] == mod._MAX_LIMIT
        _, fake = await _run(_REDDIT_PAYLOAD, limit=0)
        assert fake.calls[0][1]["limit"] == 1


# ── x adapter ────────────────────────────────────────────────────────────────
class TestX:
    async def test_normalizes_tweets(self):
        with mock.patch.dict(os.environ, {"X_BEARER_TOKEN": "t"}, clear=False):
            result, _ = await _run(_X_PAYLOAD, platform="x")
        assert result == [
            {
                "platform": "x",
                "title": "hello",
                "content": "hello",
                "author": "bob",
                "metrics": {"likes": 3, "retweets": 1, "replies": 0},
                "url": "https://x.com/i/web/status/1",
                "published_utc": "2026-01-01T00:00:00Z",
            }
        ]

    async def test_without_token_degrades_to_domain_search(self):
        # No X_BEARER_TOKEN → the x adapter must never fail: it degrades to a
        # domain-scoped indexed web search of x.com / twitter.com (provider seam).
        hits = [
            {
                "title": "q in a post",
                "snippet": "snippet text about q",
                "url": "https://x.com/alice/status/1",
            }
        ]
        envelope = {"status": "ok", "provider": "tavily", "results": hits}
        with mock.patch.dict(os.environ, {"X_BEARER_TOKEN": ""}, clear=False), mock.patch.object(
            mod, "_domain_search", return_value=envelope
        ) as fake:
            result = await mod._execute({"query": "q", "platform": "x"}, exec=None)
        assert fake.call_args.kwargs["domains"] == ["x.com", "twitter.com"]
        assert result and all(i["platform"] == "x" for i in result)
        assert result[0]["url"] == "https://x.com/alice/status/1"


# ── unsupported / bad input ──────────────────────────────────────────────────
class TestUnsupported:
    async def test_zhihu_degrades_to_domain_search(self):
        # zhihu has no public API but IS supported — the adapter degrades to a
        # domain-scoped indexed web search of zhihu.com.
        hits = [
            {
                "title": "知乎上关于 q 的回答",
                "snippet": "回答内容…",
                "url": "https://www.zhihu.com/question/12345",
            }
        ]
        envelope = {"status": "ok", "provider": "tavily", "results": hits}
        with mock.patch.object(mod, "_domain_search", return_value=envelope) as fake:
            result = await mod._execute({"query": "q", "platform": "zhihu"}, exec=None)
        assert fake.call_args.kwargs["domains"] == ["zhihu.com"]
        assert result and result[0]["platform"] == "zhihu"

    async def test_degrade_drops_off_host_hits(self):
        # The platform guarantee: an engine hit off zhihu.com never masquerades as social
        # content, even when the underlying search leaks one.
        envelope = {
            "status": "ok",
            "provider": "tavily",
            "results": [
                {"title": "off-site", "snippet": "s", "url": "https://evil.example/p"},
            ],
        }
        with mock.patch.object(mod, "_domain_search", return_value=envelope):
            result = await mod._execute({"query": "q", "platform": "zhihu"}, exec=None)
        assert result == []

    async def test_degrade_outage_raises_not_empty(self):
        # A search-backend outage must surface as an error — never as a fake empty result
        # ("0 evidence"), so gate diagnostics can distinguish infra-down from no-evidence.
        envelope = {
            "status": "degraded",
            "provider": "tavily",
            "results": [],
            "error": {"type": "timeout", "message": "request timed out"},
        }
        with mock.patch.object(mod, "_domain_search", return_value=envelope), pytest.raises(
            RuntimeError, match="degrade search unavailable"
        ):
            await mod._execute({"query": "q", "platform": "zhihu"}, exec=None)

    async def test_unknown_platform(self):
        with pytest.raises(RuntimeError, match="unknown platform"):
            await mod._execute({"query": "q", "platform": "myspace"}, exec=None)


# ── HTTP error mapping ───────────────────────────────────────────────────────
class TestHttpErrors:
    async def test_403_is_terminal_for_run_not_a_failure(self):
        # P3-5b: a 403 is source-scoped — it returns a terminal_for_run marker
        # instead of raising, so one denied platform never kills the task.
        result, _ = await _run(_REDDIT_PAYLOAD, status_code=403)
        assert len(result) == 1
        marker = result[0]
        assert marker["terminal_for_run"] is True
        assert marker["platform"] == "reddit"
        assert marker["http_status"] == 403
        assert "do not retry" in marker["hint"]

    async def test_429_is_terminal_for_run_not_a_failure(self):
        result, _ = await _run(_REDDIT_PAYLOAD, status_code=429)
        assert len(result) == 1
        marker = result[0]
        assert marker["terminal_for_run"] is True
        assert marker["platform"] == "reddit"
        assert marker["http_status"] == 429

    async def test_5xx_maps_to_status_error(self):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await _run(_REDDIT_PAYLOAD, status_code=500)


# ── platform selection: which platforms are actually CALLED ──────────────────
class TestPlatformSelection:
    async def test_omitted_platform_searches_every_available(self):
        # Product contract: naming no platform searches EVERY supported, available
        # platform — never an implicit reddit-only fallback.
        record, _ = await _run_spy()
        assert set(record) == set(mod._available()) == {"reddit", "x", "zhihu"}

    async def test_auto_matches_the_available_set(self):
        record, _ = await _run_spy(platform="auto")
        assert set(record) == set(mod._available())

    async def test_explicit_reddit_searches_only_reddit(self):
        record, _ = await _run_spy(platform="reddit")
        assert record == ["reddit"]

    async def test_explicit_x_searches_only_x(self):
        record, _ = await _run_spy(platform="x")
        assert record == ["x"]

    async def test_explicit_multi_string_searches_only_those(self):
        record, _ = await _run_spy(platform="reddit,x")
        assert set(record) == {"reddit", "x"}

    async def test_explicit_multi_list_searches_only_those(self):
        record, _ = await _run_spy(platform=["zhihu", "reddit"])
        assert set(record) == {"reddit", "zhihu"}

    async def test_no_available_provider_raises(self):
        # No platform runnable at all → a loud error, never an empty "all done" result.
        with mock.patch.object(mod, "_available", return_value=[]):
            with pytest.raises(RuntimeError, match="no social platform is available"):
                await mod._execute({"query": "q"}, exec=None)

    async def test_all_platforms_failing_raises_not_silent_empty(self):
        # Every platform failed → an honest error, never a silent empty list that would
        # read as "searched every platform, found nothing".
        raise_map = {p: RuntimeError(f"{p} down") for p in mod._available()}
        with pytest.raises(RuntimeError, match="no social platform could be searched"):
            await _run_spy(raise_map=raise_map)

    async def test_one_failing_adapter_merges_the_rest(self):
        record, result = await _run_spy(raise_map={"x": RuntimeError("x exploded")})
        assert set(record) == {"reddit", "x", "zhihu"}  # all three were attempted
        assert {i["platform"] for i in result} == {"reddit", "zhihu"}

    async def test_denied_platform_becomes_marker_others_contribute(self):
        # A 403 on one platform is a source-scoped marker, never a task failure; the
        # other platforms still contribute their results to the merge.
        _, result = await _run_spy(raise_map={"reddit": _http_error(403)})
        markers = [i for i in result if i.get("terminal_for_run")]
        assert len(markers) == 1 and markers[0]["platform"] == "reddit"
        assert {i["platform"] for i in result if not i.get("terminal_for_run")} == {"x", "zhihu"}

    async def test_subreddit_scope_forces_reddit_only_never_widens(self):
        # An explicit ``r/<sub>`` scope is reddit-ONLY: even with the platform omitted
        # it must resolve to reddit alone, never a fan-out that silently widens the
        # requested scope to other sources.
        record, _ = await _run_spy(subreddit="MachineLearning")
        assert record == ["reddit"]

    async def test_subreddit_scope_with_an_explicit_reddit_platform(self):
        record, _ = await _run_spy(platform="reddit", subreddit="MachineLearning")
        assert record == ["reddit"]

    async def test_subreddit_scope_on_a_non_reddit_platform_refuses_not_widens(self):
        # A request naming only non-reddit platforms cannot honour the scope, so it is
        # refused loudly rather than searched with the subreddit silently dropped.
        with pytest.raises(RuntimeError, match="subreddit scope needs the reddit platform"):
            await _run_spy(platform="x", subreddit="MachineLearning")


# ── reddit OAuth (official API) ──────────────────────────────────────────────
class TestRedditOAuth:
    async def _run_oauth(self, **extra):
        """Run a reddit search with the four OAuth creds set and the token stub."""
        fake = _FakeAsyncClient(_REDDIT_PAYLOAD)

        async def _fake_token(client, **kw):
            return "tok"

        with mock.patch.dict(
            os.environ,
            {
                "REDDIT_CLIENT_ID": "cid",
                "REDDIT_CLIENT_SECRET": "sec",
                "REDDIT_USERNAME": "u",
                "REDDIT_PASSWORD": "p",
            },
            clear=False,
        ):
            with mock.patch.object(mod, "_get_reddit_token", _fake_token):
                with mock.patch.object(mod.httpx, "AsyncClient", lambda **kw: fake):
                    args = {"query": "q", "platform": "reddit", **extra}
                    result = await mod._execute(args, exec=None)
        return result, fake

    async def test_switches_to_oauth_api_when_creds_set(self):
        result, fake = await self._run_oauth()
        url, params, kwargs = fake.calls[0]
        assert url == "https://oauth.reddit.com/search"
        assert params["q"] == "q"
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
        assert result and result[0]["platform"] == "reddit"

    async def test_oauth_subreddit_scoped(self):
        _, fake = await self._run_oauth(subreddit="MachineLearning")
        assert fake.calls[0][0] == "https://oauth.reddit.com/r/MachineLearning/search"


class TestRedditToken:
    async def test_exchanges_password_grant_and_caches(self):
        fake = _FakeAsyncClient({"access_token": "tok", "expires_in": 86400})
        tok = await mod._get_reddit_token(
            fake, client_id="cid", client_secret="sec", username="u", password="p"
        )
        assert tok == "tok"
        assert "cid" in mod._TOKEN_CACHE
        # a second call reuses the cache without a new token request
        fresh = _FakeAsyncClient({"access_token": "tok", "expires_in": 86400})
        again = await mod._get_reddit_token(
            fresh, client_id="cid", client_secret="sec", username="u", password="p"
        )
        assert again == "tok"
        assert fresh.post_calls == []

    async def test_bad_creds_hint(self):
        fake = _FakeAsyncClient(None, status_code=401)
        with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
            await mod._get_reddit_token(
                fake, client_id="cid", client_secret="sec", username="u", password="p"
            )

    async def test_missing_access_token_in_response(self):
        fake = _FakeAsyncClient({}, status_code=200)
        with pytest.raises(RuntimeError, match="access_token"):
            await mod._get_reddit_token(
                fake, client_id="cid", client_secret="sec", username="u", password="p"
            )


# ── settings → os.environ secret bridge (deps._agent calls export_secret_env) ─
class TestSecretEnv:
    def test_bridges_settings_into_os_environ(self, monkeypatch):
        for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USERNAME", "REDDIT_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(settings, "reddit_client_id", "cid")
        monkeypatch.setattr(settings, "reddit_client_secret", "sec")
        monkeypatch.setattr(settings, "reddit_username", "u")
        monkeypatch.setattr(settings, "reddit_password", "p")
        export_secret_env()
        assert os.environ["REDDIT_CLIENT_ID"] == "cid"
        assert os.environ["REDDIT_PASSWORD"] == "p"

    def test_existing_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "direct")
        monkeypatch.setattr(settings, "reddit_client_id", "from-dotenv")
        export_secret_env()
        assert os.environ["REDDIT_CLIENT_ID"] == "direct"

    def test_empty_settings_value_sets_nothing(self, monkeypatch):
        monkeypatch.delenv("REDDIT_USERNAME", raising=False)
        monkeypatch.setattr(settings, "reddit_username", "")
        export_secret_env()
        assert "REDDIT_USERNAME" not in os.environ

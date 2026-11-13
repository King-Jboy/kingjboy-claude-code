"""Tests for single-provider hedged requests and TLS socket pre-warming."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx2 import Request, Response
from pydantic import ValidationError

from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError
from tests.providers.support import immediate_admission


class DummyStream:
    """Mock stream with close tracking."""

    def __init__(self, key: str):
        self.key = key
        self.closed = False

    async def __aiter__(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
        )

    async def aclose(self):
        self.closed = True

    def close(self):
        self.closed = True


def _pooled_config(keys: list[str], hedge_delay: float = 0.0) -> ProviderConfig:
    return ProviderConfig(
        api_key=keys[0],
        base_url="https://openrouter.ai/api/v1",
        api_keys=tuple(keys),
        key_rate_limit=40,
        key_rate_window=60.0,
        rate_limit=40 * len(keys),
        rate_window=60.0,
        key_hedge_delay_seconds=hedge_delay,
    )


def test_settings_key_hedge_delay_seconds_validation():
    """KEY_HEDGE_DELAY_SECONDS defaults to 0.0 and rejects negative numbers."""
    s_default = Settings(_env_file=None)
    assert s_default.key_hedge_delay_seconds == 0.0

    s_custom = Settings(_env_file=None, KEY_HEDGE_DELAY_SECONDS="0.8")
    assert s_custom.key_hedge_delay_seconds == 0.8

    with pytest.raises(ValidationError):
        Settings(_env_file=None, KEY_HEDGE_DELAY_SECONDS="-0.5")


@pytest.mark.asyncio
async def test_hedged_racing_fast_first_key_does_not_fire_second_key():
    """When Key 1 responds faster than hedge delay, Key 2 is never invoked."""
    keys = ["key-1", "key-2"]
    config = _pooled_config(keys, hedge_delay=0.1)
    provider = OpenRouterProvider(config, admission=immediate_admission())

    calls: list[str] = []

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            calls.append(api_key)
            await asyncio.sleep(0.01)
            return DummyStream(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    stream = await provider._open_chat_stream({"model": "test-model"})
    assert stream is not None
    assert calls == ["key-1"]


@pytest.mark.asyncio
async def test_hedged_racing_slow_first_key_fires_second_key_and_second_wins():
    """When Key 1 stalls longer than hedge delay, Key 2 is speculatively launched and wins."""
    keys = ["key-slow", "key-fast"]
    config = _pooled_config(keys, hedge_delay=0.03)
    provider = OpenRouterProvider(config, admission=immediate_admission())

    calls: list[str] = []
    streams_created: dict[str, DummyStream] = {}

    async def fake_create_for_key(key: str, **kwargs):
        calls.append(key)
        st = DummyStream(key)
        streams_created[key] = st
        if key == "key-slow":
            await asyncio.sleep(0.3)
        else:
            await asyncio.sleep(0.01)
        return st

    mock_client = MagicMock()

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            return await fake_create_for_key(api_key, **kwargs)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    stream_adapter = await provider._open_chat_stream({"model": "test-model"})
    assert stream_adapter is not None
    # Key-fast should have won
    assert stream_adapter._stream.key == "key-fast"
    assert "key-slow" in calls
    assert "key-fast" in calls


@pytest.mark.asyncio
async def test_hedged_race_closes_a_loser_that_finished_in_the_same_tick():
    """An unused upstream stream keeps generating, and billing, until closed."""
    keys = ["key-a", "key-b"]
    config = _pooled_config(keys, hedge_delay=0.01)
    provider = OpenRouterProvider(config, admission=immediate_admission())
    release = asyncio.Event()
    streams: dict[str, DummyStream] = {}

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            streams[api_key] = DummyStream(api_key)
            await release.wait()
            return streams[api_key]

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    opening = asyncio.create_task(provider._open_chat_stream({"model": "test-model"}))
    while len(streams) < 2:
        await asyncio.sleep(0.005)
    release.set()
    winner = await opening
    for _ in range(10):
        await asyncio.sleep(0)

    loser = next(s for key, s in streams.items() if key != winner._stream.key)
    assert not winner._stream.closed
    assert loser.closed


@pytest.mark.asyncio
async def test_hedged_racing_first_key_rate_limited_failover_to_second():
    """When Key 1 returns 429 RateLimitError, failover continues smoothly to Key 2."""
    import openai

    keys = ["key-rate-limited", "key-working"]
    config = _pooled_config(keys, hedge_delay=0.05)
    provider = OpenRouterProvider(config, admission=immediate_admission())

    calls: list[str] = []

    async def fake_create_for_key(key: str, **kwargs):
        calls.append(key)
        if key == "key-rate-limited":
            req = Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = Response(429, request=req, headers={"retry-after": "5"})
            raise openai.RateLimitError("rate limited", response=resp, body=None)
        await asyncio.sleep(0.01)
        return DummyStream(key)

    mock_client = MagicMock()

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            return await fake_create_for_key(api_key, **kwargs)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    stream_adapter = await provider._open_chat_stream({"model": "test-model"})
    assert stream_adapter._stream.key == "key-working"
    assert calls == ["key-rate-limited", "key-working"]


@pytest.mark.asyncio
async def test_tls_warmup_issues_head_request():
    """Warmup sends a lightweight HEAD request to base_url without raising errors."""
    config = _pooled_config(["key-1"])
    provider = OpenRouterProvider(config, admission=immediate_admission())

    mock_raw_http = MagicMock()
    mock_raw_http.head = AsyncMock()
    mock_raw_http.aclose = AsyncMock()
    provider._client._client = mock_raw_http

    await provider.warmup()
    mock_raw_http.head.assert_awaited_once_with(
        "https://openrouter.ai/api/v1", timeout=3.0
    )

    # Calling cleanup cancels keepalive task without errors
    await provider.cleanup()


@pytest.mark.asyncio
async def test_hedged_open_does_not_exceed_admission_concurrency_bulkhead():
    """When admission concurrency has no spare slot, hedging waits on Key 1 instead of breaching max_concurrency."""
    from free_claude_code.providers.admission import ProviderAdmissionController

    keys = ["key-1", "key-2"]
    config = _pooled_config(keys, hedge_delay=0.02)
    admission = ProviderAdmissionController(
        provider_name="open_router",
        rate_limit=100,
        rate_window=60.0,
        max_concurrency=1,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )
    provider = OpenRouterProvider(config, admission=admission)

    calls: list[str] = []

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            calls.append(api_key)
            await asyncio.sleep(0.05)
            return DummyStream(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    session = admission.new_retry_session()
    stream, _, attempt = await provider._create_stream({"model": "test-model"}, session)
    await attempt.aclose()
    await provider.cleanup()

    assert stream is not None
    assert calls == ["key-1"]


@pytest.mark.asyncio
async def test_hedged_racing_switches_key_when_headers_arrive_fast_but_first_token_stalls():
    """Key 1 returns HTTP 200 and an empty role chunk immediately, then stalls before first token."""
    keys = ["key-stalled-after-headers", "key-fast-tokens"]
    config = _pooled_config(keys, hedge_delay=0.03)
    provider = OpenRouterProvider(config, admission=immediate_admission())

    class PostHeaderStallStream:
        def __init__(self, key: str):
            self.key = key
            self.closed = False

        async def __aiter__(self):
            # Empty role-only prologue chunk sent right after HTTP 200 headers
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(role="assistant", content=""),
                        finish_reason=None,
                    )
                ]
            )
            if self.key == "key-stalled-after-headers":
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.005)
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="pong"),
                        finish_reason="stop",
                    )
                ]
            )

        async def aclose(self):
            self.closed = True

    streams: dict[str, PostHeaderStallStream] = {}

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            st = PostHeaderStallStream(api_key)
            streams[api_key] = st
            return st

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    stream_adapter = await provider._open_chat_stream({"model": "test-model"})
    chunks = [c async for c in stream_adapter]
    assert stream_adapter._stream.key == "key-fast-tokens"
    assert len(chunks) == 2
    assert chunks[1].choices[0].delta.content == "pong"
    for _ in range(10):
        await asyncio.sleep(0)
    assert streams["key-stalled-after-headers"].closed is True


def _rate_limited(api_key: str) -> Exception:
    import openai

    req = Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = Response(429, request=req)
    return openai.RateLimitError(f"{api_key} rate limited", response=resp, body=None)


@pytest.mark.asyncio
async def test_hedged_race_falls_back_to_the_pool_when_both_racers_hit_key_errors():
    """Two rate-limited racers say nothing about the rest of the pool."""
    keys = ["key-a", "key-b", "key-c"]
    config = _pooled_config(keys, hedge_delay=0.01)
    provider = OpenRouterProvider(config, admission=immediate_admission())
    calls: list[str] = []

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            calls.append(api_key)
            if api_key != "key-c":
                await asyncio.sleep(0.03)
                raise _rate_limited(api_key)
            return DummyStream(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    stream_adapter = await provider._open_chat_stream({"model": "test-model"})

    assert stream_adapter._stream.key == "key-c"
    assert calls[:2] == ["key-a", "key-b"]


@pytest.mark.asyncio
async def test_hedging_never_races_more_than_two_keys():
    """Each racer is a billed generation; a silent-thinking model would pay 3x."""
    keys = ["key-a", "key-b", "key-c"]
    config = _pooled_config(keys, hedge_delay=0.01)
    provider = OpenRouterProvider(config, admission=immediate_admission())
    calls: list[str] = []

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            calls.append(api_key)
            await asyncio.sleep(0.08)
            return DummyStream(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    await provider._open_chat_stream({"model": "test-model"})

    assert calls == ["key-a", "key-b"]


def _bad_request(api_key: str) -> Exception:
    import openai

    req = Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = Response(400, request=req)
    return openai.BadRequestError(f"{api_key} rejected", response=resp, body=None)


@pytest.mark.asyncio
async def test_hedged_race_keeps_a_request_error_when_the_other_racer_is_rate_limited():
    """A 400 that finishes before a 429 is the error the client should see."""
    import openai

    keys = ["key-a", "key-b", "key-c"]
    config = _pooled_config(keys, hedge_delay=0.01)
    provider = OpenRouterProvider(config, admission=immediate_admission())
    calls: list[str] = []

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            calls.append(api_key)
            if api_key == "key-a":
                await asyncio.sleep(0.02)
                raise _bad_request(api_key)
            if api_key == "key-b":
                await asyncio.sleep(0.05)
                raise _rate_limited(api_key)
            return DummyStream(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    with pytest.raises(openai.BadRequestError):
        await provider._open_chat_stream({"model": "test-model"})

    assert calls == ["key-a", "key-b"]


@pytest.mark.asyncio
async def test_usage_only_prologue_does_not_win_the_hedge():
    keys = ["key-usage", "key-fast"]
    config = _pooled_config(keys, hedge_delay=0.03)
    provider = OpenRouterProvider(config, admission=immediate_admission())

    class UsageThenContent:
        def __init__(self, key: str):
            self.key = key
            self.closed = False

        async def __aiter__(self):
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
            )
            await asyncio.sleep(0.5 if self.key == "key-usage" else 0.01)
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="pong"),
                        finish_reason="stop",
                    )
                ]
            )

        async def aclose(self):
            self.closed = True

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            return UsageThenContent(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    stream_adapter = await provider._open_chat_stream({"model": "test-model"})
    assert stream_adapter._stream.key == "key-fast"


@pytest.mark.asyncio
async def test_an_empty_stream_does_not_win_the_hedge():
    keys = ["key-empty", "key-fast"]
    config = _pooled_config(keys, hedge_delay=0.05)
    provider = OpenRouterProvider(config, admission=immediate_admission())
    calls: list[str] = []

    class MaybeEmpty:
        def __init__(self, key: str):
            self.key = key

        async def __aiter__(self):
            if self.key == "key-empty":
                return
            await asyncio.sleep(0.01)
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
            )

        async def aclose(self):
            return None

    def fake_with_options(api_key: str):
        sub_client = MagicMock()

        async def _call(**kwargs):
            calls.append(api_key)
            return MaybeEmpty(api_key)

        sub_client.chat.completions.create = AsyncMock(side_effect=_call)
        return sub_client

    mock_client = MagicMock()
    mock_client.with_options = MagicMock(side_effect=fake_with_options)
    provider._client = mock_client

    with pytest.raises(TruncatedProviderStreamError):
        await provider._open_chat_stream({"model": "test-model"})

    assert calls == ["key-empty"]


@pytest.mark.asyncio
async def test_a_pool_with_every_key_disabled_reports_authentication_not_rate_limit():
    from free_claude_code.core.failures import ExecutionFailure, FailureKind

    keys = ["key-a", "key-b"]
    provider = OpenRouterProvider(_pooled_config(keys), admission=immediate_admission())
    assert provider.api_key_pool is not None
    for key in keys:
        for _ in range(3):
            provider.api_key_pool.mark_failed(key)

    with pytest.raises(ExecutionFailure) as exc_info:
        await provider._open_chat_stream_sequential({"model": "test-model"})

    assert exc_info.value.kind is FailureKind.AUTHENTICATION
    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False

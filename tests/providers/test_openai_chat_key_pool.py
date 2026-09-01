"""Integration tests for multi-key rotation and failover in OpenAIChatProvider."""

from unittest.mock import AsyncMock, MagicMock

import httpx2
import openai
import pytest

from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat.provider import OpenAIChatProvider
from tests.providers.support import profiled_provider


def _pooled_config(keys: list[str], key_usage_limit: int = 1000) -> ProviderConfig:
    return ProviderConfig(
        api_key=keys[0],
        api_keys=tuple(keys),
        key_usage_limit=key_usage_limit,
        base_url="https://api.groq.com/openai/v1",
        rate_limit=10_000,
        rate_window=60,
    )


def _wire_fake_client(provider: OpenAIChatProvider, on_create):
    attempted_keys = []

    def fake_with_options(*, api_key: str):
        attempted_keys.append(api_key)
        mock_client = MagicMock()

        async def _create(**kw):
            return await on_create(api_key, **kw)

        mock_client.chat.completions.create = AsyncMock(side_effect=_create)
        return mock_client

    provider._client.with_options = MagicMock(side_effect=fake_with_options)
    return attempted_keys


@pytest.mark.asyncio
async def test_consecutive_requests_spread_across_all_pooled_keys():
    """N successful requests in a row should use N different keys in LRU order."""
    keys = [f"key-{i}" for i in range(5)]
    provider = profiled_provider("groq", _pooled_config(keys))

    async def on_create(api_key: str, **kw):
        return MagicMock(name="stream")

    attempted_keys = _wire_fake_client(provider, on_create)

    for _ in range(len(keys)):
        await provider._open_chat_stream({"model": "llama-3.3-70b-versatile"})

    assert attempted_keys == keys, (
        f"Expected requests to draw fresh keys from the pool in order, got: {attempted_keys}"
    )


@pytest.mark.asyncio
async def test_rate_limited_key_fails_over_within_same_request():
    """A 429 on the first key hops to the second key within the same request."""
    keys = ["key-A", "key-B"]
    provider = profiled_provider("groq", _pooled_config(keys))

    failed_once = {"key": None}

    async def on_create(api_key: str, **kw):
        if failed_once["key"] is None:
            failed_once["key"] = api_key
            req = httpx2.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            )
            resp = httpx2.Response(429, request=req, headers={"retry-after": "60"})
            raise openai.RateLimitError("rate limited", response=resp, body=None)
        return MagicMock(name="stream")

    attempted_keys = _wire_fake_client(provider, on_create)

    stream = await provider._open_chat_stream({"model": "llama-3.3-70b-versatile"})
    assert stream is not None
    assert len(attempted_keys) == 2
    assert attempted_keys[0] == "key-A"
    assert attempted_keys[1] == "key-B"

    assert provider._key_pool is not None
    assert not provider._key_pool._key_index["key-A"].is_available()
    assert provider._key_pool._key_index["key-B"].is_available()


@pytest.mark.asyncio
async def test_auth_error_key_fails_over_within_same_request():
    """A 401 on the first key hops to the second key within the same request."""
    keys = ["bad-key", "good-key"]
    provider = profiled_provider("groq", _pooled_config(keys))

    failed_once = {"key": None}

    async def on_create(api_key: str, **kw):
        if failed_once["key"] is None:
            failed_once["key"] = api_key
            req = httpx2.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            )
            resp = httpx2.Response(401, request=req)
            raise openai.AuthenticationError("invalid key", response=resp, body=None)
        return MagicMock(name="stream")

    attempted_keys = _wire_fake_client(provider, on_create)

    stream = await provider._open_chat_stream({"model": "llama-3.3-70b-versatile"})
    assert stream is not None
    assert attempted_keys == ["bad-key", "good-key"]

    assert provider._key_pool is not None
    assert provider._key_pool._key_index["bad-key"].consecutive_failures == 1
    assert provider._key_pool._key_index["good-key"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_exhausted_key_is_skipped_on_next_request():
    """A key reaching its usage limit is skipped in subsequent requests."""
    keys = ["key-A", "key-B"]
    provider = profiled_provider("groq", _pooled_config(keys, key_usage_limit=1))

    async def on_create(api_key: str, **kw):
        return MagicMock(name="stream")

    attempted_keys = _wire_fake_client(provider, on_create)

    # First request uses key-A and exhausts its limit of 1
    await provider._open_chat_stream({"model": "x"})
    assert attempted_keys[-1] == "key-A"

    # Second request must use key-B
    await provider._open_chat_stream({"model": "x"})
    assert attempted_keys[-1] == "key-B"

    # Third request: key-B also used 1 time (exhausted) -> returns None from get_next_key
    assert provider._key_pool is not None
    assert provider._key_pool.get_next_key() is None


def test_single_key_provider_creates_no_key_pool():
    """Single key configurations do not create a KeyPool."""
    config = ProviderConfig(
        api_key="solo-key",
        api_keys=("solo-key",),
        base_url="https://api.groq.com/openai/v1",
        rate_limit=10_000,
        rate_window=60,
    )
    provider = profiled_provider("groq", config)
    assert provider._key_pool is None
    assert provider.key_pool_status() is None

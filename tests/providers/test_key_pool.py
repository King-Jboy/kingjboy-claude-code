"""Unit and regression tests for API Key Pool (LRU rotation and provider failover)."""

import time
from unittest.mock import MagicMock

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.key_pool import (
    _MAX_CONSECUTIVE_FAILURES,
    KeyPool,
)


def test_key_pool_rotation():
    """LRU rotation: keys are returned in order, then wrap evenly."""
    keys = ["key1", "key2", "key3"]
    pool = KeyPool(keys, usage_limit=2)

    assert pool.get_next_key() == "key1"
    assert pool.get_next_key() == "key2"
    assert pool.get_next_key() == "key3"
    assert pool.get_next_key() == "key1"  # wraps around evenly

    # Exhaust key1 via usage limit — it should then be skipped
    pool2 = KeyPool(keys, usage_limit=2)
    pool2.mark_key_used("key1")
    pool2.mark_key_used("key1")  # key1 now exhausted
    result = pool2.get_next_key()
    assert result in ("key2", "key3")

    # Exhaust all keys -> returns None
    pool3 = KeyPool(["k1", "k2"], usage_limit=1)
    pool3.mark_key_used("k1")
    pool3.mark_key_used("k2")
    assert pool3.get_next_key() is None


def test_key_pool_lru_even_spread():
    """LRU picks the longest-idle key — spreads load evenly across all keys."""
    pool = KeyPool(["k1", "k2", "k3", "k4", "k5", "k6", "k7"], usage_limit=1000)
    results = [pool.get_next_key() for _ in range(14)]
    round1 = results[:7]
    round2 = results[7:]
    assert round1 == ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]
    assert round2 == ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]


def test_key_pool_failure():
    """One mark_key_failed puts key on cooldown — it is skipped but not permanently dead."""
    keys = ["key1", "key2"]
    pool = KeyPool(keys, usage_limit=10)

    assert pool.get_next_key() == "key1"
    pool.mark_key_failed("key1")

    # key1 is on cooldown — should skip it and return key2
    assert pool.get_next_key() == "key2"

    key1_info = pool._key_index["key1"]
    assert not key1_info.failed
    assert key1_info.consecutive_failures == 1
    assert key1_info.rate_limited_until > time.monotonic()


def test_key_pool_hard_cooldown_after_consecutive_failures():
    """After _MAX_CONSECUTIVE_FAILURES, a key enters a 20-min hard cooldown."""
    keys = ["key1", "key2"]
    pool = KeyPool(keys, usage_limit=10)

    for _ in range(_MAX_CONSECUTIVE_FAILURES):
        pool.mark_key_failed("key1")
    key1_info = pool._key_index["key1"]
    assert not key1_info.failed
    assert key1_info.rate_limited_until > time.monotonic()
    assert not key1_info.is_available()

    for _ in range(_MAX_CONSECUTIVE_FAILURES):
        pool.mark_key_failed("key2")
    assert not pool._key_index["key2"].is_available()

    # Both keys cooling down -> pool exhausted for now
    assert pool.get_next_key() is None

    # After cooldown expires, keys recover automatically
    key1_info.rate_limited_until = time.monotonic() - 1.0
    pool._key_index["key2"].rate_limited_until = time.monotonic() - 1.0
    assert pool.get_next_key() is not None


def test_key_pool_failure_resets_on_success():
    """A successful request resets the consecutive failure counter."""
    keys = ["key1", "key2"]
    pool = KeyPool(keys, usage_limit=100)

    pool.mark_key_failed("key1")
    key1_info = pool._key_index["key1"]
    assert key1_info.consecutive_failures == 1

    pool.mark_key_used("key1")
    assert key1_info.consecutive_failures == 0


def test_key_pool_cooldown_temporary():
    """A key on failure cooldown becomes available again after the window."""
    keys = ["key1", "key2"]
    pool = KeyPool(keys, usage_limit=10)

    pool.mark_key_failed("key1")
    key1_info = pool._key_index["key1"]

    assert pool.get_next_key() == "key2"

    # Simulate cooldown expiry
    key1_info.rate_limited_until = time.monotonic() - 1.0

    assert pool.get_next_key() == "key1"


def test_key_pool_unlimited_usage_never_exhausts():
    """usage_limit <= 0 means no cap — keys never exhaust on request counts."""
    pool = KeyPool(["key1"], usage_limit=0)
    for _ in range(5000):
        pool.mark_key_used("key1")

    key1_info = pool._key_index["key1"]
    assert key1_info.usage_count == 5000
    assert key1_info.exhausted is False
    assert pool.get_next_key() == "key1"


def test_key_pool_usage_window_resets_exhausted_key():
    """A key that hits its cap recovers once its usage window rolls over."""
    pool = KeyPool(["key1"], usage_limit=2, usage_window_seconds=86400.0)
    key1_info = pool._key_index["key1"]

    pool.mark_key_used("key1")
    pool.mark_key_used("key1")
    assert key1_info.exhausted is True
    assert pool.get_next_key() is None

    # Simulate window elapsing
    key1_info._usage_window_reset_at = time.monotonic() - 1.0

    assert pool.get_next_key() == "key1"
    assert key1_info.usage_count == 0
    assert key1_info.exhausted is False


def test_key_pool_status_summary():
    """Status reflects ready, cooling, and retired counts for Admin UI."""
    pool = KeyPool(["k1", "k2", "k3"], usage_limit=10)
    status = pool.status()
    assert status.size == 3
    assert status.ready == 3
    assert status.cooling == 0
    assert status.retired == 0

    pool.mark_key_failed("k1")
    status = pool.status()
    assert status.ready == 2
    assert status.cooling == 1
    assert status.retired == 0

    for _ in range(_MAX_CONSECUTIVE_FAILURES):
        pool.mark_key_failed("k2")
    status = pool.status()
    assert status.ready == 1
    assert status.cooling == 1
    assert status.retired == 1


@pytest.mark.asyncio
async def test_run_key_local_hops_on_auth_error():
    """run_key_local walks to the next key when encountering an authentication failure."""
    keys = ["bad_key", "good_key"]

    def client_factory(key: str) -> AsyncOpenAI:
        mock = MagicMock(spec=AsyncOpenAI)
        mock.api_key = key
        return mock

    pool = KeyPool(keys, client_factory=client_factory)

    attempted_keys = []

    async def operation(client: AsyncOpenAI) -> str:
        attempted_keys.append(client.api_key)
        if client.api_key == "bad_key":
            req = httpx2.Request("POST", "https://api.test/v1/chat")
            resp = httpx2.Response(401, request=req)
            raise openai.AuthenticationError("invalid key", response=resp, body=None)
        return "success"

    result = await pool.run_key_local(operation)
    assert result == "success"
    assert attempted_keys == ["bad_key", "good_key"]
    assert pool._key_index["bad_key"].consecutive_failures == 1
    assert pool._key_index["good_key"].consecutive_failures == 0


@pytest.mark.asyncio
async def test_run_key_local_hops_on_rate_limit():
    """run_key_local hops to the next key when a 429 rate limit is encountered."""
    keys = ["rate_limited_key", "fresh_key"]

    def client_factory(key: str) -> AsyncOpenAI:
        mock = MagicMock(spec=AsyncOpenAI)
        mock.api_key = key
        return mock

    pool = KeyPool(keys, client_factory=client_factory)

    attempted_keys = []

    async def operation(client: AsyncOpenAI) -> str:
        attempted_keys.append(client.api_key)
        if client.api_key == "rate_limited_key":
            req = httpx2.Request("POST", "https://api.test/v1/chat")
            resp = httpx2.Response(429, request=req, headers={"retry-after": "10"})
            raise openai.RateLimitError("rate limited", response=resp, body=None)
        return "done"

    result = await pool.run_key_local(operation)
    assert result == "done"
    assert attempted_keys == ["rate_limited_key", "fresh_key"]
    assert pool._key_index["rate_limited_key"].rate_limited_until > time.monotonic()


@pytest.mark.asyncio
async def test_run_key_local_raises_when_all_keys_exhausted():
    """run_key_local raises ExecutionFailure when no keys are available."""
    keys = ["k1", "k2"]

    def client_factory(key: str) -> AsyncOpenAI:
        mock = MagicMock(spec=AsyncOpenAI)
        mock.api_key = key
        return mock

    pool = KeyPool(keys, client_factory=client_factory)
    pool.mark_key_rate_limited("k1", 100.0)
    pool.mark_key_rate_limited("k2", 100.0)

    async def operation(client: AsyncOpenAI) -> str:
        return "should not be called"

    with pytest.raises(ExecutionFailure) as exc_info:
        await pool.run_key_local(operation)

    assert exc_info.value.status_code == 429

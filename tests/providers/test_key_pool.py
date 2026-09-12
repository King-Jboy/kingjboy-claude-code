"""Unit and regression tests for API Key Pool (LRU rotation and provider failover)."""

import asyncio
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
async def test_key_pool_paces_each_key_in_its_own_window():
    """A saturated key waits without stealing another key's independent RPM."""
    keys = ["key-A", "key-B"]

    def client_factory(key: str) -> AsyncOpenAI:
        return MagicMock(spec=AsyncOpenAI, api_key=key)

    pool = KeyPool(
        keys,
        client_factory=client_factory,
        key_rate_limit=1,
        key_rate_window=0.05,
    )
    attempted: list[str] = []

    async def operation(client: AsyncOpenAI) -> str:
        attempted.append(client.api_key)
        return client.api_key

    assert await pool.run_key_local(operation) == "key-A"
    assert await pool.run_key_local(operation) == "key-B"

    started = time.monotonic()
    assert await pool.run_key_local(operation) == "key-A"
    assert time.monotonic() - started >= 0.03
    assert attempted == ["key-A", "key-B", "key-A"]


@pytest.mark.asyncio
async def test_key_pool_uses_capacity_ready_key_before_waiting_for_lru_key():
    """A saturated older key must not stall a newer key with immediate capacity."""
    pool = KeyPool(
        ["key-A", "key-B"],
        client_factory=lambda key: MagicMock(spec=AsyncOpenAI, api_key=key),
        key_rate_limit=1,
        key_rate_window=1.0,
    )
    assert pool._key_limiters["key-A"].try_acquire()

    selected = await pool.run_key_local(lambda client: asyncio.sleep(0, client.api_key))

    assert selected == "key-B"


@pytest.mark.asyncio
async def test_key_pool_can_surface_permission_denied_without_cooling_other_keys():
    """Ambiguous provider 403 responses are request failures, not pool failures."""
    pool = KeyPool(
        ["key-A", "key-B"],
        client_factory=lambda key: MagicMock(spec=AsyncOpenAI, api_key=key),
        rotate_on_permission_denied=False,
    )
    request = httpx2.Request("POST", "https://api.test/v1/chat")
    response = httpx2.Response(403, request=request)

    async def operation(_client: AsyncOpenAI) -> str:
        raise openai.PermissionDeniedError("model denied", response=response, body=None)

    with pytest.raises(openai.PermissionDeniedError):
        await pool.run_key_local(operation)

    assert all(key.rate_limited_until == 0.0 for key in pool.keys)


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


def test_key_pool_failure_monotonic_cooldown():
    """mark_failed must never shorten an existing longer rate limit cooldown."""
    pool = KeyPool(["key1"])
    long_cooldown = 3600.0
    pool.mark_key_rate_limited("key1", long_cooldown)
    original_until = pool._key_index["key1"].rate_limited_until

    pool.mark_key_failed("key1")
    assert pool._key_index["key1"].rate_limited_until >= original_until


@pytest.mark.asyncio
async def test_key_pool_hedged_fast_path_avoids_racing():
    """When the first key returns before the hedge delay, second key is not called."""
    keys = ["k1", "k2"]
    attempted = []

    def client_factory(key: str) -> AsyncOpenAI:
        mock = MagicMock(spec=AsyncOpenAI)
        mock.api_key = key
        return mock

    pool = KeyPool(keys, client_factory=client_factory, hedge_delay_seconds=0.1)

    async def operation(client: AsyncOpenAI) -> str:
        attempted.append(client.api_key)
        await asyncio.sleep(0.01)
        return f"result_{client.api_key}"

    result = await pool.run_key_local(operation)
    assert result == "result_k1"
    assert attempted == ["k1"]


@pytest.mark.asyncio
async def test_key_pool_hedged_racing_second_key_wins():
    """When the first key is slow, the second key races and wins."""
    keys = ["slow_k1", "fast_k2"]
    attempted = []

    def client_factory(key: str) -> AsyncOpenAI:
        mock = MagicMock(spec=AsyncOpenAI)
        mock.api_key = key
        return mock

    pool = KeyPool(keys, client_factory=client_factory, hedge_delay_seconds=0.05)

    async def operation(client: AsyncOpenAI) -> str:
        attempted.append(client.api_key)
        if client.api_key == "slow_k1":
            await asyncio.sleep(1.0)
            return "slow_result"
        await asyncio.sleep(0.01)
        return "fast_result"

    t0 = time.monotonic()
    result = await pool.run_key_local(operation)
    duration = time.monotonic() - t0

    assert result == "fast_result"
    assert "slow_k1" in attempted
    assert "fast_k2" in attempted
    assert duration < 0.5  # Much faster than slow_k1's 1.0s delay


@pytest.mark.asyncio
async def test_key_pool_hedge_releases_extra_admission_permit():
    """A physical hedge holds and releases its own provider admission permit."""
    released = asyncio.Event()

    class Permit:
        async def aclose(self) -> None:
            released.set()

    pool = KeyPool(
        ["slow", "fast"],
        client_factory=lambda key: MagicMock(spec=AsyncOpenAI, api_key=key),
        hedge_delay_seconds=0.01,
        hedge_permit_factory=lambda: asyncio.sleep(0, result=Permit()),
    )

    async def operation(client: AsyncOpenAI) -> str:
        await asyncio.sleep(0.05 if client.api_key == "slow" else 0)
        return client.api_key

    assert await pool.run_key_local(operation) == "fast"
    assert released.is_set()


@pytest.mark.asyncio
async def test_key_pool_hedged_cancellation_cancels_the_running_operation():
    pool = KeyPool(
        ["k1", "k2"],
        client_factory=lambda key: MagicMock(spec=AsyncOpenAI, api_key=key),
        hedge_delay_seconds=10,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def operation(client: AsyncOpenAI) -> str:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    task = asyncio.create_task(pool.run_key_local(operation))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_key_pool_consecutive_permission_denied_stops_iteration():
    """Consecutive 403 PermissionDeniedError stops further key attempts without trying all keys."""
    keys = ["k1", "k2", "k3", "k4"]
    attempted = []
    pool = KeyPool(
        keys,
        client_factory=lambda key: MagicMock(spec=AsyncOpenAI, api_key=key),
        hedge_delay_seconds=10.0,
    )

    async def operation(client: AsyncOpenAI):
        attempted.append(client.api_key)
        raise openai.PermissionDeniedError(
            message="Forbidden",
            response=httpx2.Response(403, request=httpx2.Request("POST", "http://test")),
            body=None,
        )

    with pytest.raises(openai.PermissionDeniedError):
        await pool.run_key_local(operation)

    assert len(attempted) == 2
    assert attempted == ["k1", "k2"]


@pytest.mark.asyncio
async def test_key_pool_hedged_cleans_up_losing_stream():
    """Losing hedged task has its completed async stream closed."""
    keys = ["k1", "k2"]
    pool = KeyPool(
        keys,
        client_factory=lambda key: MagicMock(spec=AsyncOpenAI, api_key=key),
        hedge_delay_seconds=0.01,
    )
    closed = asyncio.Event()
    gate = asyncio.Event()

    class MockStream:
        def __init__(self, key: str):
            self.key = key

        async def aclose(self) -> None:
            closed.set()

    async def operation(client: AsyncOpenAI):
        await gate.wait()
        return MockStream(client.api_key)

    async def open_gate():
        await asyncio.sleep(0.03)
        gate.set()

    asyncio.create_task(open_gate())
    result = await pool.run_key_local(operation)
    assert result.key in ("k1", "k2")
    await asyncio.wait_for(closed.wait(), timeout=1.0)
    assert closed.is_set()

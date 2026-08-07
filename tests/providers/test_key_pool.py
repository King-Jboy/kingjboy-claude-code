"""Credential-pool selection, health attribution, and the provider seam."""

import asyncio
import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.streaming import anthropic_ping_frame
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers import key_pool as key_pool_module
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.key_pool import (
    MAX_COOLDOWN_SECONDS,
    MAX_POOL_WAIT_SECONDS,
    KeyFailureAction,
    KeyPool,
    PooledKeyLease,
)
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from free_claude_code.providers.openai_chat import provider as provider_module
from free_claude_code.providers.stream_recovery import RECOVERY_BUFFER_MAX_BYTES
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, profiled_provider

_BASE_URL = "https://provider.test/v1"


class _RecordingClient(AsyncOpenAI):
    """A real client that remembers whether the pool closed it."""

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key=api_key, base_url=_BASE_URL)
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


def _pool(
    keys: tuple[str, ...],
    *,
    rate_limit: int = 4,
    rate_window: float = 60.0,
) -> KeyPool:
    return KeyPool(
        keys,
        provider_name="TEST",
        rate_limit=rate_limit,
        rate_window=rate_window,
        client_factory=_RecordingClient,
    )


def _status_error(status: int, **headers: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"{_BASE_URL}/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or None)
    return httpx.HTTPStatusError(
        f"upstream returned {status}", request=request, response=response
    )


async def _keys_used(pool: KeyPool, count: int) -> list[str]:
    used: list[str] = []
    for _ in range(count):
        lease = await pool.acquire()
        used.append(str(lease.client.api_key))
    return used


def test_a_pool_needs_at_least_one_key() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _pool(())


@pytest.mark.asyncio
async def test_selection_spreads_across_keys_by_headroom_then_least_recent() -> None:
    pool = _pool(("a", "b"))

    assert await _keys_used(pool, 4) == ["a", "b", "a", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_retired_key_is_skipped() -> None:
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    assert lease.client.api_key == "a"

    assert pool.record_failure(lease, _status_error(401)) is KeyFailureAction.HOP

    assert await _keys_used(pool, 3) == ["b", "b", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_permission_denial_sidelines_the_key_without_retiring_it() -> None:
    # Providers also return 403 for a request they refuse outright, so the key
    # is cooled rather than killed; a false positive then heals on its own.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()

    action = pool.record_failure(lease, _status_error(403))

    assert action is KeyFailureAction.HOP_AMBIGUOUS
    assert await _keys_used(pool, 2) == ["b", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_permission_denial_is_survivable_by_hopping() -> None:
    pool = _pool(("a", "b"))
    attempts: list[str] = []

    async def operation(client: AsyncOpenAI) -> str:
        api_key = str(client.api_key)
        attempts.append(api_key)
        if api_key == "a":
            raise _status_error(403)
        return "served"

    assert await pool.run_key_local(operation) == "served"
    assert attempts == ["a", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_guessed_cooldown_escalates_so_a_dead_key_leaves_rotation() -> None:
    # A dead key answers with no timing at all, so every refusal is a guess.
    # Without escalation it returned to rotation each window forever, costing a
    # wasted round-trip on roughly one request in `pool size`.
    pool = _pool(("a", "b"), rate_window=2.0)
    lease = await pool.acquire()
    waits: list[float] = []

    for _ in range(4):
        before = time.monotonic()
        pool.record_failure(lease, _status_error(403))
        waits.append(pool._keys[lease.index].cooling_until - before)

    assert waits[0] == pytest.approx(2.0, abs=0.1)
    assert waits[1] == pytest.approx(20.0, abs=0.1)
    assert waits[2] == pytest.approx(200.0, abs=0.1)
    assert waits[3] == pytest.approx(MAX_COOLDOWN_SECONDS, abs=0.1)

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_stated_reset_is_obeyed_verbatim_and_never_escalated() -> None:
    # Escalating past a reset the provider itself published would idle a
    # healthy key long after it became usable again.
    pool = _pool(("a", "b"), rate_window=2.0)
    lease = await pool.acquire()

    for _ in range(3):
        before = time.monotonic()
        pool.record_failure(lease, _status_error(429, **{"retry-after": "7"}))
        assert pool._keys[lease.index].cooling_until - before == pytest.approx(
            7.0, abs=0.1
        )

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_success_clears_escalation_so_a_recovered_key_is_not_punished() -> None:
    pool = _pool(("a", "b"), rate_window=2.0)
    lease = await pool.acquire()
    key = pool._keys[lease.index]
    pool.record_failure(lease, _status_error(403))
    pool.record_failure(lease, _status_error(403))
    assert key.consecutive_guessed_failures == 2

    pool.record_success(lease)

    assert key.consecutive_guessed_failures == 0
    # A key only serves once its cooldown has elapsed, so the next refusal
    # starts the ladder over rather than resuming where it left off.
    key.cooling_until = 0.0
    before = time.monotonic()
    pool.record_failure(lease, _status_error(403))
    assert key.cooling_until - before == pytest.approx(2.0, abs=0.1)

    await pool.aclose()


@pytest.mark.asyncio
async def test_an_unauthenticated_endpoint_cannot_clear_a_dead_key_s_backoff() -> None:
    # Verified live: NVIDIA NIM and OpenRouter both serve /v1/models with an
    # invalid key. Model discovery hops through the pool, so a dead key
    # "succeeds" there and would otherwise reset its escalation every refresh.
    # One key, so the success provably lands on the key that was struck.
    pool = _pool(("a",), rate_window=2.0)
    key = pool._keys[0]
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(403))
    pool.record_failure(lease, _status_error(403))
    assert key.consecutive_guessed_failures == 2

    key.cooling_until = 0.0
    await pool.run_key_local(_succeeds, proves_credential=False)

    assert key.consecutive_guessed_failures == 2, (
        "a call that never checks the credential must not count as recovery"
    )

    await pool.aclose()


@pytest.mark.asyncio
async def test_an_authenticated_success_still_clears_the_backoff() -> None:
    pool = _pool(("a",), rate_window=2.0)
    key = pool._keys[0]
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(403))
    assert key.consecutive_guessed_failures == 1

    key.cooling_until = 0.0
    await pool.run_key_local(_succeeds)

    assert key.consecutive_guessed_failures == 0

    await pool.aclose()


async def _succeeds(client: AsyncOpenAI) -> str:
    return str(client.api_key)


@pytest.mark.asyncio
async def test_waiting_is_bounded_in_total_not_merely_per_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Checking only the instantaneous projection let concurrent traffic keep a
    # waiter alive forever: every re-cool looked individually survivable.
    monkeypatch.setattr(key_pool_module, "MAX_POOL_WAIT_SECONDS", 2.0)
    pool = _pool(("a", "b"), rate_window=1.0)

    async def keep_re_cooling() -> None:
        while True:
            for key in pool._keys:
                key.cooling_until = time.monotonic() + 1.5
            await asyncio.sleep(0.2)

    for key in pool._keys:
        key.cooling_until = time.monotonic() + 1.5
    churn = asyncio.create_task(keep_re_cooling())
    started = time.monotonic()
    try:
        with pytest.raises(ExecutionFailure) as error:
            await asyncio.wait_for(pool.acquire(), timeout=15)
    finally:
        churn.cancel()

    assert error.value.kind is FailureKind.RATE_LIMIT
    assert time.monotonic() - started < 10, "the total wait was not bounded"
    await pool.aclose()


@pytest.mark.asyncio
async def test_status_counts_ready_cooling_and_retired_keys() -> None:
    pool = _pool(("a", "b", "c"), rate_window=30.0)
    assert pool.status().as_dict() == {
        "size": 3,
        "ready": 3,
        "cooling": 0,
        "retired": 0,
        "soonest_ready_in": None,
    }

    pool.record_failure(PooledKeyLease(0, pool._keys[0].client), _status_error(401))
    pool.record_failure(PooledKeyLease(1, pool._keys[1].client), _status_error(403))
    status = pool.status()

    # 401 retires key #0, 403 cools key #1, key #2 is untouched.
    assert (status.size, status.ready, status.cooling, status.retired) == (3, 1, 1, 1)
    await pool.aclose()


@pytest.mark.asyncio
async def test_status_reports_when_no_key_is_ready() -> None:
    # The operator-facing case: everything is cooling and throughput is zero.
    pool = _pool(("a", "b"), rate_window=30.0)
    for index in (0, 1):
        pool.record_failure(
            PooledKeyLease(index, pool._keys[index].client), _status_error(403)
        )

    status = pool.status()

    assert status.ready == 0
    assert status.cooling == 2
    assert status.soonest_ready_in is not None
    assert 0 < status.soonest_ready_in <= 30.0
    await pool.aclose()


@pytest.mark.asyncio
async def test_a_request_level_refusal_does_not_count_as_a_strike() -> None:
    # Every key refusing alike proves the request was at fault, so no key may
    # be escalated toward retirement for it.
    pool = _pool(("a", "b"), rate_window=2.0)

    async def operation(client: AsyncOpenAI) -> str:
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError):
        await pool.run_key_local(operation)

    assert [key.consecutive_guessed_failures for key in pool._keys] == [0, 0]
    assert await _keys_used(pool, 2) == ["a", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_several_dead_keys_do_not_stop_a_live_one_from_serving() -> None:
    # Verified against NVIDIA NIM, which answers 403 - not 401 - for an invalid
    # key. Giving up early here stranded a working key behind two dead ones.
    pool = _pool(("dud-a", "dud-b", "dud-c", "good"))
    attempts: list[str] = []

    async def operation(client: AsyncOpenAI) -> str:
        api_key = str(client.api_key)
        attempts.append(api_key)
        if api_key.startswith("dud"):
            raise _status_error(403)
        return "served"

    assert await pool.run_key_local(operation) == "served"
    assert attempts == ["dud-a", "dud-b", "dud-c", "good"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_refusal_every_key_repeats_is_raised_as_the_request_s_fault() -> None:
    # Only exhaustion proves the refusal is not key-local, and the caller must
    # see the provider's own error rather than a bogus pool-exhausted 429.
    pool = _pool(("a", "b", "c"))
    attempts: list[str] = []

    async def operation(client: AsyncOpenAI) -> str:
        attempts.append(str(client.api_key))
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError) as error:
        await pool.run_key_local(operation)

    assert error.value.response.status_code == 403
    assert attempts == ["a", "b", "c"]
    # Refusals that proved request-level must not leave the pool sidelined.
    assert await _keys_used(pool, 3) == ["a", "b", "c"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_an_authentication_failure_still_retires_permanently() -> None:
    # 401 is unambiguous, so the key stays dead rather than cooling.
    pool = _pool(("a", "b"))
    attempts: list[str] = []

    async def operation(client: AsyncOpenAI) -> str:
        api_key = str(client.api_key)
        attempts.append(api_key)
        if api_key == "a":
            raise _status_error(401)
        return "served"

    assert await pool.run_key_local(operation) == "served"
    assert await _keys_used(pool, 3) == ["b", "b", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_rate_limit_cools_every_key_because_the_limit_is_account_scoped() -> (
    None
):
    # Both pooled providers meter per account, so one 429 refuses every key.
    # Hopping would re-send an already-declined request: OpenRouter charges the
    # retry to the same daily quota and NIM lengthens the lockout for it.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()

    action = pool.record_failure(lease, _status_error(429, **{"retry-after": "30"}))

    assert action is KeyFailureAction.ESCALATE
    assert pool.status().cooling == 2
    assert pool.status().ready == 0

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_rate_limit_charges_no_key_a_retirement_strike() -> None:
    # An account-wide refusal implicates no individual credential, so it must
    # not walk any key toward retirement.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()

    pool.record_failure(lease, _status_error(429))

    assert pool.status().retired == 0

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_short_cooldown_is_waited_out_rather_than_failed() -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(429, **{"retry-after": "0.05"}))

    reacquired = await asyncio.wait_for(pool.acquire(), timeout=5)

    assert reacquired.client.api_key == "a"

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_long_cooldown_fails_immediately_instead_of_parking() -> None:
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    # One refusal is enough: it cools the whole pool, because it describes the
    # account rather than the key it happened to arrive on.
    pool.record_failure(lease, _status_error(429, **{"retry-after": "7200"}))

    started = time.monotonic()
    with pytest.raises(ExecutionFailure) as error:
        await pool.acquire()

    assert time.monotonic() - started < MAX_POOL_WAIT_SECONDS
    assert error.value.kind is FailureKind.RATE_LIMIT
    assert error.value.status_code == 429
    assert error.value.retryable is False

    await pool.aclose()


@pytest.mark.asyncio
async def test_rate_limit_reset_epoch_milliseconds_is_read_as_a_deadline() -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    reset_ms = str(int((time.time() + 7200) * 1000))

    pool.record_failure(lease, _status_error(429, **{"x-ratelimit-reset": reset_ms}))

    with pytest.raises(ExecutionFailure, match="rate limiting this account"):
        await pool.acquire()

    await pool.aclose()


@pytest.mark.asyncio
async def test_rate_limit_reset_epoch_seconds_is_read_as_a_deadline() -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    reset_s = str(int(time.time() + 7200))

    pool.record_failure(lease, _status_error(429, **{"x-ratelimit-reset": reset_s}))

    with pytest.raises(ExecutionFailure, match="rate limiting this account"):
        await pool.acquire()

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_headerless_rate_limit_cools_for_one_window() -> None:
    pool = _pool(("a",), rate_window=7200.0)
    lease = await pool.acquire()

    pool.record_failure(lease, _status_error(429))

    with pytest.raises(ExecutionFailure, match="rate limiting this account"):
        await pool.acquire()

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_server_error_escalates_and_leaves_the_key_usable() -> None:
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    assert lease.client.api_key == "a"

    action = pool.record_failure(lease, _status_error(503))

    assert action is KeyFailureAction.ESCALATE
    # The key keeps serving: a failing backend is not a failing credential.
    assert "a" in await _keys_used(pool, 2)

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_timeout_escalates_rather_than_blaming_the_key() -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()

    action = pool.record_failure(lease, TimeoutError("upstream stalled"))

    assert action is KeyFailureAction.ESCALATE

    await pool.aclose()


@pytest.mark.asyncio
async def test_retiring_every_key_fails_fast_with_an_auth_failure() -> None:
    pool = _pool(("a", "b"))
    for _ in range(2):
        lease = await pool.acquire()
        pool.record_failure(lease, _status_error(401))

    with pytest.raises(ExecutionFailure) as error:
        await pool.acquire()

    assert error.value.kind is FailureKind.AUTHENTICATION
    assert error.value.status_code == 401
    assert error.value.retryable is False
    assert "2 configured" in error.value.message

    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_admits_the_sum_of_its_key_windows() -> None:
    pool = _pool(("a", "b", "c"), rate_limit=2)

    used = await _keys_used(pool, 6)

    assert sorted(used) == ["a", "a", "b", "b", "c", "c"]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(pool.acquire(), timeout=0.2)

    await pool.aclose()


@pytest.mark.asyncio
async def test_run_key_local_hops_past_a_key_local_failure() -> None:
    pool = _pool(("a", "b"))
    seen: list[str] = []

    async def operation(client: AsyncOpenAI) -> str:
        key = str(client.api_key)
        seen.append(key)
        if key == "a":
            raise _status_error(401)
        return key

    assert await pool.run_key_local(operation) == "b"
    assert seen == ["a", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_run_key_local_reraises_an_escalating_failure() -> None:
    pool = _pool(("a", "b"))
    attempts = 0

    async def operation(_client: AsyncOpenAI) -> str:
        nonlocal attempts
        attempts += 1
        raise _status_error(500)

    with pytest.raises(httpx.HTTPStatusError):
        await pool.run_key_local(operation)

    # Escalation belongs to the shared recovery episode, not to key hopping.
    assert attempts == 1

    await pool.aclose()


@pytest.mark.asyncio
async def test_run_key_local_stops_when_the_pool_is_exhausted() -> None:
    pool = _pool(("a", "b"))

    async def operation(_client: AsyncOpenAI) -> str:
        raise openai.AuthenticationError(
            "bad key", response=_status_error(401).response, body=None
        )

    with pytest.raises(ExecutionFailure) as error:
        await pool.run_key_local(operation)

    assert error.value.kind is FailureKind.AUTHENTICATION

    await pool.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_every_pooled_client() -> None:
    pool = _pool(("a", "b", "c"))

    await pool.aclose()

    assert [client.closed for client in _pooled_clients(pool)] == [True, True, True]


@pytest.mark.asyncio
async def test_the_metering_key_is_the_key_that_authenticates() -> None:
    pool = _pool(("a", "b"))

    for _ in range(4):
        lease = await pool.acquire()
        # The lease carries one client, so auth and metering cannot drift apart.
        assert lease.client.api_key == ("a", "b")[lease.index]

    await pool.aclose()


def _pooled_clients(pool: KeyPool) -> list[_RecordingClient]:
    clients = [key.client for key in pool._keys]
    assert all(isinstance(client, _RecordingClient) for client in clients)
    return [client for client in clients if isinstance(client, _RecordingClient)]


def _provider_config(*api_keys: str) -> ProviderConfig:
    return ProviderConfig(
        api_key=api_keys[0] if api_keys else "single",
        base_url=_BASE_URL,
        rate_limit=10,
        rate_window=60,
        api_keys=tuple(api_keys),
    )


def _pooled_provider(*api_keys: str) -> OpenAIChatProvider:
    return profiled_provider("groq", _provider_config(*api_keys))


def test_a_single_credential_provider_has_no_pool() -> None:
    assert _pooled_provider()._key_pool is None


def test_one_pooled_credential_still_has_no_pool() -> None:
    assert _pooled_provider("only")._key_pool is None


def test_multiple_credentials_build_a_pool() -> None:
    pool = _pooled_provider("a", "b", "c")._key_pool

    assert pool is not None
    assert pool.size == 3


def _patch_pooled_create(pool: KeyPool, outcomes: dict[str, object]):
    """Patch each pooled client's create with a per-key outcome."""
    patches = []
    for key in pool._keys:
        outcome = outcomes[str(key.client.api_key)]
        target = key.client.chat.completions
        if isinstance(outcome, BaseException):
            patches.append(
                patch.object(
                    target, "create", new_callable=AsyncMock, side_effect=outcome
                )
            )
        else:
            patches.append(
                patch.object(
                    target, "create", new_callable=AsyncMock, return_value=outcome
                )
            )
    return patches


@pytest.mark.asyncio
async def test_a_pooled_stream_does_not_burn_the_pool_on_a_rate_limit() -> None:
    # The second key would have served this mock, but a real provider would
    # have refused it too - the limit is on the account. Spending the pool to
    # discover that costs quota per key and, on NIM, lengthens the lockout.
    provider = _pooled_provider("a", "b")
    pool = provider._key_pool
    assert pool is not None
    outcomes: dict[str, object] = {
        "a": _status_error(429, **{"retry-after": "30"}),
        "b": "opened-stream",
    }

    with ExitStack() as stack:
        for patcher in _patch_pooled_create(pool, outcomes):
            stack.enter_context(patcher)
        with pytest.raises(httpx.HTTPStatusError):
            await provider._open_chat_stream({"model": "m", "messages": []})

    assert pool.status().ready == 0, "the refusal applies to every key"

    await provider.cleanup()


@pytest.mark.asyncio
async def test_a_key_hop_does_not_spend_a_provider_retry_attempt() -> None:
    provider = _pooled_provider("a", "b")
    pool = provider._key_pool
    assert pool is not None
    retry_session = provider._admission.new_retry_session()
    outcomes: dict[str, object] = {
        "a": _status_error(401),
        "b": "opened-stream",
    }

    with ExitStack() as stack:
        for patcher in _patch_pooled_create(pool, outcomes):
            stack.enter_context(patcher)
        stream, _body, attempt = await provider._create_stream(
            {"model": "m", "messages": []}, retry_session
        )
        await attempt.succeeded()
        await attempt.aclose()

    assert stream == "opened-stream"
    # Two keys were tried, but the shared five-attempt budget saw one attempt.
    assert retry_session.attempts_started == 1

    await provider.cleanup()


@pytest.mark.asyncio
async def test_a_server_error_still_spends_a_provider_retry_attempt() -> None:
    provider = _pooled_provider("a", "b")
    pool = provider._key_pool
    assert pool is not None
    retry_session = provider._admission.new_retry_session()
    outcomes: dict[str, object] = {
        "a": _status_error(503),
        "b": _status_error(503),
    }

    with ExitStack() as stack:
        for patcher in _patch_pooled_create(pool, outcomes):
            stack.enter_context(patcher)
        with pytest.raises(httpx.HTTPStatusError):
            await provider._create_stream({"model": "m", "messages": []}, retry_session)

    # Backend failures are not key-local, so they keep consuming the shared budget.
    assert retry_session.attempts_started == retry_session.max_attempts

    await provider.cleanup()


class _FailingStream:
    """Yield one chunk, then fail the way an interrupted upstream stream does."""

    def __init__(self, content: str) -> None:
        self._content = content

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        yield _text_chunk(self._content)
        raise TimeoutError("upstream cut the stream")


def _text_chunk(content: str):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


async def _slow_recovery(*_args: object, **_kwargs: object) -> list[str]:
    await asyncio.sleep(0.2)
    return ["event: message_stop\ndata: {}\n\n"]


async def _collect_frames(provider: OpenAIChatProvider, content: str) -> list[str]:
    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_FailingStream(content),
        ),
        patch.object(
            provider_module,
            "KEEPALIVE_INTERVAL_SECONDS",
            0.01,
        ),
        patch.object(
            provider_module._OpenAIChatStreamRunner,
            "_recovery_events",
            _slow_recovery,
        ),
    ):
        return [
            frame
            async for frame in provider.stream_response(
                make_messages_request(), request_id="req_ping"
            )
        ]


@pytest.mark.asyncio
async def test_a_committed_stream_is_kept_alive_while_recovery_waits() -> None:
    provider = _pooled_provider()

    # Exceeding the holdback budget commits the response before recovery starts.
    frames = await _collect_frames(provider, "x" * (RECOVERY_BUFFER_MAX_BYTES + 1))

    assert any(frame.startswith("event: ping") for frame in frames)

    await provider.cleanup()


@pytest.mark.asyncio
async def test_an_uncommitted_stream_stays_silent_so_errors_stay_typed() -> None:
    provider = profiled_provider(
        "groq",
        _provider_config(),
        admission=immediate_admission(provider_name="groq", max_attempts=2),
    )

    # Small output stays inside the holdback, so nothing has reached the client
    # yet and a failure must remain eligible for a typed non-2xx response.
    frames = await _collect_frames(provider, "short")

    assert not any(frame.startswith("event: ping") for frame in frames)

    await provider.cleanup()


def test_the_ping_frame_matches_the_anthropic_event_shape() -> None:
    assert anthropic_ping_frame() == 'event: ping\ndata: {"type": "ping"}\n\n'


@pytest.mark.asyncio
async def test_cleanup_releases_the_pool() -> None:
    provider = _pooled_provider("a", "b")
    pool = provider._key_pool
    assert pool is not None

    with patch.object(pool, "aclose", new_callable=AsyncMock) as closed:
        await provider.cleanup()

    closed.assert_awaited_once()
    await pool.aclose()

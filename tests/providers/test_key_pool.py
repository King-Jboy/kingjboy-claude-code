"""Credential-pool selection, health attribution, and the provider seam."""

import asyncio
import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.streaming import anthropic_ping_frame
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers import key_pool as key_pool_module
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.key_pool import (
    _AUTH_COOLDOWN_S,
    _AUTH_HARD_COOLDOWN_S,
    _MAX_ATTEMPTS_PER_KEY,
    _MAX_CONSECUTIVE_FAILURES,
    _PERMISSION_COOLDOWN_S,
    _RATE_LIMIT_COOLDOWN_S,
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


class _Clock:
    """Deterministic stand-in for the pool's clock.

    ``time`` keeps a plausible epoch base so header tests can encode resets as
    epoch seconds or milliseconds and have the magnitude heuristics read them.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_800_000_000.0 + self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    stub = _Clock()
    monkeypatch.setattr(key_pool_module, "time", stub)
    return stub


def _pool(
    keys: tuple[str, ...],
    *,
    usage_limit: int = 0,
    usage_window_seconds: float | None = None,
) -> KeyPool:
    return KeyPool(
        keys,
        provider_name="TEST",
        client_factory=_RecordingClient,
        usage_limit=usage_limit,
        usage_window_seconds=usage_window_seconds,
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
async def test_selection_rotates_least_recently_used() -> None:
    pool = _pool(("a", "b"))

    assert await _keys_used(pool, 4) == ["a", "b", "a", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_cooling_key_is_skipped_while_healthy_keys_serve() -> None:
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
async def test_an_authentication_refusal_cools_but_never_retires_the_key(
    clock: _Clock,
) -> None:
    # A single 401 must not kill the key for the life of the process: providers
    # also answer 401 during their own outages and while a fresh key propagates.
    # Two ordinary cooldowns, then the hard one - never a tombstone.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    key = pool._keys[lease.index]
    holds: list[float] = []

    for _expected in (
        _AUTH_COOLDOWN_S,
        _AUTH_COOLDOWN_S,
        _AUTH_HARD_COOLDOWN_S,
        _AUTH_HARD_COOLDOWN_S,
    ):
        clock.advance(2 * _AUTH_HARD_COOLDOWN_S)  # any prior cooldown has elapsed
        before = clock.monotonic()
        pool.record_failure(lease, _status_error(401))
        holds.append(key.cooling_until - before)

    assert holds == pytest.approx(
        [
            _AUTH_COOLDOWN_S,
            _AUTH_COOLDOWN_S,
            _AUTH_HARD_COOLDOWN_S,
            _AUTH_HARD_COOLDOWN_S,
        ]
    )
    assert key.consecutive_failures == _MAX_CONSECUTIVE_FAILURES + 1

    await pool.aclose()


@pytest.mark.asyncio
async def test_an_authentication_cooldown_expires_and_the_key_is_probed_again(
    clock: _Clock,
) -> None:
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(401))
    assert pool.status().retired == 1

    clock.advance(_AUTH_COOLDOWN_S + 1.0)

    assert pool.status().retired == 0
    # The probed key and the never-used key are both selectable again.
    assert sorted(await _keys_used(pool, 2)) == ["a", "b"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_success_clears_the_failure_streak_so_recovery_is_immediate(
    clock: _Clock,
) -> None:
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    key = pool._keys[lease.index]
    pool.record_failure(lease, _status_error(401))
    pool.record_failure(lease, _status_error(401))
    assert key.consecutive_failures == 2

    clock.advance(_AUTH_COOLDOWN_S + 1.0)
    pool.record_success(lease)

    assert key.consecutive_failures == 0
    before = clock.monotonic()
    pool.record_failure(lease, _status_error(401))
    assert key.cooling_until - before == pytest.approx(_AUTH_COOLDOWN_S)

    await pool.aclose()


@pytest.mark.asyncio
async def test_an_unauthenticated_endpoint_cannot_clear_a_dead_key_s_streak(
    clock: _Clock,
) -> None:
    # Verified live: NVIDIA NIM and OpenRouter both serve /v1/models with an
    # invalid key. Model discovery hops through the pool, so a dead key
    # "succeeds" there and would otherwise reset its streak every refresh.
    pool = _pool(("a",), usage_limit=5)
    key = pool._keys[0]
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(401))
    assert key.consecutive_failures == 1

    clock.advance(_AUTH_COOLDOWN_S + 1.0)
    await pool.run_key_local(_succeeds, proves_credential=False)

    assert key.consecutive_failures == 1, (
        "a call that never checks the credential must not count as recovery"
    )
    assert key.usage_count == 0, "free endpoints must not spend the key's budget"

    await pool.aclose()


@pytest.mark.asyncio
async def test_an_authenticated_success_still_clears_the_streak(clock: _Clock) -> None:
    pool = _pool(("a",))
    key = pool._keys[0]
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(401))
    assert key.consecutive_failures == 1

    clock.advance(_AUTH_COOLDOWN_S + 1.0)
    await pool.run_key_local(_succeeds)

    assert key.consecutive_failures == 0

    await pool.aclose()


async def _succeeds(client: AsyncOpenAI) -> str:
    return str(client.api_key)


@pytest.mark.asyncio
async def test_a_key_reaching_its_usage_budget_sits_out_until_the_window_rolls(
    clock: _Clock,
) -> None:
    # OpenRouter's free tier caps each key daily; the pool models that budget
    # locally instead of paying a 429 for the surplus requests.
    pool = _pool(("a", "b"), usage_limit=2, usage_window_seconds=3600.0)
    lease = await pool.acquire()
    pool.record_success(lease)
    clock.advance(1.0)
    pool.record_success(lease)

    assert pool._keys[lease.index].exhausted
    assert await _keys_used(pool, 2) == ["b", "b"]

    clock.advance(3601.0)

    assert (await pool.acquire()).client.api_key == "a"
    clock.advance(1.0)
    assert (await pool.acquire()).client.api_key == "b"

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_windowless_usage_budget_never_resets(clock: _Clock) -> None:
    pool = _pool(("a", "b"), usage_limit=1)
    for _ in range(2):
        lease = await pool.acquire()
        pool.record_success(lease)
        clock.advance(1.0)

    clock.advance(10 * _AUTH_HARD_COOLDOWN_S)
    status = pool.status()

    assert (status.ready, status.cooling) == (0, 2)
    assert status.soonest_ready_in is None
    with pytest.raises(ExecutionFailure, match="cooling or exhausted"):
        await pool.acquire()

    await pool.aclose()


@pytest.mark.asyncio
async def test_the_usage_budget_does_not_meter_without_a_limit() -> None:
    pool = _pool(("a",))

    for _ in range(50):
        lease = await pool.acquire()
        pool.record_success(lease)

    assert pool.status().ready == 1

    await pool.aclose()


@pytest.mark.asyncio
async def test_acquire_never_waits_for_a_cooling_pool() -> None:
    # The pool reports an empty pool immediately; the caller's recovery policy
    # owns the backoff. Parking here would stall a client connection behind a
    # cooldown the provider chose.
    pool = _pool(("a", "b"))
    for _ in range(2):
        lease = await pool.acquire()
        pool.record_failure(lease, _status_error(429, **{"retry-after": "7200"}))

    started = key_pool_module.time.monotonic()
    with pytest.raises(ExecutionFailure) as error:
        await asyncio.wait_for(pool.acquire(), timeout=5)

    assert key_pool_module.time.monotonic() - started < 1.0
    assert error.value.kind is FailureKind.RATE_LIMIT
    assert error.value.status_code == 429
    assert error.value.retryable is True

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_headerless_rate_limit_cools_for_sixty_seconds(
    clock: _Clock,
) -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    before = clock.monotonic()

    pool.record_failure(lease, _status_error(429))

    key = pool._keys[lease.index]
    assert key.cooling_until - before == pytest.approx(_RATE_LIMIT_COOLDOWN_S)

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_stated_reset_is_obeyed_verbatim(clock: _Clock) -> None:
    # Escalating past a reset the provider itself published would idle a healthy
    # key long after it became usable again.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    key = pool._keys[lease.index]

    for _ in range(3):
        clock.advance(_RATE_LIMIT_COOLDOWN_S + 1.0)
        before = clock.monotonic()
        pool.record_failure(lease, _status_error(429, **{"retry-after": "7"}))
        assert key.cooling_until - before == pytest.approx(7.0)

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_permission_denial_cools_for_sixty_seconds(clock: _Clock) -> None:
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    before = clock.monotonic()

    pool.record_failure(lease, _status_error(403))

    key = pool._keys[lease.index]
    assert key.cooling_until - before == pytest.approx(_PERMISSION_COOLDOWN_S)

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_stated_reset_of_zero_is_read_as_no_statement(clock: _Clock) -> None:
    # A cooldown of zero has already elapsed by the time it is set, so obeying
    # it verbatim left the key instantly selectable and the same refusal
    # repeated at full speed. Zero states no timing; the default covers that.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    before = clock.monotonic()

    pool.record_failure(lease, _status_error(429, **{"retry-after": "0"}))

    key = pool._keys[lease.index]
    assert key.cooling_until - before == pytest.approx(_RATE_LIMIT_COOLDOWN_S)
    assert not key.ready(clock.monotonic())

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_reset_deadline_already_past_is_read_as_no_statement(
    clock: _Clock,
) -> None:
    # Clock skew alone is enough to put `x-ratelimit-reset` behind us, and the
    # epoch readings clamp a past deadline to zero.
    pool = _pool(("a", "b"))
    lease = await pool.acquire()
    stale = str(int(clock.time()) - 30)
    before = clock.monotonic()

    pool.record_failure(lease, _status_error(429, **{"x-ratelimit-reset": stale}))

    key = pool._keys[lease.index]
    assert key.cooling_until - before == pytest.approx(_RATE_LIMIT_COOLDOWN_S)
    assert not key.ready(clock.monotonic())

    await pool.aclose()


@pytest.mark.asyncio
async def test_rate_limit_reset_epoch_milliseconds_is_read_as_a_deadline(
    clock: _Clock,
) -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    reset_ms = str(int((clock.time() + 7200) * 1000))
    before = clock.monotonic()

    pool.record_failure(lease, _status_error(429, **{"x-ratelimit-reset": reset_ms}))

    assert pool._keys[lease.index].cooling_until - before == pytest.approx(
        7200.0, abs=1.0
    )

    await pool.aclose()


@pytest.mark.asyncio
async def test_rate_limit_reset_epoch_seconds_is_read_as_a_deadline(
    clock: _Clock,
) -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    reset_s = str(int(clock.time() + 7200))
    before = clock.monotonic()

    pool.record_failure(lease, _status_error(429, **{"x-ratelimit-reset": reset_s}))

    assert pool._keys[lease.index].cooling_until - before == pytest.approx(
        7200.0, abs=1.0
    )

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_cooldown_that_elapses_returns_the_key_to_rotation(
    clock: _Clock,
) -> None:
    pool = _pool(("a",))
    lease = await pool.acquire()
    pool.record_failure(lease, _status_error(429, **{"retry-after": "30"}))
    with pytest.raises(ExecutionFailure):
        await pool.acquire()

    clock.advance(31.0)

    assert await _keys_used(pool, 1) == ["a"]

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_refusal_that_never_cools_cannot_retry_forever(
    clock: _Clock,
) -> None:
    # How long a key cools is a number the provider chooses, so a reset it keeps
    # stating as near-zero must not be able to re-offer the same key without
    # end. Left unbounded this served no request and hammered the upstream.
    pool = _pool(("a", "b"))
    attempts = 0

    async def operation(client: AsyncOpenAI) -> str:
        nonlocal attempts
        attempts += 1
        # Each attempt outlasts the previous near-zero cooldowns, so both keys
        # keep returning to rotation; only the attempt budget can stop this.
        # Advancing a stub clock keeps that true on any runner speed - real
        # wall-clock time between awaits is microseconds on some CI machines
        # and milliseconds on others.
        clock.advance(0.01)
        raise _status_error(429, **{"retry-after": "0.001"})

    with pytest.raises(httpx.HTTPStatusError):
        await asyncio.wait_for(pool.run_key_local(operation), timeout=10)

    assert attempts == _MAX_ATTEMPTS_PER_KEY * pool.size

    await pool.aclose()


@pytest.mark.asyncio
async def test_status_counts_ready_cooling_and_retired_keys(clock: _Clock) -> None:
    pool = _pool(("a", "b", "c"))
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

    # 401 holds key #0 out as auth-cooling, 403 cools key #1, key #2 untouched.
    assert (status.size, status.ready, status.cooling, status.retired) == (3, 1, 1, 1)
    await pool.aclose()


@pytest.mark.asyncio
async def test_status_reports_when_no_key_is_ready(clock: _Clock) -> None:
    # The operator-facing case: everything is cooling and throughput is zero.
    pool = _pool(("a", "b"))
    for index in (0, 1):
        pool.record_failure(
            PooledKeyLease(index, pool._keys[index].client), _status_error(403)
        )

    status = pool.status()

    assert status.ready == 0
    assert status.cooling == 2
    assert status.soonest_ready_in is not None
    assert 0 < status.soonest_ready_in <= _PERMISSION_COOLDOWN_S
    await pool.aclose()


@pytest.mark.asyncio
async def test_a_request_level_refusal_does_not_leave_the_pool_sidelined() -> None:
    # Every key refusing alike proves the request was at fault, so no key may
    # stay out of rotation for it.
    pool = _pool(("a", "b"))

    async def operation(client: AsyncOpenAI) -> str:
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError):
        await pool.run_key_local(operation)

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
async def test_an_authentication_failure_holds_the_key_out_of_rotation() -> None:
    # 401 is unambiguous enough to stop using the key, so it leaves rotation for
    # far longer than a rate-limit cooldown - but not for the life of the
    # process; see the recovery test below.
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
async def test_every_key_refusing_authentication_fails_fast() -> None:
    # A cooling key must not turn a misconfigured pool into a long stall
    # instead of a clear "check your keys".
    pool = _pool(("a", "b"))
    for index in (0, 1):
        pool.record_failure(
            PooledKeyLease(index, pool._keys[index].client), _status_error(401)
        )

    started = key_pool_module.time.monotonic()
    with pytest.raises(ExecutionFailure) as error:
        await pool.acquire()

    assert key_pool_module.time.monotonic() - started < 1.0
    assert error.value.kind is FailureKind.AUTHENTICATION
    assert error.value.status_code == 401
    assert error.value.retryable is False
    assert "2 configured" in error.value.message

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_rollback_leaves_a_concurrent_cooldown_standing() -> None:
    # One pool serves every in-flight request, so undoing an ambiguous refusal
    # must not also discard a cooldown another request earned meanwhile - the
    # provider did ask for that one.
    pool = _pool(("a", "b"))

    async def operation(client: AsyncOpenAI) -> str:
        if str(client.api_key) == "b":
            # Stands in for a concurrent request whose own 429 cools key #0.
            pool.record_failure(
                PooledKeyLease(0, pool._keys[0].client),
                _status_error(429, **{"retry-after": "600"}),
            )
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError):
        await pool.run_key_local(operation)

    assert pool._keys[0].cooling_until - key_pool_module.time.monotonic() == (
        pytest.approx(600, abs=5)
    )
    # The key that only ever refused this request is still rolled back.
    assert pool._keys[1].ready(key_pool_module.time.monotonic())

    await pool.aclose()


@pytest.mark.asyncio
async def test_a_refusal_that_extended_nothing_rolls_back_nothing() -> None:
    # A concurrent request's 429 cooled this same key far longer than our own
    # ambiguous refusal would. Proving the refusal request-level must undo only
    # our contribution: the concurrent cooldown stands, reason and all, while
    # the refusal still counts toward the every-key-refused verdict.
    pool = _pool(("a", "b"))

    async def operation(client: AsyncOpenAI) -> str:
        if str(client.api_key) == "a":
            pool.record_failure(
                PooledKeyLease(0, pool._keys[0].client),
                _status_error(429, **{"retry-after": "600"}),
            )
        raise _status_error(403)

    with pytest.raises(httpx.HTTPStatusError):
        await pool.run_key_local(operation)

    key_a = pool._keys[0]
    assert key_a.cooling_until - key_pool_module.time.monotonic() == (
        pytest.approx(600, abs=5)
    )
    assert key_a.cooling_reason == "rate limit"
    assert pool._keys[1].ready(key_pool_module.time.monotonic())

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
async def test_run_key_local_reports_the_last_upstream_error_when_the_pool_empties() -> (
    None
):
    # The provider's own 429 is more actionable than the pool's summary of it.
    pool = _pool(("a",))

    async def operation(_client: AsyncOpenAI) -> str:
        raise _status_error(429, **{"retry-after": "30"})

    with pytest.raises(httpx.HTTPStatusError) as error:
        await pool.run_key_local(operation)

    assert error.value.response.status_code == 429

    await pool.aclose()


@pytest.mark.asyncio
async def test_run_key_local_stops_when_the_pool_is_exhausted() -> None:
    pool = _pool(("a", "b"))

    async def operation(_client: AsyncOpenAI) -> str:
        raise openai.AuthenticationError(
            "bad key",
            response=httpx2.Response(
                401,
                request=httpx2.Request("POST", f"{_BASE_URL}/chat/completions"),
            ),
            body=None,
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
async def test_pooled_stream_hops_past_a_rate_limited_key() -> None:
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
        opened = await provider._open_chat_stream({"model": "m", "messages": []})

    assert opened == "opened-stream"

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
async def test_the_first_keepalive_lands_on_the_quiet_threshold() -> None:
    # Quiet time accrues in fixed interval slices; without a shortened final
    # slice the first keepalive lands a full interval past the documented
    # threshold (0.3s of silence would first be reported at 0.4s).
    async def silent_stream():
        await asyncio.sleep(1.2)
        yield "chunk"

    started = time.monotonic()
    pings = [
        time.monotonic() - started
        async for item in provider_module._chunks_with_keepalive(
            silent_stream(), quiet_after=0.5, interval=0.3
        )
        if item is provider_module._KEEPALIVE
    ]

    assert pings, "no keepalive fired while the upstream stayed silent"
    # 0.5s is the threshold; a full interval late would be 0.6s. The margins
    # absorb timer jitter from a loaded parallel test run.
    assert 0.42 < pings[0] < 0.58, (
        f"first keepalive fired at {pings[0]:.3f}s; the threshold is 0.5s"
    )


@pytest.mark.asyncio
async def test_cleanup_releases_the_pool() -> None:
    provider = _pooled_provider("a", "b")
    pool = provider._key_pool
    assert pool is not None

    with patch.object(pool, "aclose", new_callable=AsyncMock) as closed:
        await provider.cleanup()

    closed.assert_awaited_once()
    await pool.aclose()

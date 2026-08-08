"""In-memory credential pool for providers that accept many equivalent keys.

A pool turns N interchangeable API keys into one virtual key. Each key owns an
independent rate window and health record, so a revoked or rate-limited key is
skipped while the remaining keys keep serving at full speed.

Ownership is deliberately split. Key-local outcomes are settled here: ``401``
retires a key, ``429`` cools it until the reset the provider itself reported,
and either way the caller hops to another key without spending a provider retry
attempt. Everything that is not key-specific - ``5xx``, timeouts, connection
errors - escalates to the provider-wide recovery episode owned by
:mod:`free_claude_code.providers.admission`, so a genuinely failing backend is
not hammered once per key.

``403`` sits between those two. NVIDIA NIM answers ``403`` for an invalid key,
while other providers use it to refuse a request outright, and the response
rarely says which. Retiring on it would let one refused prompt walk the pool and
kill every key, then report the result as "check the configured keys". So a
``403`` cools its key and moves on. Only when *every* key has refused the same
request is the refusal proven not to be key-local: the cooldowns that proved
wrong are undone and the provider's own error is raised to the caller.

Cooldowns are only guessed when the provider states no usable timing. A
published ``Retry-After`` or ``X-RateLimit-Reset`` is obeyed verbatim, because
idling a key past its own reset wastes capacity - but a stated reset of zero, or
one whose deadline has already passed, is timing-free in the same way an absent
header is, and is answered by the guess rather than by a cooldown that has
already elapsed. A guess instead escalates on each
consecutive refusal from the same key, so a credential that is never coming
back drops out of rotation within a couple of strikes rather than costing a
wasted round-trip every window forever. Any success clears the escalation.

All state is in memory and resets on restart. That is self-correcting: a key
still cooling upstream costs at most one wasted attempt before its fresh ``429``
re-establishes the cooldown.
"""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import openai
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.failure_policy import retry_after_seconds

T = TypeVar("T")

KeyClientFactory = Callable[[str], AsyncOpenAI]

# A cooldown at or under this bound is an ordinary per-minute window: waiting is
# cheaper than failing. Anything longer is a real wall (for example a daily cap)
# and is reported immediately rather than parking the client connection.
MAX_POOL_WAIT_SECONDS = 60.0

# Re-evaluate the pool at least this often while waiting, so a key that frees up
# early is picked up promptly instead of after the full projected wait.
_WAIT_SLICE_SECONDS = 1.0

# Only narrate waits long enough to matter; sub-second contention is normal.
_WAIT_LOG_THRESHOLD_SECONDS = 0.5

# How much each consecutive guessed cooldown multiplies the last. With a 60s
# window that walks 60s, 10min, capped - a dead key stops costing a wasted
# round-trip per window within two strikes, while a key that recovers is still
# retried and reset by its first success.
_COOLDOWN_ESCALATION_FACTOR = 10.0

# Ceiling on a guessed cooldown. Past this the key is effectively out of
# rotation, but never permanently: it is a wait, not the tombstone a 401 sets.
MAX_COOLDOWN_SECONDS = 600.0

# How long an authentication refusal holds a key out of rotation before it is
# probed again. A 401 usually means a revoked credential, but providers also
# answer 401 during their own auth outages and while a freshly issued key is
# still propagating. Retiring permanently turned a moment like that into
# capacity lost until the next restart, so a retirement expires instead. Each
# further refusal from the same key backs the next probe off, so a credential
# that really is revoked costs at most one request per interval.
RETIREMENT_PROBE_SECONDS = 300.0
MAX_RETIREMENT_SECONDS = 3600.0

# Attempts one logical operation may spend, per key in the pool. One pass maps
# which keys refuse it; the second lets a key whose cooldown genuinely elapsed
# serve after all. Past that the pool is not making progress, and the length of
# a cooldown is a number the provider chooses - so this, not the cooldown
# arithmetic, is what makes `run_key_local` terminate.
_MAX_ATTEMPTS_PER_KEY = 2

# ``X-RateLimit-Reset`` is sent as epoch milliseconds by some providers, epoch
# seconds by others, and a plain delta by the rest. Magnitude disambiguates.
_EPOCH_MILLISECONDS_FLOOR = 1e11
_EPOCH_SECONDS_FLOOR = 1e9


class KeyFailureAction(StrEnum):
    """What a caller should do after one pooled key failed."""

    HOP = "hop"
    HOP_AMBIGUOUS = "hop_ambiguous"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class KeyPoolStatus:
    """A point-in-time count of how a pool's keys are faring.

    Counts only, never key material: this crosses into the Admin API.
    """

    size: int
    ready: int
    cooling: int
    retired: int
    soonest_ready_in: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "ready": self.ready,
            "cooling": self.cooling,
            "retired": self.retired,
            "soonest_ready_in": (
                None
                if self.soonest_ready_in is None
                else round(self.soonest_ready_in, 1)
            ),
        }


@dataclass(frozen=True, slots=True)
class PooledKeyLease:
    """One admitted use of a pooled key and the client bound to that key."""

    index: int
    client: AsyncOpenAI


@dataclass(frozen=True, slots=True)
class _Waiter:
    """One caller queued for a key, in arrival order."""

    ticket: int
    skip: frozenset[int]


@dataclass(frozen=True, slots=True)
class _HealthRollback:
    """One key's health before a refusal, beside what that refusal applied.

    Both halves are needed because a pool is shared by every concurrent request
    on the loop. Undoing a refusal unconditionally would also discard a cooldown
    some other request established in the meantime - a cooldown the provider
    actually asked for - so a field is rolled back only while it still holds the
    value this refusal wrote.
    """

    previous_cooling_until: float
    previous_strikes: int
    applied_cooling_until: float
    applied_strikes: int


@dataclass(slots=True)
class _PooledKey:
    index: int
    client: AsyncOpenAI
    limiter: StrictSlidingWindowLimiter
    retired_until: float = 0.0
    retirements: int = 0
    cooling_until: float = 0.0
    last_used: float = 0.0
    consecutive_guessed_failures: int = 0

    def retired(self, now: float) -> bool:
        """Return whether an authentication refusal still holds this key out."""
        return self.retired_until > now

    def ready(self, now: float) -> bool:
        """Return whether this key may be considered for selection."""
        return self.retired_until <= now and self.cooling_until <= now

    def available_in(self, now: float) -> float:
        """Return seconds until this key could serve.

        Never infinite: a retirement expires so the key is probed again rather
        than being written off for the life of the process.
        """
        return max(
            self.retired_until - now,
            self.cooling_until - now,
            self.limiter.next_available_in(),
            0.0,
        )


class KeyPool:
    """Select among interchangeable provider credentials and track their health."""

    def __init__(
        self,
        keys: Sequence[str],
        *,
        provider_name: str,
        rate_limit: int,
        rate_window: float,
        client_factory: KeyClientFactory,
        on_capacity_change: Callable[[int], None] | None = None,
    ) -> None:
        if not keys:
            raise ValueError("A key pool requires at least one API key")
        self._provider_name = provider_name
        self._rate_window = float(rate_window)
        # Arrival order for waiting callers, and the futures used to hand
        # capacity straight to the next one instead of making it wait out a
        # polling slice. Without an order, every waiter raced on each tick and
        # the winner was arbitrary, so a request could lose repeatedly while the
        # pool served everyone around it.
        self._waiting: list[_Waiter] = []
        self._next_ticket = 0
        self._wakeups: list[asyncio.Future[None]] = []
        # Told how many keys are usable whenever that changes, so a provider-wide
        # gate can stop admitting at a rate this pool can no longer serve.
        self._on_capacity_change = on_capacity_change
        self._published_capacity = len(keys)
        self._keys = tuple(
            _PooledKey(
                index=index,
                client=client_factory(key),
                limiter=StrictSlidingWindowLimiter(rate_limit, rate_window),
            )
            for index, key in enumerate(keys)
        )
        logger.info(
            "Key pool initialized for {} ({} keys, {} req / {}s each)",
            provider_name,
            len(self._keys),
            rate_limit,
            rate_window,
        )

    @property
    def size(self) -> int:
        """Return how many keys this pool manages."""
        return len(self._keys)

    def status(self) -> KeyPoolStatus:
        """Summarize key health so operators can see silent capacity loss.

        Readiness is measured the way :meth:`acquire` measures it, through
        ``available_in``. Counting only cooldowns would report a pool whose rate
        windows are all spent as fully ready, and would suppress the "soonest
        free" hint at exactly the moment an operator needs it.
        """
        now = time.monotonic()
        live = [key for key in self._keys if not key.retired(now)]
        waits = [key.available_in(now) for key in live]
        pending = [wait for wait in waits if wait > 0.0]
        ready = len(waits) - len(pending)
        return KeyPoolStatus(
            size=len(self._keys),
            ready=ready,
            cooling=len(pending),
            retired=len(self._keys) - len(live),
            soonest_ready_in=min(pending) if pending and not ready else None,
        )

    async def acquire(self, *, exclude: Collection[int] = ()) -> PooledKeyLease:
        """Admit one attempt on the healthiest key, waiting only when worthwhile.

        ``exclude`` holds keys the caller has already tried within one logical
        operation, so a retry walks forward instead of re-testing the same key.

        Raises a terminal :class:`ExecutionFailure` when every key is retired, or
        when the soonest key would not free up within :data:`MAX_POOL_WAIT_SECONDS`.
        """
        # Retirements expire on a clock rather than an event, so recheck here:
        # this is the one place every acquisition passes through.
        self._publish_capacity()
        skip = frozenset(exclude)
        waiter = _Waiter(ticket=self._next_ticket, skip=skip)
        self._next_ticket += 1
        self._waiting.append(waiter)
        deadline: float | None = None
        try:
            while True:
                now = time.monotonic()
                my_turn = self._may_attempt(waiter, now)
                if my_turn:
                    candidate = self._select(now, skip)
                    if candidate is not None and candidate.limiter.try_acquire():
                        candidate.last_used = now
                        return PooledKeyLease(
                            index=candidate.index, client=candidate.client
                        )
                if deadline is None:
                    # Bound the whole wait, not each slice of it. Checking only
                    # the instantaneous projection let a caller wait indefinitely
                    # while other traffic kept re-cooling whichever key was next.
                    deadline = now + MAX_POOL_WAIT_SECONDS
                if my_turn:
                    await self._wait_for_capacity(skip, deadline)
                else:
                    # Someone ahead of us can use a key right now. Nothing we
                    # could observe about the pool changes until they take it
                    # and leave - which wakes us - so wait for that instead of
                    # re-reading a state only they can move. Polling it here
                    # projected a zero wait (the key is ready, just not ours)
                    # and spun every queued caller against the whole queue on
                    # every event-loop tick until the head was scheduled.
                    await self._sleep_until_capacity(_WAIT_SLICE_SECONDS)
        finally:
            self._waiting.remove(waiter)
            # Whether this caller took a key or gave up, the next in line may now
            # be able to proceed - and should not sit out a polling slice first.
            self._wake_waiters()

    def _may_attempt(self, waiter: _Waiter, now: float) -> bool:
        """Return whether this caller is next in line for a key.

        Strict arrival order, with one exception: an older caller that cannot use
        any currently ready key must not hold up one that can. Otherwise a retry
        excluding the only free key would stall the whole pool behind itself.
        """
        for queued in self._waiting:
            if queued.ticket == waiter.ticket:
                return True
            if self._select(now, queued.skip) is not None:
                return False
        return True

    def _wake_waiters(self) -> None:
        """Release every queued caller to re-check the pool immediately."""
        wakeups = self._wakeups
        self._wakeups = []
        for wakeup in wakeups:
            if not wakeup.done():
                wakeup.set_result(None)

    async def _sleep_until_capacity(self, timeout: float) -> None:
        """Sleep for ``timeout``, or until another caller frees the pool."""
        wakeup: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._wakeups.append(wakeup)
        try:
            await asyncio.wait_for(wakeup, timeout)
        except TimeoutError:
            pass
        finally:
            if wakeup in self._wakeups:
                self._wakeups.remove(wakeup)

    def _select(
        self, now: float, skip: frozenset[int] = frozenset()
    ) -> _PooledKey | None:
        """Return the readiest key: most window headroom, then least recently used."""
        ready = [key for key in self._keys if key.index not in skip and key.ready(now)]
        if not ready:
            return None
        return max(ready, key=lambda key: (key.limiter.headroom(), -key.last_used))

    async def _wait_for_capacity(self, skip: frozenset[int], deadline: float) -> None:
        """Sleep until a key could serve, or fail when only a wall is left.

        Two very different states both read as "no key right now", and only one
        of them is worth failing over:

        * Our own rate window is full. Nothing upstream is wrong and the
          provider would serve us; the wait is bounded by ``rate_window`` by
          construction. Failing here would manufacture an error out of ordinary
          throughput, which is what turned a busy pool into a client-visible
          outage. So a self-imposed wait ignores the deadline.
        * Every usable key is in a cooldown the provider itself imposed. That is
          a real wall, and parking a client connection behind an hours-long
          reset is worse than reporting it.
        """
        now = time.monotonic()
        candidates = [key for key in self._keys if key.index not in skip]
        if not candidates or all(key.retired(now) for key in candidates):
            # Every key refused authentication. A retirement does expire, so a
            # later request probes them again, but waiting out an auth refusal
            # would only hide a misconfiguration behind a long stall.
            raise self._all_keys_retired_failure()
        live = [key for key in candidates if not key.retired(now)]
        wait = min(key.available_in(now) for key in live)
        self_imposed = min(
            (
                key.limiter.next_available_in()
                for key in live
                if key.cooling_until <= now
            ),
            default=math.inf,
        )
        if math.isfinite(self_imposed):
            await self._sleep_until_capacity(min(self_imposed, _WAIT_SLICE_SECONDS))
            return
        if wait > max(deadline - now, 0.0):
            raise self._pool_exhausted_failure(wait)
        if wait >= _WAIT_LOG_THRESHOLD_SECONDS:
            logger.info(
                "{} key pool saturated; soonest key frees in {:.1f}s",
                self._provider_name,
                wait,
            )
            trace_event(
                stage="provider",
                event="provider.key_pool.wait",
                source="provider",
                provider=self._provider_name,
                wait_s=round(wait, 3),
                pool_size=len(self._keys),
            )
        await self._sleep_until_capacity(min(wait, _WAIT_SLICE_SECONDS))

    def record_failure(
        self, lease: PooledKeyLease, error: BaseException
    ) -> KeyFailureAction:
        """Attribute one failure to its key and decide whether hopping can help.

        Providers disagree on which status refuses a credential, so both
        branches below are load-bearing. Observed live 2026-08-04: OpenRouter
        answers ``401``, NVIDIA NIM answers ``403``. Pinned by
        ``smoke/product/test_key_pool_product_live.py``.
        """
        key = self._keys[lease.index]
        status = _status_code(error)
        if isinstance(error, openai.AuthenticationError) or status == 401:
            return self._retire(key, reason="authentication")
        if isinstance(error, openai.PermissionDeniedError) or status == 403:
            # A 403 is not reliably about the credential: providers also use it
            # to reject the request itself, for content policy or for a model
            # this account cannot reach. Sideline the key rather than retiring
            # it, and escalate only once every key has refused alike - which
            # proves the request, not the keys, was refused.
            self._cool(key, error, reason="permission")
            return KeyFailureAction.HOP_AMBIGUOUS
        if isinstance(error, openai.RateLimitError) or status == 429:
            self._cool(key, error, reason="rate limit")
            return KeyFailureAction.HOP
        return KeyFailureAction.ESCALATE

    def _retire(self, key: _PooledKey, *, reason: str) -> KeyFailureAction:
        """Hold a key out over an authentication refusal, but not forever."""
        now = time.monotonic()
        if not key.retired(now):
            key.retirements += 1
            backoff = RETIREMENT_PROBE_SECONDS * _COOLDOWN_ESCALATION_FACTOR ** (
                key.retirements - 1
            )
            key.retired_until = now + min(backoff, MAX_RETIREMENT_SECONDS)
            logger.warning(
                "{} key pool retiring key #{} ({}) for {:.0f}s (strike {}); "
                "{} of {} keys still usable",
                self._provider_name,
                key.index,
                reason,
                key.retired_until - now,
                key.retirements,
                self._usable_count(),
                len(self._keys),
            )
            trace_event(
                stage="provider",
                event="provider.key_pool.key_retired",
                source="provider",
                provider=self._provider_name,
                key_index=key.index,
                reason=reason,
                retired_for_s=round(key.retired_until - now, 3),
                strike=key.retirements,
                usable_keys=self._usable_count(),
                pool_size=len(self._keys),
            )
            self._publish_capacity()
        return KeyFailureAction.HOP

    def _cool(self, key: _PooledKey, error: BaseException, *, reason: str) -> None:
        cooldown = self._cooldown_seconds(key, error)
        key.cooling_until = max(key.cooling_until, time.monotonic() + cooldown)
        logger.info(
            "{} key pool cooling key #{} for {:.1f}s after upstream {} (strike {})",
            self._provider_name,
            key.index,
            cooldown,
            reason,
            key.consecutive_guessed_failures,
        )
        trace_event(
            stage="provider",
            event="provider.key_pool.key_cooling",
            source="provider",
            provider=self._provider_name,
            key_index=key.index,
            reason=reason,
            cooldown_s=round(cooldown, 3),
            strike=key.consecutive_guessed_failures,
            usable_keys=self._usable_count(),
            pool_size=len(self._keys),
        )

    def _cooldown_seconds(self, key: _PooledKey, error: BaseException) -> float:
        """Honour the provider's own reset, and escalate only when guessing.

        A stated reset is fact: obey it exactly, or a healthy key would sit idle
        past the moment it became usable. Absent one - the shape of a dead
        credential, which carries no timing at all - each further refusal from
        the same key multiplies the wait, so a key that is never coming back
        leaves the rotation instead of costing a round-trip every window.

        A reset of zero - sent literally, or left over from a deadline that has
        already passed - states no timing either. Obeying it verbatim would set
        a cooldown that has already elapsed, so ``_cool`` would not cool: the
        key stays instantly selectable and the refusal repeats at full speed.
        Those are read as the absent case and answered with the guess.
        """
        stated = retry_after_seconds(error)
        if stated is None:
            stated = _rate_limit_reset_seconds(error)
        if stated is not None and stated > 0.0:
            return stated

        key.consecutive_guessed_failures += 1
        escalation = _COOLDOWN_ESCALATION_FACTOR ** (
            key.consecutive_guessed_failures - 1
        )
        return min(max(0.0, self._rate_window) * escalation, MAX_COOLDOWN_SECONDS)

    def record_success(self, lease: PooledKeyLease) -> None:
        """Clear a key's escalation once it serves, so recovery is immediate.

        A key that serves has proved its credential, so the retirement ladder is
        cleared too: the next authentication refusal starts from the shortest
        probe rather than resuming a backoff the key has since grown out of.
        """
        key = self._keys[lease.index]
        key.consecutive_guessed_failures = 0
        key.retirements = 0
        self._publish_capacity()

    def _usable_count(self) -> int:
        now = time.monotonic()
        return sum(1 for key in self._keys if not key.retired(now))

    def _publish_capacity(self) -> None:
        """Report a change in usable keys, so a gate above can follow the pool.

        Only retirement counts here. A cooldown is short and self-clearing;
        retuning the gate on every one would make it flap without ever
        describing a different pool.
        """
        if self._on_capacity_change is None:
            return
        usable = self._usable_count()
        if usable != self._published_capacity:
            self._published_capacity = usable
            self._on_capacity_change(usable)

    def _restore(self, health: Mapping[int, _HealthRollback]) -> None:
        """Undo sidelining from refusals that proved not to be key-local.

        The strike count is rolled back with the cooldown: a key must not be
        escalated toward retirement for a refusal the request itself caused.
        Each field is restored only while it still holds the value this refusal
        wrote, so a cooldown a concurrent request established afterwards - one
        the provider did ask for - is left standing.
        """
        for index, rollback in health.items():
            key = self._keys[index]
            if key.cooling_until == rollback.applied_cooling_until:
                key.cooling_until = rollback.previous_cooling_until
            if key.consecutive_guessed_failures == rollback.applied_strikes:
                key.consecutive_guessed_failures = rollback.previous_strikes

    async def run_key_local(
        self,
        operation: Callable[[AsyncOpenAI], Awaitable[T]],
        *,
        proves_credential: bool = True,
    ) -> T:
        """Run one operation, hopping keys past key-local failures.

        For a streaming call this covers opening the stream; failures raised
        once the body is flowing belong to the recovery controller and never
        reach this method.

        Set ``proves_credential=False`` for an endpoint that serves requests
        without checking the key. Succeeding there says nothing about the
        credential, so it must not clear a dead key's escalating backoff.

        Termination is bounded three ways. Pool state bounds the ordinary case:
        retiring or cooling every key makes the next :meth:`acquire` raise
        instead of looping. An ambiguous refusal is bounded separately, by
        refusing to blame a second key for what is evidently the request. Both
        of those, though, rest on a refusal actually taking its key out of
        rotation - and how long a key cools is a number the *provider* chooses.
        A refusal that keeps stating a reset of nearly zero would re-offer the
        same key forever, so the attempt budget bounds the loop unconditionally
        and reports the upstream failure the caller can act on.
        """
        refused: dict[int, _HealthRollback] = {}
        last_refusal: BaseException | None = None
        last_error: BaseException | None = None
        budget = _MAX_ATTEMPTS_PER_KEY * len(self._keys)
        attempts = 0
        while True:
            if last_refusal is not None and len(refused) >= len(self._keys):
                # Every key refused this one request alike. That cannot be a
                # property of the keys, so undo the sidelining and report the
                # refusal to the caller who caused it. Checked before the budget
                # so the verdict that proves something always wins.
                self._restore(refused)
                logger.warning(
                    "{} key pool escalating a refusal that all {} keys rejected "
                    "alike; treating it as request-level",
                    self._provider_name,
                    len(self._keys),
                )
                raise last_refusal
            if last_error is not None and attempts >= budget:
                # The refusals stand: unlike the case above, nothing here proves
                # they were the request's fault.
                logger.warning(
                    "{} key pool spent its {}-attempt budget on one request "
                    "without progress; reporting the last upstream failure",
                    self._provider_name,
                    budget,
                )
                trace_event(
                    stage="provider",
                    event="provider.key_pool.budget_exhausted",
                    source="provider",
                    provider=self._provider_name,
                    attempts=attempts,
                    pool_size=len(self._keys),
                    exc_type=type(last_error).__name__,
                )
                raise last_error
            lease = await self.acquire(exclude=refused.keys())
            key = self._keys[lease.index]
            previous_cooling_until = key.cooling_until
            previous_strikes = key.consecutive_guessed_failures
            attempts += 1
            try:
                result = await operation(lease.client)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                action = self.record_failure(lease, error)
                if action is KeyFailureAction.ESCALATE:
                    raise
                if action is KeyFailureAction.HOP_AMBIGUOUS:
                    refused[lease.index] = _HealthRollback(
                        previous_cooling_until=previous_cooling_until,
                        previous_strikes=previous_strikes,
                        applied_cooling_until=key.cooling_until,
                        applied_strikes=key.consecutive_guessed_failures,
                    )
                    last_refusal = error
            else:
                if proves_credential:
                    self.record_success(lease)
                return result

    async def aclose(self) -> None:
        """Close every pooled client, so one failure cannot strand the others."""
        for key in self._keys:
            try:
                await key.client.close()
            except Exception as exc:
                trace_event(
                    stage="provider",
                    event="provider.key_pool.close_failed",
                    source="provider",
                    provider=self._provider_name,
                    key_index=key.index,
                    exc_type=type(exc).__name__,
                )

    def _all_keys_retired_failure(self) -> ExecutionFailure:
        return ExecutionFailure(
            kind=FailureKind.AUTHENTICATION,
            status_code=401,
            message=(
                f"Every {self._provider_name} API key in the pool was rejected "
                f"({len(self._keys)} configured). Check the configured keys."
            ),
            retryable=False,
        )

    def _pool_exhausted_failure(self, wait: float) -> ExecutionFailure:
        # Retryable: reaching here means every key sits behind a provider
        # cooldown, which is by nature temporary. Marking it terminal handed the
        # client a hard 429 and, under a fan-out, handed one to every queued
        # request at once. Letting the provider-wide recovery episode own the
        # backoff keeps a transient wall from surfacing as an outage.
        return ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message=(
                f"Every {self._provider_name} API key in the pool is rate "
                f"limited. The soonest key frees in about "
                f"{_humanize_wait(wait)}."
            ),
            retryable=True,
        )


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _rate_limit_reset_seconds(error: BaseException) -> float | None:
    """Read ``X-RateLimit-Reset`` as a delay, tolerating its three encodings."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("x-ratelimit-reset")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    now = time.time()
    if value > _EPOCH_MILLISECONDS_FLOOR:
        return max(0.0, value / 1000.0 - now)
    if value > _EPOCH_SECONDS_FLOOR:
        return max(0.0, value - now)
    return value


def _humanize_wait(wait: float) -> str:
    if wait >= 3600:
        return f"{wait / 3600:.1f} hours"
    return f"{wait / 60:.0f} minutes"

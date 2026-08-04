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

Cooldowns are only guessed when the provider states nothing. A published
``Retry-After`` or ``X-RateLimit-Reset`` is obeyed verbatim, because idling a
key past its own reset wastes capacity. A guess instead escalates on each
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


@dataclass(slots=True)
class _PooledKey:
    index: int
    client: AsyncOpenAI
    limiter: StrictSlidingWindowLimiter
    dead: bool = False
    cooling_until: float = 0.0
    last_used: float = 0.0
    consecutive_guessed_failures: int = 0

    def ready(self, now: float) -> bool:
        """Return whether this key may be considered for selection."""
        return not self.dead and self.cooling_until <= now

    def available_in(self, now: float) -> float:
        """Return seconds until this key could serve, or infinity when retired."""
        if self.dead:
            return math.inf
        return max(self.cooling_until - now, self.limiter.next_available_in(), 0.0)


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
    ) -> None:
        if not keys:
            raise ValueError("A key pool requires at least one API key")
        self._provider_name = provider_name
        self._rate_window = float(rate_window)
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
        """Summarize key health so operators can see silent capacity loss."""
        now = time.monotonic()
        retired = sum(1 for key in self._keys if key.dead)
        ready = sum(1 for key in self._keys if key.ready(now))
        waits = [
            key.cooling_until - now
            for key in self._keys
            if not key.dead and key.cooling_until > now
        ]
        return KeyPoolStatus(
            size=len(self._keys),
            ready=ready,
            cooling=len(waits),
            retired=retired,
            soonest_ready_in=min(waits) if waits and not ready else None,
        )

    async def acquire(self, *, exclude: Collection[int] = ()) -> PooledKeyLease:
        """Admit one attempt on the healthiest key, waiting only when worthwhile.

        ``exclude`` holds keys the caller has already tried within one logical
        operation, so a retry walks forward instead of re-testing the same key.

        Raises a terminal :class:`ExecutionFailure` when every key is retired, or
        when the soonest key would not free up within :data:`MAX_POOL_WAIT_SECONDS`.
        """
        skip = frozenset(exclude)
        deadline: float | None = None
        while True:
            now = time.monotonic()
            candidate = self._select(now, skip)
            if candidate is not None and candidate.limiter.try_acquire():
                candidate.last_used = now
                return PooledKeyLease(index=candidate.index, client=candidate.client)
            if deadline is None:
                # Bound the whole wait, not each slice of it. Checking only the
                # instantaneous projection let a caller wait indefinitely while
                # other traffic kept re-cooling whichever key was next.
                deadline = now + MAX_POOL_WAIT_SECONDS
            await self._wait_for_capacity(skip, deadline)

    def _select(
        self, now: float, skip: frozenset[int] = frozenset()
    ) -> _PooledKey | None:
        """Return the readiest key: most window headroom, then least recently used."""
        ready = [key for key in self._keys if key.index not in skip and key.ready(now)]
        if not ready:
            return None
        return max(ready, key=lambda key: (key.limiter.headroom(), -key.last_used))

    async def _wait_for_capacity(self, skip: frozenset[int], deadline: float) -> None:
        now = time.monotonic()
        candidates = [key for key in self._keys if key.index not in skip]
        wait = min((key.available_in(now) for key in candidates), default=math.inf)
        if not math.isfinite(wait):
            raise self._all_keys_retired_failure()
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
        await asyncio.sleep(min(wait, _WAIT_SLICE_SECONDS))

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
        if not key.dead:
            key.dead = True
            logger.warning(
                "{} key pool retiring key #{} ({}); {} of {} keys still usable",
                self._provider_name,
                key.index,
                reason,
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
                usable_keys=self._usable_count(),
                pool_size=len(self._keys),
            )
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
        """
        stated = retry_after_seconds(error)
        if stated is None:
            stated = _rate_limit_reset_seconds(error)
        if stated is not None:
            return max(0.0, stated)

        key.consecutive_guessed_failures += 1
        escalation = _COOLDOWN_ESCALATION_FACTOR ** (
            key.consecutive_guessed_failures - 1
        )
        return min(max(0.0, self._rate_window) * escalation, MAX_COOLDOWN_SECONDS)

    def record_success(self, lease: PooledKeyLease) -> None:
        """Clear a key's escalation once it serves, so recovery is immediate."""
        self._keys[lease.index].consecutive_guessed_failures = 0

    def _usable_count(self) -> int:
        return sum(1 for key in self._keys if not key.dead)

    def _restore(self, health: Mapping[int, tuple[float, int]]) -> None:
        """Undo sidelining from refusals that proved not to be key-local.

        The strike count is rolled back with the cooldown: a key must not be
        escalated toward retirement for a refusal the request itself caused.
        """
        for index, (cooling_until, strikes) in health.items():
            key = self._keys[index]
            key.cooling_until = cooling_until
            key.consecutive_guessed_failures = strikes

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

        Termination is bounded by pool state: retiring or cooling every key
        makes the next :meth:`acquire` raise instead of looping. An ambiguous
        refusal is bounded separately, by refusing to blame a second key for
        what is evidently the request.
        """
        refused: dict[int, tuple[float, int]] = {}
        last_refusal: BaseException | None = None
        while True:
            if last_refusal is not None and len(refused) >= len(self._keys):
                # Every key refused this one request alike. That cannot be a
                # property of the keys, so undo the sidelining and report the
                # refusal to the caller who caused it.
                self._restore(refused)
                logger.warning(
                    "{} key pool escalating a refusal that all {} keys rejected "
                    "alike; treating it as request-level",
                    self._provider_name,
                    len(self._keys),
                )
                raise last_refusal
            lease = await self.acquire(exclude=refused.keys())
            key = self._keys[lease.index]
            previous_health = (key.cooling_until, key.consecutive_guessed_failures)
            try:
                result = await operation(lease.client)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                action = self.record_failure(lease, error)
                if action is KeyFailureAction.ESCALATE:
                    raise
                if action is KeyFailureAction.HOP_AMBIGUOUS:
                    refused[lease.index] = previous_health
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
        return ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message=(
                f"Every {self._provider_name} API key in the pool is rate "
                f"limited. The soonest key frees in about "
                f"{_humanize_wait(wait)}."
            ),
            retryable=False,
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

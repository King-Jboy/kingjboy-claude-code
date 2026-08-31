"""In-memory credential pool for providers that accept many equivalent keys.

A pool turns N interchangeable API keys into one virtual key. Selection is
least-recently-used: every request takes the key that has been idle the
longest, so load spreads across the pool by itself and the first round of
requests walks the keys in configured order.

The pool never throttles on its own and never parks a caller. The provider is
the only judge of a key: a request is attempted on a key, and the outcome
moves that key's health. ``429`` cools the key for the reset the provider
itself stated - or sixty seconds when it stated nothing - and the caller hops
to the next key immediately. An authentication refusal cools the key for five
minutes, then twenty minutes once refusals repeat, but never retires it
outright: a provider-side auth outage or a freshly issued key still
propagating looks exactly like a revoked credential, and a tombstone would
turn that moment into capacity lost until the next restart. Any success
clears a key's failure streak, so transient hiccups never accumulate.

``403`` is treated as ambiguous, because providers disagree about it: NVIDIA
NIM answers ``403`` for an invalid key, while other providers use it to
refuse a request outright. A ``403`` cools its key and the caller hops, but
only while other keys have not refused the same request; once every key has
refused alike, the refusal is proven not key-local, the cooldowns are undone,
and the provider's own error reaches the caller.

Keys can carry a usage budget with an optional rolling window (OpenRouter's
daily cap is the motivating case): each served request counts against its
key, a key that reaches its budget sits out until the window rolls over, and
the window resets on its schedule rather than on server activity.

All state is in memory and resets on restart. That is self-correcting: a key
still cooling upstream costs at most one wasted attempt before its fresh
``429`` re-establishes the cooldown.
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
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.failure_policy import retry_after_seconds

T = TypeVar("T")

KeyClientFactory = Callable[[str], AsyncOpenAI]

# How many consecutive authentication refusals escalate a key from the ordinary
# cooldown to the hard one. The streak is cleared by any success, so only a
# credential that genuinely never works walks up this ladder.
_MAX_CONSECUTIVE_FAILURES = 3

# Cooldown after an authentication refusal, before the streak escalates.
_AUTH_COOLDOWN_S = 300.0

# Cooldown once an authentication refusal has repeated _MAX_CONSECUTIVE_FAILURES
# times in a row. Still a cooldown, never a tombstone: the key is probed again.
_AUTH_HARD_COOLDOWN_S = 1200.0

# Cooldown for a rate-limited key when the provider stated no usable timing.
_RATE_LIMIT_COOLDOWN_S = 60.0

# Cooldown for an ambiguous permission refusal while other keys are still
# untried. Short on purpose: if the refusal was about the request, every key
# refuses alike and the pool escalates within one pass.
_PERMISSION_COOLDOWN_S = 60.0

# Attempts one logical operation may spend, per key in the pool. One pass maps
# which keys refuse it; the second lets a key whose cooldown genuinely elapsed
# serve after all. Past that the pool is not making progress, and this - not
# cooldown arithmetic - is what makes `run_key_local` terminate.
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
class _HealthRollback:
    """One key's health before a refusal, beside what that refusal applied.

    Both halves are needed because a pool is shared by every concurrent request
    on the loop. Undoing a refusal unconditionally would also discard a cooldown
    some other request established in the meantime - a cooldown the provider
    actually asked for - so a field is rolled back only while it still holds the
    value this refusal wrote. When the refusal extended nothing (a longer
    concurrent cooldown already stood), both halves are equal and the rollback
    is inert: it still counts the refusal, but restores nothing.
    """

    previous_cooling_until: float
    previous_cooling_reason: str
    applied_cooling_until: float
    applied_cooling_reason: str


@dataclass(slots=True)
class _PooledKey:
    index: int
    client: AsyncOpenAI
    usage_limit: int
    usage_window_seconds: float | None
    usage_count: int = 0
    exhausted: bool = False
    consecutive_failures: int = 0
    cooling_until: float = 0.0
    cooling_reason: str = ""
    window_reset_at: float = 0.0
    # LRU stamp. Initial values are staggered so the first round of requests
    # walks the keys in configured order (key[0] first, then key[1], ...)
    # without needing a separate rotation index.
    last_used: float = 0.0
    active_requests: int = 0

    def cooling(self, now: float) -> bool:
        """Return whether a refusal still holds this key out."""
        return self.cooling_until > now

    def auth_cooling(self, now: float) -> bool:
        return self.cooling(now) and self.cooling_reason == "authentication"

    def ready(self, now: float) -> bool:
        """Return whether this key may be considered for selection."""
        return not self.exhausted and not self.cooling(now)

    def available_in(self, now: float) -> float:
        """Return seconds until this key could serve; ``math.inf`` when only a
        window-less usage budget stands in the way."""
        wait = max(self.cooling_until - now, 0.0)
        if self.exhausted:
            if self.usage_window_seconds:
                wait = max(wait, max(self.window_reset_at - now, 0.0))
            else:
                return math.inf
        return wait


class KeyPool:
    """Select among interchangeable provider credentials and track their health."""

    def __init__(
        self,
        keys: Sequence[str],
        *,
        provider_name: str,
        client_factory: KeyClientFactory,
        usage_limit: int = 0,
        usage_window_seconds: float | None = None,
    ) -> None:
        if not keys:
            raise ValueError("A key pool requires at least one API key")
        self._provider_name = provider_name
        now = time.monotonic()
        self._keys = tuple(
            _PooledKey(
                index=index,
                client=client_factory(key),
                usage_limit=usage_limit,
                usage_window_seconds=usage_window_seconds,
                window_reset_at=(
                    now + usage_window_seconds if usage_window_seconds else 0.0
                ),
                last_used=-float(len(keys) - index),
            )
            for index, key in enumerate(keys)
        )
        logger.info(
            "Key pool initialized for {} ({} keys{})",
            provider_name,
            len(self._keys),
            (
                f", {usage_limit} uses per key per {usage_window_seconds:.0f}s window"
                if usage_limit > 0 and usage_window_seconds
                else f", {usage_limit} uses per key"
                if usage_limit > 0
                else ""
            ),
        )

    @property
    def size(self) -> int:
        """Return how many keys this pool manages."""
        return len(self._keys)

    def status(self) -> KeyPoolStatus:
        """Summarize key health so operators can see silent capacity loss.

        ``retired`` counts keys held out by an authentication refusal; those
        are the ones an operator can fix by rotating credentials. Everything
        else not ready - rate limits, ambiguous refusals, spent usage budgets -
        is ``cooling`` and recovers on its own.
        """
        now = time.monotonic()
        ready = 0
        cooling = 0
        retired = 0
        soonest: list[float] = []
        for key in self._keys:
            self._roll_usage_window(key, now)
            if key.ready(now):
                ready += 1
                continue
            if key.auth_cooling(now):
                retired += 1
            else:
                cooling += 1
            wait = key.available_in(now)
            if math.isfinite(wait):
                soonest.append(wait)
        return KeyPoolStatus(
            size=len(self._keys),
            ready=ready,
            cooling=cooling,
            retired=retired,
            soonest_ready_in=min(soonest) if soonest else None,
        )

    async def acquire(self, *, exclude: Collection[int] = ()) -> PooledKeyLease:
        """Lease the available key idle the longest, or fail without waiting.

        ``exclude`` holds keys the caller has already tried within one logical
        operation, so a retry walks forward instead of re-testing the same key.

        The pool never parks a caller: when no eligible key can serve right
        now, the refusal is reported immediately and the caller's own recovery
        policy owns the backoff.
        """
        skip = frozenset(exclude)
        now = time.monotonic()
        best: _PooledKey | None = None
        for key in self._keys:
            if key.index in skip:
                continue
            self._roll_usage_window(key, now)
            if not key.ready(now):
                continue
            if (
                best is None
                or key.active_requests < best.active_requests
                or (
                    key.active_requests == best.active_requests
                    and key.last_used < best.last_used
                )
            ):
                best = key
        if best is None:
            raise self._no_available_key_failure(skip, now)
        best.active_requests += 1
        best.last_used = now
        return PooledKeyLease(index=best.index, client=best.client)

    def release(self, lease: PooledKeyLease) -> None:
        """Release an in-flight lease for a key."""
        key = self._keys[lease.index]
        if key.active_requests > 0:
            key.active_requests -= 1

    def record_failure(
        self, lease: PooledKeyLease, error: BaseException
    ) -> KeyFailureAction:
        """Attribute one failure to its key and decide whether hopping can help.

        Providers disagree on which status refuses a credential, so both
        branches below are load-bearing. Observed live 2026-08-04: OpenRouter
        answers ``401``, NVIDIA NIM answers ``403``. Pinned by
        ``smoke/product/test_key_pool_product_live.py``.
        """
        action, _applied = self._record_failure(lease, error)
        return action

    def _record_failure(
        self, lease: PooledKeyLease, error: BaseException
    ) -> tuple[KeyFailureAction, float | None]:
        """Settle one failure, returning the action and the cooldown applied.

        The applied cooldown is ``None`` when a longer concurrent cooldown
        already stood and this refusal extended nothing - the caller needs that
        to know whether its rollback would undo anything.
        """
        key = self._keys[lease.index]
        status = _status_code(error)
        if isinstance(error, openai.AuthenticationError) or status == 401:
            return KeyFailureAction.HOP, self._auth_cooldown(key)
        if isinstance(error, openai.PermissionDeniedError) or status == 403:
            # A 403 is not reliably about the credential: providers also use it
            # to reject the request itself, for content policy or for a model
            # this account cannot reach. Cool the key briefly and escalate only
            # once every key has refused alike - which proves the request, not
            # the keys, was refused.
            return (
                KeyFailureAction.HOP_AMBIGUOUS,
                self._cool(key, _PERMISSION_COOLDOWN_S, reason="permission"),
            )
        if isinstance(error, openai.RateLimitError) or status == 429:
            cooldown = self._stated_reset(error) or _RATE_LIMIT_COOLDOWN_S
            return (
                KeyFailureAction.HOP,
                self._cool(key, cooldown, reason="rate limit"),
            )
        return KeyFailureAction.ESCALATE, None

    def record_success(
        self, lease: PooledKeyLease, *, proves_credential: bool = True
    ) -> None:
        """Settle one served request: meter usage and clear the failure streak.

        Set ``proves_credential=False`` for an endpoint that serves requests
        without checking the key. Succeeding there says nothing about the
        credential and costs no budget, so it must neither reset a dead key's
        streak nor count against its usage.
        """
        key = self._keys[lease.index]
        now = time.monotonic()
        self._roll_usage_window(key, now)
        if not proves_credential:
            return
        if key.consecutive_failures:
            key.consecutive_failures = 0
        if key.usage_limit <= 0:
            return
        key.usage_count += 1
        if key.usage_count >= key.usage_limit and not key.exhausted:
            key.exhausted = True
            logger.warning(
                "{} key pool: key #{} reached its usage limit ({}/{}). "
                "Rotating to the next key.",
                self._provider_name,
                key.index,
                key.usage_count,
                key.usage_limit,
            )

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

        Termination is bounded two ways. Pool state bounds the ordinary case:
        cooling every eligible key makes the next :meth:`acquire` raise instead
        of looping. An ambiguous refusal is bounded separately, by refusing to
        blame a second key for what is evidently the request. Both rest on a
        refusal actually taking its key out of rotation, so the attempt budget
        bounds the loop unconditionally and reports the upstream failure the
        caller can act on.
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
            try:
                lease = await self.acquire(exclude=refused.keys())
            except ExecutionFailure as failure:
                if (
                    last_error is not None
                    and failure.kind is not FailureKind.AUTHENTICATION
                ):
                    # A prior attempt already surfaced the real upstream error -
                    # with its headers and status - and that beats the pool's
                    # summary of it. An authentication verdict is the exception:
                    # "every key was rejected, check the configured keys" tells
                    # the operator more than the last raw 401 did.
                    raise last_error from failure
                raise
            key = self._keys[lease.index]
            previous_cooling_until = key.cooling_until
            previous_cooling_reason = key.cooling_reason
            attempts += 1
            try:
                try:
                    result = await operation(lease.client)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    last_error = error
                    action, applied = self._record_failure(lease, error)
                    if action is KeyFailureAction.ESCALATE:
                        raise
                    if action is KeyFailureAction.HOP_AMBIGUOUS:
                        # When this refusal extended nothing (a longer concurrent
                        # cooldown stood), the rollback is inert: the refusal still
                        # counts toward the every-key-refused tally, but restoring
                        # it must not undo the other request's cooldown.
                        if applied is None:
                            applied = previous_cooling_until
                            applied_reason = previous_cooling_reason
                        else:
                            applied_reason = key.cooling_reason
                        refused[lease.index] = _HealthRollback(
                            previous_cooling_until=previous_cooling_until,
                            previous_cooling_reason=previous_cooling_reason,
                            applied_cooling_until=applied,
                            applied_cooling_reason=applied_reason,
                        )
                        last_refusal = error
                else:
                    self.record_success(lease, proves_credential=proves_credential)
                    return result
            finally:
                self.release(lease)

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

    def _no_available_key_failure(
        self, skip: frozenset[int], now: float
    ) -> ExecutionFailure:
        """Report an immediate refusal to serve, naming the soonest recovery.

        Every eligible key refused authentication: that is a configuration
        problem the operator must fix, reported as terminal. Any other holdout
        - rate limits, ambiguous refusals, spent budgets - is temporary, so the
        failure is retryable and the caller's recovery policy owns the backoff.
        """
        candidates = [key for key in self._keys if key.index not in skip] or list(
            self._keys
        )
        holdouts = [key for key in candidates if not key.ready(now)]
        if holdouts and all(key.auth_cooling(now) for key in holdouts):
            return ExecutionFailure(
                kind=FailureKind.AUTHENTICATION,
                status_code=401,
                message=(
                    f"Every {self._provider_name} API key in the pool was "
                    f"rejected ({len(self._keys)} configured). Check the "
                    f"configured keys."
                ),
                retryable=False,
            )
        finite = [
            key.available_in(now)
            for key in holdouts
            if math.isfinite(key.available_in(now))
        ]
        wait = min(finite) if finite else None
        return ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message=(
                f"Every {self._provider_name} API key in the pool is cooling "
                f"or exhausted."
                + (
                    f" The soonest key frees in about {_humanize_wait(wait)}."
                    if wait is not None
                    else ""
                )
            ),
            retryable=True,
        )

    def _auth_cooldown(self, key: _PooledKey) -> float | None:
        """Cool a key past an authentication refusal, escalating on streaks."""
        key.consecutive_failures += 1
        if key.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            cooldown = _AUTH_HARD_COOLDOWN_S
            detail = (
                f"key #{key.index} refused {key.consecutive_failures} times in a "
                f"row - hard cooldown for {_AUTH_HARD_COOLDOWN_S / 60:.0f}min "
                "before it is probed again"
            )
        else:
            cooldown = _AUTH_COOLDOWN_S
            detail = (
                f"key #{key.index} refused "
                f"({key.consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}) - "
                f"cooldown for {_AUTH_COOLDOWN_S:.0f}s"
            )
        return self._cool(key, cooldown, reason="authentication", detail=detail)

    def _cool(
        self, key: _PooledKey, seconds: float, *, reason: str, detail: str = ""
    ) -> float | None:
        """Hold a key out for ``seconds``, or skip when a longer cooldown stands.

        Returns the deadline applied, or ``None`` when this call extended
        nothing because a concurrent request's cooldown already held the key
        out for longer.
        """
        now = time.monotonic()
        until = now + seconds
        if until <= key.cooling_until:
            return None
        key.cooling_until = until
        key.cooling_reason = reason
        now = time.monotonic()
        usable = sum(1 for k in self._keys if not k.cooling(now))
        logger.warning(
            "{} key pool: {} ({} of {} keys usable)",
            self._provider_name,
            detail or f"key #{key.index} cooling for {seconds:.0f}s ({reason})",
            usable,
            len(self._keys),
        )
        trace_event(
            stage="provider",
            event="provider.key_pool.key_cooling",
            source="provider",
            provider=self._provider_name,
            key_index=key.index,
            reason=reason,
            cooldown_s=round(seconds, 3),
            usable_keys=usable,
            pool_size=len(self._keys),
        )
        return until

    def _stated_reset(self, error: BaseException) -> float | None:
        """Return the cooldown the provider itself stated, when usable."""
        stated = retry_after_seconds(error)
        if stated is None:
            stated = _rate_limit_reset_seconds(error)
        if stated is not None and math.isfinite(stated) and stated > 0.0:
            return stated
        return None

    def _roll_usage_window(self, key: _PooledKey, now: float) -> None:
        """Roll a key's usage counter over once its window has elapsed."""
        if not key.usage_window_seconds or now < key.window_reset_at:
            return
        if key.usage_count or key.exhausted:
            logger.info(
                "{} key pool: key #{} usage window elapsed, resetting "
                "({} uses last window).",
                self._provider_name,
                key.index,
                key.usage_count,
            )
        key.usage_count = 0
        key.exhausted = False
        key.window_reset_at = now + key.usage_window_seconds

    def _restore(self, health: Mapping[int, _HealthRollback]) -> None:
        """Undo sidelining from refusals that proved not to be key-local.

        The cooldown and its reason are restored only while each still holds
        the value this refusal wrote, so a cooldown a concurrent request
        established afterwards - one the provider did ask for - is left
        standing, classified by the reason that request earned.
        """
        for index, rollback in health.items():
            key = self._keys[index]
            if key.cooling_until == rollback.applied_cooling_until:
                key.cooling_until = rollback.previous_cooling_until
                key.cooling_reason = rollback.previous_cooling_reason


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
    if wait < 90:
        return f"{wait:.0f} seconds"
    if wait >= 3600:
        return f"{wait / 3600:.1f} hours"
    return f"{wait / 60:.0f} minutes"

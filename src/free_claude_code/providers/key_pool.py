"""API Key Pool and LRU rotation logic for providers.

A pool turns N interchangeable API keys into one self-healing virtual key.
Selection is least-recently-used (LRU): every request takes the key that has
been idle the longest, so load spreads evenly across all configured keys.
"""

import asyncio
import math
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import openai
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.providers.failure_policy import retry_after_seconds

T = TypeVar("T")

KeyClientFactory = Callable[[str], AsyncOpenAI]
HedgePermitFactory = Callable[[], Awaitable[Any]]

# A key must fail this many times consecutively before it enters a hard cooldown.
# A single 401/403 hiccup will not kill the key permanently — it gets a temporary
# cooldown instead and re-enters the rotation after _FAIL_COOLDOWN_S.
_MAX_CONSECUTIVE_FAILURES = 3
_FAIL_COOLDOWN_S = 300.0  # 5 min between early failure retries
_HARD_COOLDOWN_S = 1200.0  # 20 min after hitting _MAX_CONSECUTIVE_FAILURES
_RATE_LIMIT_COOLDOWN_S = 60.0

_EPOCH_MILLISECONDS_FLOOR = 1e11
_EPOCH_SECONDS_FLOOR = 1e9


@dataclass(frozen=True, slots=True)
class KeyPoolStatus:
    """A point-in-time summary of pooled key health for the Admin UI."""

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


class ApiKeyInfo:
    """Tracks state and cooldowns for a single API key."""

    def __init__(
        self,
        key: str,
        usage_limit: int = 0,
        initial_offset: float = 0.0,
        usage_window_seconds: float | None = None,
    ):
        self.key = key
        self.usage_limit = usage_limit
        self.usage_window_seconds = usage_window_seconds
        self.usage_count = 0
        self.lock = threading.Lock()
        self.exhausted = False
        self.failed = False
        self.consecutive_failures = 0
        self.rate_limited_until = 0.0
        self._usage_window_reset_at = (
            time.monotonic() + usage_window_seconds if usage_window_seconds else 0.0
        )
        self.last_used_at: float = -initial_offset

    def _key_suffix(self) -> str:
        return f"...{self.key[-8:]}" if len(self.key) >= 8 else "..."

    def _maybe_reset_usage_window_locked(self) -> None:
        """Roll the usage counter over once its window has elapsed."""
        if (
            self.usage_window_seconds
            and time.monotonic() >= self._usage_window_reset_at
        ):
            if self.usage_count or self.exhausted:
                logger.info(
                    "key_pool: key {} usage window elapsed, resetting ({} uses last window).",
                    self._key_suffix(),
                    self.usage_count,
                )
            self.usage_count = 0
            self.exhausted = False
            self._usage_window_reset_at = time.monotonic() + self.usage_window_seconds

    def increment(self) -> None:
        with self.lock:
            self._maybe_reset_usage_window_locked()
            self.usage_count += 1
            if self.usage_limit > 0 and self.usage_count >= self.usage_limit:
                self.exhausted = True
                logger.warning(
                    "key_pool: key {} reached usage limit ({}/{}). Rotating to next key.",
                    self._key_suffix(),
                    self.usage_count,
                    self.usage_limit,
                )

    def mark_failed(self) -> None:
        """Record an authentication or permission failure."""
        with self.lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                self.rate_limited_until = max(
                    self.rate_limited_until, time.monotonic() + _HARD_COOLDOWN_S
                )
                logger.warning(
                    "key_pool: key {} hit {} consecutive failures - hard cooldown for {}min before retry.",
                    self._key_suffix(),
                    self.consecutive_failures,
                    int(_HARD_COOLDOWN_S // 60),
                )
            else:
                self.rate_limited_until = max(
                    self.rate_limited_until, time.monotonic() + _FAIL_COOLDOWN_S
                )
                logger.warning(
                    "key_pool: key {} failure {}/{} - cooling down for {}s before retry.",
                    self._key_suffix(),
                    self.consecutive_failures,
                    _MAX_CONSECUTIVE_FAILURES,
                    int(_FAIL_COOLDOWN_S),
                )

    def mark_rate_limited(self, cooldown_seconds: float = 60.0) -> None:
        with self.lock:
            self.rate_limited_until = max(
                self.rate_limited_until, time.monotonic() + cooldown_seconds
            )
            logger.warning(
                "key_pool: key {} is rate limited for {}s. Rotating to next key.",
                self._key_suffix(),
                int(cooldown_seconds),
            )

    def reset_consecutive_failures(self) -> None:
        with self.lock:
            if self.consecutive_failures > 0:
                self.consecutive_failures = 0

    def mark_used(self) -> None:
        with self.lock:
            self.last_used_at = time.monotonic()

    def is_available(self) -> bool:
        with self.lock:
            self._maybe_reset_usage_window_locked()
            if self.exhausted:
                return False
            return time.monotonic() >= self.rate_limited_until

    def lru_score(self) -> float:
        with self.lock:
            return self.last_used_at

    def available_in(self, now: float) -> float:
        with self.lock:
            self._maybe_reset_usage_window_locked()
            wait = max(self.rate_limited_until - now, 0.0)
            if self.exhausted:
                if self.usage_window_seconds:
                    wait = max(wait, max(self._usage_window_reset_at - now, 0.0))
                else:
                    return math.inf
            return wait


class KeyPool:
    """Manages an LRU pool of interchangeable API keys."""

    def __init__(
        self,
        keys: Sequence[str],
        *,
        provider_name: str = "",
        client_factory: KeyClientFactory | None = None,
        usage_limit: int = 0,
        usage_window_seconds: float | None = None,
        key_rate_limit: int = 0,
        key_rate_window: float = 60.0,
        hedge_delay_seconds: float = 0.0,
        hedge_permit_factory: HedgePermitFactory | None = None,
        rotate_on_permission_denied: bool = True,
    ) -> None:
        if not keys:
            raise ValueError("A key pool requires at least one API key")
        self._provider_name = provider_name
        self._client_factory = client_factory
        self._hedge_delay_seconds = max(0.0, hedge_delay_seconds)
        self._hedge_permit_factory = hedge_permit_factory
        self._rotate_on_permission_denied = rotate_on_permission_denied
        self.keys = [
            ApiKeyInfo(
                key,
                usage_limit,
                initial_offset=float(len(keys) - i),
                usage_window_seconds=usage_window_seconds,
            )
            for i, key in enumerate(keys)
        ]
        self._key_index: dict[str, ApiKeyInfo] = {ki.key: ki for ki in self.keys}
        self._key_limiters: dict[str, StrictSlidingWindowLimiter] = (
            {
                ki.key: StrictSlidingWindowLimiter(key_rate_limit, key_rate_window)
                for ki in self.keys
            }
            if key_rate_limit > 0
            else {}
        )
        self.lock = threading.Lock()

        logger.info(
            "Key pool initialized for {} ({} keys{})",
            provider_name or "provider",
            len(self.keys),
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
        return len(self.keys)

    def status(self) -> KeyPoolStatus:
        now = time.monotonic()
        ready = 0
        cooling = 0
        retired = 0
        soonest: list[float] = []

        with self.lock:
            for ki in self.keys:
                if ki.is_available():
                    ready += 1
                else:
                    is_permanently_exhausted = (
                        ki.exhausted and not ki.usage_window_seconds
                    )
                    if (
                        ki.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES
                        or is_permanently_exhausted
                    ):
                        retired += 1
                    else:
                        cooling += 1
                    wait = ki.available_in(now)
                    if math.isfinite(wait):
                        soonest.append(wait)

        return KeyPoolStatus(
            size=len(self.keys),
            ready=ready,
            cooling=cooling,
            retired=retired,
            soonest_ready_in=min(soonest) if soonest else None,
        )

    def get_next_key(self) -> str | None:
        """Return the available key idle the longest (LRU), or None."""
        with self.lock:
            available = [ki for ki in self.keys if ki.is_available()]
            if not available:
                return None
            best = min(available, key=lambda ki: ki.lru_score())
            best.mark_used()
            return best.key

    def usable_key_count(self) -> int:
        """Return keys that may accept an upstream request right now."""
        with self.lock:
            return sum(key.is_available() for key in self.keys)

    def _get_next_key_with_immediate_rate_slot(self) -> str | None:
        """Claim the LRU key that has an immediately available RPM slot.

        A pool must not wait behind an older, rate-saturated key while another
        healthy credential can send now.  Claiming the slot here keeps the
        selection and limiter admission together without an await boundary.
        """
        with self.lock:
            candidates = [
                key
                for key in self.keys
                if key.is_available()
                and (
                    (limiter := self._key_limiters.get(key.key)) is None
                    or limiter.headroom() > 0
                )
            ]
            if not candidates:
                return None
            best = min(candidates, key=lambda key: key.lru_score())
            limiter = self._key_limiters.get(best.key)
            if limiter is not None and not limiter.try_acquire():
                return None
            best.mark_used()
            return best.key

    def mark_key_used(self, key: str, *, proves_credential: bool = True) -> None:
        key_info = self._key_index.get(key)
        if key_info is not None and proves_credential:
            key_info.increment()
            key_info.reset_consecutive_failures()

    def mark_key_failed(self, key: str) -> None:
        key_info = self._key_index.get(key)
        if key_info is not None:
            key_info.mark_failed()

    def mark_key_rate_limited(self, key: str, cooldown_seconds: float = 60.0) -> None:
        key_info = self._key_index.get(key)
        if key_info is not None:
            key_info.mark_rate_limited(cooldown_seconds)

    async def _acquire_key_rate_slot(self, key: str) -> bool:
        """Wait for this key's own RPM window without consuming a stale slot."""
        limiter = self._key_limiters.get(key)
        if limiter is None:
            return True
        key_info = self._key_index.get(key)
        if key_info is None:
            return False
        return await limiter.acquire_if(key_info.is_available)

    def _handle_key_error(self, key: str, error: Exception) -> bool:
        if isinstance(error, openai.AuthenticationError):
            logger.warning(
                "{} key rotation: AuthenticationError for key ...{}",
                self._provider_name,
                key[-8:] if len(key) >= 8 else "...",
            )
            self.mark_key_failed(key)
            return True
        if isinstance(error, openai.PermissionDeniedError):
            if not self._rotate_on_permission_denied:
                return False
            logger.warning(
                "{} key rotation: PermissionDeniedError for key ...{}",
                self._provider_name,
                key[-8:] if len(key) >= 8 else "...",
            )
            self.mark_key_failed(key)
            return True
        if isinstance(error, openai.RateLimitError):
            cooldown = (
                _rate_limit_reset_seconds(error)
                or retry_after_seconds(error)
                or _RATE_LIMIT_COOLDOWN_S
            )
            logger.warning(
                "{} key rotation: RateLimitError for key ...{}, cooling for {}s",
                self._provider_name,
                key[-8:] if len(key) >= 8 else "...",
                int(cooldown),
            )
            self.mark_key_rate_limited(key, cooldown)
            return True
        return False

    async def run_key_local(
        self,
        operation: Callable[[AsyncOpenAI], Awaitable[T]],
        *,
        proves_credential: bool = True,
    ) -> T:
        """Execute an operation using keys from the pool, rotating on key failures."""
        if not self._client_factory:
            raise ValueError(
                "KeyPool requires a client_factory to execute run_key_local"
            )

        if self._hedge_delay_seconds <= 0.0 or len(self.keys) < 2:
            return await self._run_key_sequential(
                operation, proves_credential=proves_credential
            )
        return await self._run_key_hedged(
            operation, proves_credential=proves_credential
        )

    async def _run_key_sequential(
        self,
        operation: Callable[[AsyncOpenAI], Awaitable[T]],
        *,
        proves_credential: bool = True,
    ) -> T:
        assert self._client_factory is not None
        last_error: Exception | None = None
        for _ in range(len(self.keys)):
            current_key = self._get_next_key_with_immediate_rate_slot()
            rate_slot_claimed = current_key is not None
            if current_key is None:
                current_key = self.get_next_key()
            if not current_key:
                break

            if not rate_slot_claimed and not await self._acquire_key_rate_slot(
                current_key
            ):
                continue

            client = self._client_factory(current_key)
            try:
                result = await operation(client)
                self.mark_key_used(current_key, proves_credential=proves_credential)
                return result
            except Exception as error:
                if self._handle_key_error(current_key, error):
                    last_error = error
                    continue
                raise

        if last_error is not None:
            raise last_error

        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message=f"Every {self._provider_name} API key in the pool is cooling or exhausted.",
            retryable=True,
        )

    async def _run_key_hedged(
        self,
        operation: Callable[[AsyncOpenAI], Awaitable[T]],
        *,
        proves_credential: bool = True,
    ) -> T:
        assert self._client_factory is not None

        async def execute_on_key(key: str) -> T:
            assert self._client_factory is not None
            admitted = await self._acquire_key_rate_slot(key)
            if not admitted:
                raise ExecutionFailure(
                    kind=FailureKind.RATE_LIMIT,
                    status_code=429,
                    message=f"{self._provider_name} API key is cooling down.",
                    retryable=True,
                )
            client = self._client_factory(key)
            result = await operation(client)
            self.mark_key_used(key, proves_credential=proves_credential)
            return result

        key1 = self._get_next_key_with_immediate_rate_slot()
        key1_rate_slot_claimed = key1 is not None
        if key1 is None:
            key1 = self.get_next_key()
        if not key1:
            return await self._run_key_sequential(
                operation, proves_credential=proves_credential
            )

        async def execute_claimed_key(key: str, rate_slot_claimed: bool) -> T:
            if not rate_slot_claimed:
                return await execute_on_key(key)
            assert self._client_factory is not None
            result = await operation(self._client_factory(key))
            self.mark_key_used(key, proves_credential=proves_credential)
            return result

        task1: asyncio.Task[T] = asyncio.create_task(
            execute_claimed_key(key1, key1_rate_slot_claimed)
        )
        tasks: dict[asyncio.Task[T], str] = {task1: key1}
        hedge_permit: Any | None = None
        try:
            done, _ = await asyncio.wait([task1], timeout=self._hedge_delay_seconds)
            if task1 in done:
                tasks.pop(task1)
                exc = task1.exception()
                if exc is None:
                    return task1.result()
                if isinstance(exc, ExecutionFailure):
                    return await self._run_key_sequential(
                        operation, proves_credential=proves_credential
                    )
                if isinstance(exc, Exception) and self._handle_key_error(key1, exc):
                    return await self._run_key_sequential(
                        operation, proves_credential=proves_credential
                    )
                if isinstance(exc, BaseException):
                    raise exc

            if self._hedge_permit_factory is not None:
                hedge_permit = await self._hedge_permit_factory()
                if hedge_permit is None:
                    return await task1
            key2 = self._get_next_key_with_immediate_rate_slot()
            key2_rate_slot_claimed = key2 is not None
            if key2 is None:
                key2 = self.get_next_key()
            if not key2 or key2 == key1:
                return await task1

            logger.info(
                "{} key hedging: key ...{} quiet after {}s, racing with key ...{}",
                self._provider_name,
                key1[-6:] if len(key1) >= 6 else "...",
                self._hedge_delay_seconds,
                key2[-6:] if len(key2) >= 6 else "...",
            )
            task2: asyncio.Task[T] = asyncio.create_task(
                execute_claimed_key(key2, key2_rate_slot_claimed)
            )
            tasks[task2] = key2

            while tasks:
                done_tasks, _ = await asyncio.wait(
                    tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done_tasks:
                    k = tasks.pop(finished)
                    exc = finished.exception()
                    if exc is None:
                        return finished.result()

                    if isinstance(exc, ExecutionFailure):
                        continue
                    if isinstance(exc, Exception) and self._handle_key_error(k, exc):
                        continue
                    if isinstance(exc, BaseException):
                        raise exc

            return await self._run_key_sequential(
                operation, proves_credential=proves_credential
            )
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if hedge_permit is not None:
                await hedge_permit.aclose()

    async def aclose(self) -> None:
        """Release any resources held by the pool."""
        pass


def _rate_limit_reset_seconds(error: BaseException) -> float | None:
    """Read X-RateLimit-Reset as a delay, tolerating delta and epoch timestamps."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("x-ratelimit-reset")
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw_str = raw.strip().lower()
    try:
        if raw_str.endswith("ms"):
            value = float(raw_str[:-2].strip()) / 1000.0
        elif raw_str.endswith("s"):
            value = float(raw_str[:-1].strip())
        else:
            value = float(raw_str)
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

"""API Key Pool and LRU rotation logic for providers.

A pool turns N interchangeable API keys into one self-healing virtual key.
Selection is least-recently-used (LRU): every request takes the key that has
been idle the longest, so load spreads evenly across all configured keys.
"""

import math
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import openai
from loguru import logger
from openai import AsyncOpenAI

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.failure_policy import retry_after_seconds

T = TypeVar("T")

KeyClientFactory = Callable[[str], AsyncOpenAI]

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
                self.rate_limited_until = time.monotonic() + _HARD_COOLDOWN_S
                logger.warning(
                    "key_pool: key {} hit {} consecutive failures - hard cooldown for {}min before retry.",
                    self._key_suffix(),
                    self.consecutive_failures,
                    int(_HARD_COOLDOWN_S // 60),
                )
            else:
                self.rate_limited_until = time.monotonic() + _FAIL_COOLDOWN_S
                logger.warning(
                    "key_pool: key {} failure {}/{} - cooling down for {}s before retry.",
                    self._key_suffix(),
                    self.consecutive_failures,
                    _MAX_CONSECUTIVE_FAILURES,
                    int(_FAIL_COOLDOWN_S),
                )

    def mark_rate_limited(self, cooldown_seconds: float = 60.0) -> None:
        with self.lock:
            self.rate_limited_until = time.monotonic() + cooldown_seconds
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
    ) -> None:
        if not keys:
            raise ValueError("A key pool requires at least one API key")
        self._provider_name = provider_name
        self._client_factory = client_factory
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
                    if ki.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
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

        last_error: Exception | None = None
        for _ in range(len(self.keys)):
            current_key = self.get_next_key()
            if not current_key:
                break

            client = self._client_factory(current_key)
            try:
                result = await operation(client)
                self.mark_key_used(current_key, proves_credential=proves_credential)
                return result
            except openai.AuthenticationError as error:
                logger.warning(
                    "{} key rotation: AuthenticationError for key ...{}",
                    self._provider_name,
                    current_key[-8:] if len(current_key) >= 8 else "...",
                )
                self.mark_key_failed(current_key)
                last_error = error
            except openai.PermissionDeniedError as error:
                logger.warning(
                    "{} key rotation: PermissionDeniedError for key ...{}",
                    self._provider_name,
                    current_key[-8:] if len(current_key) >= 8 else "...",
                )
                self.mark_key_failed(current_key)
                last_error = error
            except openai.RateLimitError as error:
                cooldown = (
                    _rate_limit_reset_seconds(error)
                    or retry_after_seconds(error)
                    or _RATE_LIMIT_COOLDOWN_S
                )
                logger.warning(
                    "{} key rotation: RateLimitError for key ...{}, cooling for {}s",
                    self._provider_name,
                    current_key[-8:] if len(current_key) >= 8 else "...",
                    int(cooldown),
                )
                self.mark_key_rate_limited(current_key, cooldown)
                last_error = error
            except Exception:
                raise

        if last_error is not None:
            raise last_error

        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message=f"Every {self._provider_name} API key in the pool is cooling or exhausted.",
            retryable=True,
        )

    async def aclose(self) -> None:
        """Release any resources held by the pool."""
        pass


ApiKeyPool = KeyPool


def _rate_limit_reset_seconds(error: BaseException) -> float | None:
    """Read X-RateLimit-Reset as a delay, tolerating delta and epoch timestamps."""
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

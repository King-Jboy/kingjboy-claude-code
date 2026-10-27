"""Credential rotation for the NIM and OpenRouter OpenAI-compatible paths."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter

_MAX_CONSECUTIVE_FAILURES = 3
_FAILURE_COOLDOWN_SECONDS = 300.0
_RATE_LIMIT_COOLDOWN_SECONDS = 60.0


@dataclass(slots=True)
class _KeyState:
    key: str
    limiter: StrictSlidingWindowLimiter
    last_used_at: float
    consecutive_failures: int = 0
    failed: bool = False
    unavailable_until: float = 0.0


class ApiKeyPool:
    """Select eligible credentials in LRU order with an independent RPM gate."""

    def __init__(
        self,
        keys: tuple[str, ...],
        *,
        rate_limit: int,
        rate_window: float,
        on_capacity_change: Callable[[int], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._rate_limit = rate_limit
        self._on_capacity_change = on_capacity_change
        unique_keys = tuple(dict.fromkeys(key for key in keys if key))
        self._keys = tuple(
            _KeyState(
                key=key,
                limiter=StrictSlidingWindowLimiter(rate_limit, rate_window),
                last_used_at=-float(len(unique_keys) - index),
            )
            for index, key in enumerate(unique_keys)
        )
        self._by_key = {state.key: state for state in self._keys}

    def active_key_count(self, now: float | None = None) -> int:
        """Return how many credentials are currently eligible for traffic."""
        with self._lock:
            check_now = time.monotonic() if now is None else now
            return sum(
                1
                for state in self._keys
                if not state.failed and check_now >= state.unavailable_until
            )

    def active_rate_limit(self, now: float | None = None) -> int:
        """Return the aggregate RPM capacity of currently eligible keys."""
        return max(1, self.active_key_count(now)) * self._rate_limit

    def _notify_capacity(self, now: float | None = None) -> None:
        if self._on_capacity_change is not None:
            self._on_capacity_change(self.active_rate_limit(now))

    def has_eligible_key(self, *, exclude: str | None = None) -> bool:
        """Return whether a key (optionally distinct from ``exclude``) has immediate capacity."""
        with self._lock:
            now = time.monotonic()
            return any(
                state.key != exclude
                and not state.failed
                and now >= state.unavailable_until
                and state.limiter.headroom() > 0
                for state in self._keys
            )

    def next_available_in(self, *, exclude: str | None = None) -> float | None:
        """Return the minimum seconds until an eligible key has RPM capacity."""
        with self._lock:
            now = time.monotonic()
            waits = [
                state.limiter.next_available_in()
                for state in self._keys
                if state.key != exclude
                and not state.failed
                and now >= state.unavailable_until
            ]
            return min(waits) if waits else None

    def get_next_key(self, *, exclude: str | None = None) -> str | None:
        """Reserve the longest-idle key that currently has RPM capacity."""
        with self._lock:
            now = time.monotonic()
            self._notify_capacity(now)
            eligible = sorted(
                (
                    state
                    for state in self._keys
                    if state.key != exclude
                    and not state.failed
                    and now >= state.unavailable_until
                ),
                key=lambda state: state.last_used_at,
            )
            for state in eligible:
                if state.limiter.try_acquire():
                    state.last_used_at = now
                    return state.key
        return None

    def mark_succeeded(self, key: str) -> None:
        """Clear a transient authentication-failure streak after stream open."""
        with self._lock:
            if state := self._by_key.get(key):
                now = time.monotonic()
                state.consecutive_failures = 0
                if state.unavailable_until and now >= state.unavailable_until:
                    state.unavailable_until = 0.0
                self._notify_capacity(now)

    def mark_failed(self, key: str) -> None:
        """Temporarily hold failed keys, then retire only persistent failures."""
        with self._lock:
            state = self._by_key.get(key)
            if state is None:
                return
            state.consecutive_failures += 1
            if state.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                state.failed = True
                self._notify_capacity()
                logger.warning(
                    "key_pool: permanently disabled a credential after {} failures.",
                    state.consecutive_failures,
                )
                return
            now = time.monotonic()
            state.unavailable_until = now + _FAILURE_COOLDOWN_SECONDS
            self._notify_capacity(now)
            logger.warning(
                "key_pool: credential failure {}/{}; holding it for {} seconds.",
                state.consecutive_failures,
                _MAX_CONSECUTIVE_FAILURES,
                int(_FAILURE_COOLDOWN_SECONDS),
            )

    def mark_rate_limited(self, key: str) -> None:
        """Hold one credential after its upstream explicitly reports a 429."""
        with self._lock:
            state = self._by_key.get(key)
            if state is None:
                return
            now = time.monotonic()
            state.unavailable_until = max(
                state.unavailable_until,
                now + _RATE_LIMIT_COOLDOWN_SECONDS,
            )
            self._notify_capacity(now)
            logger.warning(
                "key_pool: credential is rate limited; holding it for {} seconds.",
                int(_RATE_LIMIT_COOLDOWN_SECONDS),
            )

"""Credential rotation for the NIM and OpenRouter OpenAI-compatible paths."""

import threading
import time
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
    ) -> None:
        self._lock = threading.RLock()
        self._keys = tuple(
            _KeyState(
                key=key,
                limiter=StrictSlidingWindowLimiter(rate_limit, rate_window),
                last_used_at=-float(len(keys) - index),
            )
            for index, key in enumerate(keys)
        )
        self._by_key = {state.key: state for state in self._keys}

    def get_next_key(self) -> str | None:
        """Reserve the longest-idle key that currently has RPM capacity."""
        with self._lock:
            now = time.monotonic()
            eligible = sorted(
                (
                    state
                    for state in self._keys
                    if not state.failed and now >= state.unavailable_until
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
                state.consecutive_failures = 0

    def mark_failed(self, key: str) -> None:
        """Temporarily hold failed keys, then retire only persistent failures."""
        with self._lock:
            state = self._by_key.get(key)
            if state is None:
                return
            state.consecutive_failures += 1
            if state.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                state.failed = True
                logger.warning(
                    "key_pool: permanently disabled a credential after {} failures.",
                    state.consecutive_failures,
                )
                return
            state.unavailable_until = time.monotonic() + _FAILURE_COOLDOWN_SECONDS
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
            state.unavailable_until = max(
                state.unavailable_until,
                time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS,
            )
            logger.warning(
                "key_pool: credential is rate limited; holding it for {} seconds.",
                int(_RATE_LIMIT_COOLDOWN_SECONDS),
            )

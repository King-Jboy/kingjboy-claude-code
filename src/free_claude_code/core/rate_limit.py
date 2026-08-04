"""Shared strict sliding-window rate limiting primitives."""

import asyncio
import time
from collections import deque
from collections.abc import Callable


class StrictSlidingWindowLimiter:
    """Strict sliding window limiter.

    Guarantees: at most ``rate_limit`` acquisitions in any interval of length
    ``rate_window`` (seconds).

    Implemented as an async context manager so call sites can do::

        async with limiter:
            ...
    """

    def __init__(self, rate_limit: int, rate_window: float) -> None:
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")

        self._rate_limit = int(rate_limit)
        self._rate_window = float(rate_window)
        self._times: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._acquire(None)

    async def acquire_if(self, allowed: Callable[[], bool]) -> bool:
        """Record an acquisition only if ``allowed`` still holds at admission.

        Capacity is awaited first. The synchronous condition and timestamp write
        then run without yielding, so a rejected admission consumes no quota.
        """
        return await self._acquire(allowed)

    # ``headroom`` and ``next_available_in`` are synchronous reads used to rank
    # limiters against each other. They never await, so their eviction pass
    # cannot interleave with a locked acquisition on the same event loop.
    def headroom(self) -> int:
        """Return how many acquisitions the current window still admits."""
        self._evict_expired(time.monotonic())
        return max(0, self._rate_limit - len(self._times))

    def try_acquire(self) -> bool:
        """Record an acquisition only when the window admits one right now."""
        now = time.monotonic()
        self._evict_expired(now)
        if len(self._times) >= self._rate_limit:
            return False
        self._times.append(now)
        return True

    def next_available_in(self) -> float:
        """Return seconds until one acquisition frees up (0.0 when admissible now)."""
        now = time.monotonic()
        self._evict_expired(now)
        if len(self._times) < self._rate_limit:
            return 0.0
        return max(0.0, (self._times[0] + self._rate_window) - now)

    def _evict_expired(self, now: float) -> None:
        """Drop timestamps that have left the trailing window."""
        cutoff = now - self._rate_window
        while self._times and self._times[0] <= cutoff:
            self._times.popleft()

    async def _acquire(self, allowed: Callable[[], bool] | None) -> bool:
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.monotonic()
                self._evict_expired(now)

                if len(self._times) < self._rate_limit:
                    if allowed is not None and not allowed():
                        return False
                    self._times.append(time.monotonic())
                    return True

                oldest = self._times[0]
                wait_time = max(0.0, (oldest + self._rate_window) - now)

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    async def __aenter__(self) -> StrictSlidingWindowLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

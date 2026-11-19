"""Shared strict sliding-window rate limiting primitives."""

import asyncio
import contextlib
import threading
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
        self._state_lock = threading.Lock()
        self._capacity_waiters: list[
            tuple[asyncio.AbstractEventLoop, asyncio.Event]
        ] = []

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
    def set_rate_limit(self, rate_limit: int) -> None:
        """Retune the window's capacity in place.

        Capacity is not always known once and for all: a limiter standing in
        front of a pool of credentials admits the pool's total, and that total
        falls as keys are held out and rises as they return. Already-recorded
        acquisitions are kept, so lowering capacity throttles from now on rather
        than pretending past requests did not happen.
        """
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        new_limit = int(rate_limit)
        waiters_to_wake: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        with self._state_lock:
            increased = new_limit > self._rate_limit
            self._rate_limit = new_limit
            if increased and self._capacity_waiters:
                waiters_to_wake = list(self._capacity_waiters)
        for loop, waiter in waiters_to_wake:
            if not loop.is_closed():
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(waiter.set)

    def recorded_times(self) -> tuple[float, ...]:
        """Return the acquisitions still inside the window, oldest first."""
        self._evict_expired(time.monotonic())
        return tuple(self._times)

    def record_times(self, times: tuple[float, ...]) -> None:
        """Replace recorded acquisitions, e.g. those of a predecessor limiter."""
        self._times = deque(sorted(times))
        self._evict_expired(time.monotonic())

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

    def refund_latest(self) -> None:
        """Drop the newest acquisition.

        Only call this before the next ``await``, so the newest timestamp is
        still the one just recorded.
        """
        if self._times:
            self._times.pop()

    def next_available_in(self) -> float:
        """Return seconds until one acquisition frees up (0.0 when admissible now)."""
        now = time.monotonic()
        self._evict_expired(now)
        limit = self._rate_limit
        if len(self._times) < limit:
            return 0.0
        unlock_index = len(self._times) - limit
        return max(0.0, (self._times[unlock_index] + self._rate_window) - now)

    def _evict_expired(self, now: float) -> None:
        """Drop timestamps that have left the trailing window."""
        cutoff = now - self._rate_window
        while self._times and self._times[0] <= cutoff:
            self._times.popleft()

    async def _acquire(self, allowed: Callable[[], bool] | None) -> bool:
        while True:
            wait_time = 0.0
            waiter_entry: tuple[asyncio.AbstractEventLoop, asyncio.Event] | None = None
            async with self._lock:
                now = time.monotonic()
                self._evict_expired(now)
                limit = self._rate_limit

                if len(self._times) < limit:
                    if allowed is not None and not allowed():
                        return False
                    self._times.append(time.monotonic())
                    return True

                unlock_index = len(self._times) - limit
                oldest_blocking = self._times[unlock_index]
                wait_time = max(0.0, (oldest_blocking + self._rate_window) - now)
                if wait_time > 0:
                    waiter_entry = (asyncio.get_running_loop(), asyncio.Event())
                    with self._state_lock:
                        self._capacity_waiters.append(waiter_entry)

            if wait_time > 0 and waiter_entry is not None:
                sleep_task = asyncio.create_task(asyncio.sleep(wait_time))
                wake_task = asyncio.create_task(waiter_entry[1].wait())
                try:
                    done, pending = await asyncio.wait(
                        (sleep_task, wake_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        if not task.cancelled():
                            task.result()
                finally:
                    for task in (sleep_task, wake_task):
                        if not task.done():
                            task.cancel()
                    with (
                        self._state_lock,
                        contextlib.suppress(ValueError),
                    ):
                        self._capacity_waiters.remove(waiter_entry)
            else:
                await asyncio.sleep(0)

    async def __aenter__(self) -> StrictSlidingWindowLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

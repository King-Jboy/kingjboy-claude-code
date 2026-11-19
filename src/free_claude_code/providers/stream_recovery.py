"""Provider-owned stream holdback and recovery decisions."""

import asyncio
import ssl
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import cast

import httpx
import httpx2
import openai

from free_claude_code.core.async_tasks import wait_owned
from free_claude_code.core.failures import ExecutionFailure

from .failure_policy import RetryableProviderProtocolError, retryable_transient_status

EARLY_HOLDBACK_SECONDS = 0.02
RECOVERY_BUFFER_MAX_BYTES = 65_536


class TruncatedProviderStreamError(RetryableProviderProtocolError):
    """An upstream stream ended without its required terminal marker."""


class RecoveryFailureAction(StrEnum):
    """How one provider stream should respond to an upstream failure."""

    EARLY_RETRY = "early_retry"
    MIDSTREAM_RECOVERY = "midstream_recovery"
    FINAL_ERROR = "final_error"


class HoldbackSignal(Enum):
    """Wake a transport to release its buffer without cancelling its read."""

    EXPIRED = auto()


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Failure decision for one provider stream attempt."""

    action: RecoveryFailureAction
    retryable: bool
    committed: bool
    has_buffered: bool


class RecoveryHoldbackBuffer:
    """Briefly retain SSE so early cutoffs can be retried invisibly."""

    def __init__(
        self,
        *,
        holdback_seconds: float = EARLY_HOLDBACK_SECONDS,
        max_bytes: int = RECOVERY_BUFFER_MAX_BYTES,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._holdback_seconds = holdback_seconds
        self._max_bytes = max_bytes
        self._now = now or time.monotonic
        self._events: list[str] = []
        self._bytes = 0
        self._started_at: float | None = None
        self.committed = False

    def push(self, event: str) -> list[str]:
        if self.committed:
            return [event]
        if self._started_at is None:
            self._started_at = self._now()
        self._events.append(event)
        self._bytes += (
            len(event)
            if event.isascii()
            else len(event.encode("utf-8", errors="replace"))
        )
        if (
            self._bytes >= self._max_bytes
            or self._now() - self._started_at >= self._holdback_seconds
        ):
            return self.flush()
        return []

    def flush(self) -> list[str]:
        if self.committed:
            return []
        self.committed = True
        events = self._events
        self._events = []
        self._bytes = 0
        self._started_at = None
        return events

    def discard(self) -> None:
        self._events = []
        self._bytes = 0
        self._started_at = None

    @property
    def has_buffered(self) -> bool:
        return bool(self._events)

    @property
    def remaining_delay(self) -> float | None:
        if self.committed or self._started_at is None:
            return None
        return max(0.0, self._holdback_seconds - (self._now() - self._started_at))


class _HoldbackReader[EventT](AsyncIterator[EventT | HoldbackSignal]):
    """Borrow a source and retain only the read interrupted by a buffer deadline."""

    def __init__(
        self, source: AsyncIterator[EventT], remaining_delay: Callable[[], float | None]
    ) -> None:
        self._source = source
        self._remaining_delay = remaining_delay
        self._pending: asyncio.Task[EventT] | None = None
        self._deadline_checked = False

    def __aiter__(self) -> _HoldbackReader[EventT]:
        return self

    async def __anext__(self) -> EventT | HoldbackSignal:
        pending = self._pending
        delay = self._remaining_delay()
        if pending is not None and pending.done():
            self._pending = None
            self._deadline_checked = delay == 0
            return pending.result()
        # Check one ready event at expiry, then release even if it was metadata.
        if delay == 0 and self._deadline_checked:
            return HoldbackSignal.EXPIRED
        if pending is None:
            if delay is None:
                return await anext(self._source)
            pending = self._pending = asyncio.create_task(self._read())
        await asyncio.wait({pending}, timeout=delay)
        if pending.done():
            self._pending = None
            self._deadline_checked = self._remaining_delay() == 0
            return pending.result()
        return HoldbackSignal.EXPIRED

    async def _read(self) -> EventT:
        return await anext(self._source)

    async def aclose(self) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        if not pending.done():
            pending.cancel()

        async def drain() -> None:
            await asyncio.gather(pending, return_exceptions=True)

        await wait_owned(asyncio.create_task(drain()))


class RecoveryController:
    """Own commit-boundary holdback for one provider stream lifecycle."""

    def __init__(self) -> None:
        self._holdback = RecoveryHoldbackBuffer()

    @property
    def committed(self) -> bool:
        return self._holdback.committed

    @property
    def has_buffered(self) -> bool:
        return self._holdback.has_buffered

    @property
    def remaining_delay(self) -> float | None:
        return self._holdback.remaining_delay

    @asynccontextmanager
    async def read_stream[EventT](
        self, source: AsyncIterator[EventT]
    ) -> AsyncGenerator[_HoldbackReader[EventT]]:
        reader = _HoldbackReader(source, lambda: self.remaining_delay)
        try:
            yield reader
        finally:
            await reader.aclose()

    def push(self, event: str) -> list[str]:
        return self._holdback.push(event)

    def flush(self) -> list[str]:
        return self._holdback.flush()

    def discard(self) -> None:
        self._holdback.discard()

    def flush_uncommitted(self, decision: RecoveryDecision) -> list[str]:
        if not decision.committed and decision.has_buffered:
            return self.flush()
        return []

    def advance_failure(
        self,
        error: BaseException,
        *,
        stream_opened: bool,
        generated_output: bool,
        complete_tool_salvageable: bool,
        attempts_remaining: int,
        retryable_override: bool | None = None,
    ) -> RecoveryDecision:
        retryable = (
            is_retryable_stream_error(error)
            if retryable_override is None
            else retryable_override
        )
        committed = self._holdback.committed
        has_buffered = self._holdback.has_buffered
        retry_available = attempts_remaining > 0
        reserve_last_attempt_for_recovery = generated_output and attempts_remaining == 1

        # Keepalives can commit the response before any content exists; the
        # client has then seen only message_start and pings, so a replay is
        # still invisible to it.
        if (
            retryable
            and retry_available
            and stream_opened
            and (not committed or not generated_output)
            and not complete_tool_salvageable
            and not reserve_last_attempt_for_recovery
        ):
            self._holdback.discard()
            self._holdback = RecoveryHoldbackBuffer()
            return RecoveryDecision(
                action=RecoveryFailureAction.EARLY_RETRY,
                retryable=True,
                committed=committed,
                has_buffered=has_buffered,
            )

        if (
            retryable
            and generated_output
            and (retry_available or complete_tool_salvageable)
        ):
            return RecoveryDecision(
                action=RecoveryFailureAction.MIDSTREAM_RECOVERY,
                retryable=True,
                committed=committed,
                has_buffered=has_buffered,
            )

        return RecoveryDecision(
            action=RecoveryFailureAction.FINAL_ERROR,
            retryable=retryable,
            committed=committed,
            has_buffered=has_buffered,
        )


def is_retryable_stream_error(exc: BaseException) -> bool:
    """Return whether one stream failure qualifies for retry or recovery."""
    if isinstance(exc, RetryableProviderProtocolError):
        return True
    if isinstance(exc, ExecutionFailure):
        return exc.retryable
    if isinstance(
        exc,
        openai.AuthenticationError
        | openai.PermissionDeniedError
        | openai.BadRequestError,
    ):
        return False
    if retryable_transient_status(exc) is not None:
        return True
    return isinstance(
        exc,
        (
            TimeoutError,
            ssl.SSLError,
            httpx.ReadTimeout,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx2.ReadTimeout,
            httpx2.ReadError,
            httpx2.RemoteProtocolError,
            httpx2.ConnectError,
            httpx2.NetworkError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
    )


def _keepalive_wait_step(quiet: float, quiet_after: float, interval: float) -> float:
    if quiet >= quiet_after:
        return interval
    return max(0.0, quiet_after - quiet)


async def _discard_pending_task(task: asyncio.Task[object]) -> None:
    """Cancel and drain a task the caller stopped awaiting."""
    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def iter_with_keepalive[ItemT, SentinelT](
    source: AsyncIterator[ItemT],
    *,
    sentinel: SentinelT,
    quiet_after: float,
    interval: float,
    pop_ready: Callable[[], tuple[bool, ItemT]] | None = None,
) -> AsyncIterator[ItemT | SentinelT]:
    """Yield items from ``source``, emitting ``sentinel`` during quiet periods."""
    iterator = aiter(source)
    loop = asyncio.get_running_loop()

    pull_waiter: asyncio.Future[None] | None = None
    result_waiter: asyncio.Future[object] | None = None
    done_sentinel = object()

    async def _reader() -> None:
        nonlocal pull_waiter, result_waiter
        while True:
            waiter: asyncio.Future[None] = loop.create_future()
            pull_waiter = waiter
            await waiter
            pull_waiter = None
            target = result_waiter
            try:
                item = await anext(iterator)
            except StopAsyncIteration:
                if target is not None and not target.done():
                    target.set_result(done_sentinel)
                return
            except asyncio.CancelledError:
                if target is not None and not target.done():
                    target.cancel()
                raise
            except BaseException as exc:
                if target is not None and not target.done():
                    target.set_exception(exc)
                return
            else:
                if target is not None and not target.done():
                    target.set_result(item)

    reader_task: asyncio.Task[None] | None = None
    quiet = 0.0
    try:
        while True:
            if pop_ready is not None:
                has_ready, ready_item = pop_ready()
                if has_ready:
                    quiet = 0.0
                    yield ready_item
                    continue

            if reader_task is None:
                reader_task = asyncio.create_task(_reader())

            while pull_waiter is None:
                if reader_task.done():
                    return
                await asyncio.sleep(0)

            res_fut: asyncio.Future[object] = loop.create_future()
            result_waiter = res_fut
            waiter = pull_waiter
            pull_waiter = None
            if not waiter.done():
                waiter.set_result(None)

            while not res_fut.done():
                step = _keepalive_wait_step(quiet, quiet_after, interval)
                if step <= 0.0:
                    yield sentinel
                    step = interval
                    if res_fut.done():
                        break
                timeout_fut: asyncio.Future[None] = loop.create_future()

                def _wake_on_done(
                    _fut: asyncio.Future[object],
                    tf: asyncio.Future[None] = timeout_fut,
                ) -> None:
                    if not tf.done():
                        tf.set_result(None)

                res_fut.add_done_callback(_wake_on_done)
                timer = loop.call_later(
                    step,
                    lambda tf=timeout_fut: None if tf.done() else tf.set_result(None),
                )
                try:
                    await timeout_fut
                finally:
                    timer.cancel()
                    res_fut.remove_done_callback(_wake_on_done)

                if not res_fut.done():
                    quiet += step
                    if quiet >= quiet_after:
                        yield sentinel

            result_waiter = None
            item = res_fut.result()
            if item is done_sentinel:
                return
            quiet = 0.0
            yield cast(ItemT, item)
    finally:
        if pull_waiter is not None and not pull_waiter.done():
            pull_waiter.cancel()
        if reader_task is not None:
            await _discard_pending_task(reader_task)
        if result_waiter is not None:
            if not result_waiter.done():
                result_waiter.cancel()
            elif not result_waiter.cancelled():
                result_waiter.exception()

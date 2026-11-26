"""Provider execution shared by inbound API adapters."""

import asyncio
import math
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any, Literal

from loguru import logger

from free_claude_code.core.anthropic import (
    Message,
    SystemContent,
    Tool,
    anthropic_request_snapshot,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.trace import (
    LazyTraceValue,
    close_stream_input,
    trace_event,
    traced_async_stream,
)

from .ports import ProviderResolver
from .routing import RoutedMessagesRequest

TokenCounter = Callable[
    [list[Message], str | list[SystemContent] | None, list[Tool] | None],
    int,
]
WireApi = Literal["messages", "responses", "chat_completions"]


class ProviderExecutor:
    """Resolve a provider and execute one routed Anthropic Messages stream."""

    def __init__(
        self,
        provider_resolver: ProviderResolver,
        *,
        progress_timeout_seconds: float,
        token_counter: TokenCounter | None = None,
        generation_id: int | None = None,
        log_raw_payloads: bool = False,
    ) -> None:
        if not math.isfinite(progress_timeout_seconds) or progress_timeout_seconds <= 0:
            raise ValueError("progress_timeout_seconds must be finite and positive")
        self._provider_resolver = provider_resolver
        self._token_counter = token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads
        self._progress_timeout_seconds = float(progress_timeout_seconds)

    def _progress_timeout_failure(
        self,
        *,
        request_id: str,
        provider_id: str,
    ) -> ExecutionFailure:
        trace_event(
            stage="execution",
            event="free_claude_code.provider.progress_timeout",
            source="application",
            request_id=request_id,
            provider_id=provider_id,
            timeout_seconds=self._progress_timeout_seconds,
        )
        timeout_text = f"{self._progress_timeout_seconds:g}"
        return ExecutionFailure(
            kind=FailureKind.TIMEOUT,
            status_code=504,
            message=(
                f"Provider execution made no progress for {timeout_text} seconds.\n\n"
                f"Request ID: {request_id}"
            ),
            retryable=False,
        )

    def stream(
        self,
        routed: RoutedMessagesRequest,
        *,
        wire_api: WireApi,
        raw_log_label: str,
        raw_log_payload: object,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Preflight synchronously, then return the traced provider stream."""
        provider = self._provider_resolver(routed.resolved.provider_id)
        provider.preflight_stream(
            routed.request,
            reasoning=routed.reasoning,
        )

        gateway_model = routed.resolved.original_model
        route_trace: dict[str, object] = {
            "stage": "routing",
            "event": "free_claude_code.api.route.resolved",
            "source": "api",
            "request_id": request_id,
            "provider_id": routed.resolved.provider_id,
            "provider_model": routed.resolved.provider_model,
            "provider_model_ref": routed.resolved.provider_model_ref,
            "gateway_model": gateway_model,
            "reasoning_control": routed.reasoning.control.value,
            "reasoning_effort": (
                routed.reasoning.effort.value
                if routed.reasoning.effort is not None
                else None
            ),
            "reasoning_budget_tokens": routed.reasoning.budget_tokens,
        }
        if wire_api in ("responses", "chat_completions"):
            route_trace["wire_api"] = wire_api
        if self._generation_id is not None:
            route_trace["generation_id"] = self._generation_id
        trace_event(**route_trace)

        def _build_request_snapshot() -> dict[str, object]:
            snapshot = anthropic_request_snapshot(routed.request)
            snapshot["model"] = gateway_model
            return snapshot

        trace_event(
            stage="ingress",
            event=(
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.chat_completions.request.received"
                if wire_api == "chat_completions"
                else "free_claude_code.api.request.received"
            ),
            source="api",
            message_count=len(routed.request.messages),
            snapshot=LazyTraceValue(_build_request_snapshot),
            request_id=request_id,
        )

        if self._log_raw_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        async def provider_body() -> AsyncIterator[str]:
            provider_stream: AsyncIterator[str] | None = None
            token_count_task: asyncio.Task[int] | None = None
            try:
                if self._token_counter is not None:
                    # Always off the loop: an uncached large block (a file read,
                    # a long tool result) costs ~90ms per 100k tokens to encode,
                    # and every other in-flight stream would stall for it.
                    # Overlap counting with upstream connect when the provider
                    # accepts a deferred input-token awaitable.
                    if getattr(provider, "_supports_deferred_input_tokens", False):
                        token_count_task = asyncio.create_task(
                            asyncio.to_thread(
                                self._token_counter,
                                routed.request.messages,
                                routed.request.system,
                                routed.request.tools,
                            )
                        )
                        input_tokens: Any = token_count_task
                    else:
                        input_tokens = await asyncio.to_thread(
                            self._token_counter,
                            routed.request.messages,
                            routed.request.system,
                            routed.request.tools,
                        )
                else:
                    input_tokens = 0
                provider_stream = provider.stream_response(
                    routed.request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    response_model=gateway_model,
                    reasoning=routed.reasoning,
                )
                async for chunk in self._watch_progress(
                    provider_stream,
                    request_id=request_id,
                    provider_id=routed.resolved.provider_id,
                ):
                    yield chunk
            finally:
                if token_count_task is not None:
                    if not token_count_task.done():
                        token_count_task.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await token_count_task
                    elif not token_count_task.cancelled():
                        # Retrieve any unconsumed exception to avoid loop warnings.
                        token_count_task.exception()
                if provider_stream is not None:
                    await close_stream_input(
                        provider_stream,
                        owner="provider_executor",
                        source="api",
                        preserved_error=sys.exception(),
                    )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "provider_id": routed.resolved.provider_id,
            "gateway_model": gateway_model,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            provider_body(),
            stage="egress",
            source="api",
            complete_event=(
                "free_claude_code.api.responses.stream_completed"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_completed"
            ),
            interrupted_event=(
                "free_claude_code.api.responses.stream_interrupted"
                if wire_api == "responses"
                else "free_claude_code.api.response.stream_interrupted"
            ),
            chunk_event=None,
            extra=stream_trace,
        )

    def wrap_passthrough_stream(
        self,
        provider_stream: AsyncIterator[str],
        *,
        provider_id: str,
        gateway_model: str,
        request_id: str,
        raw_log_label: str | None = None,
        raw_log_payload: object = None,
    ) -> AsyncIterator[str]:
        """Apply progress-timeout protection and egress tracing to a passthrough stream."""
        if self._log_raw_payloads and raw_log_label is not None:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        async def passthrough_body() -> AsyncIterator[str]:
            try:
                async for chunk in self._watch_progress(
                    provider_stream,
                    request_id=request_id,
                    provider_id=provider_id,
                ):
                    yield chunk
            finally:
                await close_stream_input(
                    provider_stream,
                    owner="provider_executor_passthrough",
                    source="api",
                    preserved_error=sys.exception(),
                )

        stream_trace: dict[str, object] = {
            "request_id": request_id,
            "provider_id": provider_id,
            "gateway_model": gateway_model,
            "wire_api": "chat_completions",
            "passthrough": True,
        }
        if self._generation_id is not None:
            stream_trace["generation_id"] = self._generation_id

        return traced_async_stream(
            passthrough_body(),
            stage="egress",
            source="api",
            complete_event="free_claude_code.api.response.stream_completed",
            interrupted_event="free_claude_code.api.response.stream_interrupted",
            chunk_event=None,
            extra=stream_trace,
        )

    async def _watch_progress(
        self,
        provider_stream: AsyncIterator[str],
        *,
        request_id: str,
        provider_id: str,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        timeout_s = self._progress_timeout_seconds
        scheduled_deadline = loop.time() + timeout_s
        in_yield = False
        timed_out = False
        active_task: asyncio.Task[Any] | None = None
        cancelled_task: asyncio.Task[Any] | None = None
        watchdog: asyncio.TimerHandle | None = None

        def _is_non_progress_chunk(text: str) -> bool:
            first = text[0]
            if first == ":":
                return True
            if first == "e":
                return text.startswith("event: ping\n") or text == "event: ping"
            if first <= " ":
                stripped = text.strip()
                return (
                    stripped.startswith(":")
                    or stripped.startswith("event: ping\n")
                    or stripped == "event: ping"
                )
            return False

        def _on_progress_watchdog() -> None:
            nonlocal watchdog, timed_out, cancelled_task
            watchdog = None
            if in_yield:
                return
            now = loop.time()
            if now >= scheduled_deadline - 1e-6:
                timed_out = True
                target = active_task
                if target is not None and not target.done():
                    cancelled_task = target
                    target.cancel()
                return
            watchdog = loop.call_at(scheduled_deadline, _on_progress_watchdog)

        provider_iter = aiter(provider_stream)
        watchdog = loop.call_at(scheduled_deadline, _on_progress_watchdog)
        try:
            while True:
                active_task = asyncio.current_task()
                try:
                    chunk = await anext(provider_iter)
                except StopAsyncIteration:
                    break
                finally:
                    active_task = None

                if not chunk:
                    if loop.time() >= scheduled_deadline:
                        raise self._progress_timeout_failure(
                            request_id=request_id,
                            provider_id=provider_id,
                        )
                    active_task = asyncio.current_task()
                    try:
                        await asyncio.sleep(0)
                    finally:
                        active_task = None
                    continue

                is_non_progress = _is_non_progress_chunk(chunk)
                remaining_budget = (
                    max(0.0, scheduled_deadline - loop.time())
                    if is_non_progress
                    else timeout_s
                )
                in_yield = True
                try:
                    yield chunk
                finally:
                    in_yield = False
                if not is_non_progress:
                    scheduled_deadline = loop.time() + timeout_s
                else:
                    if remaining_budget <= 0.0:
                        raise self._progress_timeout_failure(
                            request_id=request_id,
                            provider_id=provider_id,
                        )
                    scheduled_deadline = loop.time() + remaining_budget
                if watchdog is None:
                    watchdog = loop.call_at(scheduled_deadline, _on_progress_watchdog)
        except asyncio.CancelledError as exc:
            if not timed_out:
                raise
            current = asyncio.current_task()
            if (
                current is None
                or cancelled_task is None
                or current is not cancelled_task
                or current.uncancel() > 0
            ):
                raise
            raise self._progress_timeout_failure(
                request_id=request_id,
                provider_id=provider_id,
            ) from exc
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog = None

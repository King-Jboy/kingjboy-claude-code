"""ChatGPT Codex backend provider using OpenAI Responses."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress
from typing import Any

import httpx
import orjson

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import anthropic_ping_frame
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
    extract_upstream_error_detail,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import ReplayOrigin, prepare_history
from free_claude_code.core.openai_responses import (
    ResponsesConversionError,
    ResponsesProviderStream,
    ResponsesStreamFailure,
    build_responses_provider_request,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.core.version import package_version
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderAttempt,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
)
from free_claude_code.providers.history_replay import history_retry_body
from free_claude_code.providers.stream_recovery import (
    HoldbackSignal,
    RecoveryController,
    RecoveryFailureAction,
)

from .auth import OpenAIAccess, OpenAIAuthManager, OpenAIReconnectRequired
from .login import OPENAI_CODEX_ORIGINATOR

FCC_VERSION = package_version()

KEEPALIVE_INTERVAL_SECONDS = 1.5
UPSTREAM_QUIET_KEEPALIVE_SECONDS = 5.0


async def _settled_within(task: asyncio.Task[Any], timeout: float) -> bool:
    """Return whether ``task`` finished within ``timeout`` seconds."""
    if task.done():
        return True
    done, _ = await asyncio.wait((task,), timeout=timeout)
    return bool(done)


def _keepalive_wait_step(quiet: float, quiet_after: float, interval: float) -> float:
    if quiet >= quiet_after:
        return interval
    return max(0.0, quiet_after - quiet)


async def _discard_pending_task(task: asyncio.Task[Any]) -> None:
    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def _sse_with_keepalive(
    events: AsyncIterator[tuple[str, dict[str, Any]]],
    *,
    quiet_after: float = UPSTREAM_QUIET_KEEPALIVE_SECONDS,
    interval: float = KEEPALIVE_INTERVAL_SECONDS,
) -> AsyncIterator[tuple[str, dict[str, Any]] | None]:
    """Yield upstream Responses SSE events, emitting None during long silences."""
    iterator = aiter(events)
    loop = asyncio.get_running_loop()

    pull_waiter: asyncio.Future[None] | None = None
    result_waiter: asyncio.Future[Any] | None = None
    done_sentinel = object()

    async def _reader() -> None:
        nonlocal pull_waiter, result_waiter
        while True:
            waiter = loop.create_future()
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
            if reader_task is None:
                reader_task = asyncio.create_task(_reader())

            while pull_waiter is None:
                if reader_task.done():
                    return
                await asyncio.sleep(0)

            res_fut: asyncio.Future[Any] = loop.create_future()
            result_waiter = res_fut
            waiter = pull_waiter
            pull_waiter = None
            if not waiter.done():
                waiter.set_result(None)

            while not res_fut.done():
                step = _keepalive_wait_step(quiet, quiet_after, interval)
                if step <= 0.0:
                    yield None
                    step = interval
                    if res_fut.done():
                        break
                timeout_fut: asyncio.Future[None] = loop.create_future()

                def _wake_on_done(
                    _fut: asyncio.Future[Any],
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
                        yield None

            result_waiter = None
            item = res_fut.result()
            if item is done_sentinel:
                return
            quiet = 0.0
            yield item
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

class _TruncatedResponsesStream(RetryableProviderProtocolError):
    """A Responses stream ended without a terminal lifecycle event."""


class OpenAICodexProvider(BaseProvider):
    """Use a ChatGPT subscription through OpenAI's Codex backend."""

    _supports_deferred_input_tokens: bool = True

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: OpenAIAuthManager,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._auth = auth
        self._admission = admission
        self._client_headers = {
            "User-Agent": f"{OPENAI_CODEX_ORIGINATOR}/{FCC_VERSION}",
            "originator": OPENAI_CODEX_ORIGINATOR,
            "version": FCC_VERSION,
        }
        use_http2 = not (config.proxy and config.proxy.lower().startswith("socks"))
        self._client = client or httpx.AsyncClient(
            base_url=f"{config.base_url.rstrip('/')}/",
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            headers=self._client_headers,
            http2=use_http2,
        )
        self._owns_client = client is None

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate and adapt the private Codex request before upstream I/O."""

        self._build_body(request, reasoning=reasoning)

    async def cleanup(self) -> None:
        """Close only provider-owned transport resources."""

        if self._owns_client:
            await self._client.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Discover models visible to the currently connected ChatGPT account."""

        async def fetch() -> Any:
            access = await self._auth.access()
            response = await self._client.get(
                "models",
                params={"client_version": FCC_VERSION},
                headers={**self._client_headers, **_auth_headers(access)},
            )
            if response.status_code == 401:
                access = await self._auth.recover_unauthorized(access.access_token)
                response = await self._client.get(
                    "models",
                    params={"client_version": FCC_VERSION},
                    headers={**self._client_headers, **_auth_headers(access)},
                )
            response.raise_for_status()
            return response.json()

        payload = await self._admission.run_with_retry(fetch)
        return _model_infos(payload)

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int | Awaitable[int] = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream Responses output in Anthropic Messages format."""

        tool_names = OpenAIToolNameCodec.from_request(request)
        body = self._build_body(request, reasoning=reasoning)
        return self._run_stream(
            body,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            tool_names=tool_names,
        )

    @staticmethod
    def _build_body(
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        try:
            body = build_responses_provider_request(request, reasoning=reasoning)
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc
        # The private Codex backend rejects these public Responses fields.
        # Codex itself omits the output cap and uses separate internal metadata.
        body.pop("max_output_tokens", None)
        body.pop("metadata", None)
        origin = ReplayOrigin("openai_codex", "responses", "", "", request.model)
        return prepare_history(body, origin)

    async def _run_stream(
        self,
        body: dict[str, Any],
        *,
        input_tokens: int | Awaitable[int],
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
    ) -> AsyncIterator[str]:
        downstream_model = body.get("model")
        retry_session = self._admission.new_retry_session(
            request_id=request_id,
            model=downstream_model if isinstance(downstream_model, str) else None,
        )
        recovery = RecoveryController()
        message_id = f"msg_{uuid.uuid4()}"
        session_id = str(uuid.uuid4())
        authentication_recovered = False
        message_started_committed = False
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider="openai",
            request_id=request_id,
            gateway_model=response_model,
            downstream_model=downstream_model,
            item_count=len(body.get("input", [])),
            tool_count=len(body.get("tools", [])),
        )

        resolved_input_tokens: int | None = (
            input_tokens if isinstance(input_tokens, int) else None
        )

        async def _get_input_tokens() -> int:
            nonlocal resolved_input_tokens
            if resolved_input_tokens is None:
                resolved_input_tokens = int(await input_tokens)  # type: ignore[misc]
            return resolved_input_tokens

        while retry_session.can_attempt:
            stream = ResponsesProviderStream(
                message_id=message_id,
                model=response_model,
                input_tokens=resolved_input_tokens or 0,
                origin=ReplayOrigin(
                    "openai_codex", "responses", "", "", response_model
                ),
                log_raw_events=self._config.log_raw_sse_events,
                tool_names=tool_names,
            )
            if message_started_committed:
                stream.ledger.message_started = True

            response: httpx.Response | None = None
            attempt: ProviderAttempt | None = None
            stream_opened = False
            try:
                access = await self._auth.access()
                attempt = await self._admission.open_attempt(retry_session)
                request_headers = {
                    **self._client_headers,
                    **_auth_headers(access),
                    "Accept": "text/event-stream",
                    "session_id": session_id,
                }
                try:
                    body_bytes = orjson.dumps(body)
                except TypeError:
                    http_request = self._client.build_request(
                        "POST",
                        "responses",
                        json=body,
                        headers=request_headers,
                    )
                else:
                    http_request = self._client.build_request(
                        "POST",
                        "responses",
                        content=body_bytes,
                        headers={
                            **request_headers,
                            "Content-Type": "application/json",
                        },
                    )
                send_task = asyncio.ensure_future(
                    self._client.send(http_request, stream=True)
                )
                try:
                    waited = 0.0
                    while not await _settled_within(
                        send_task, KEEPALIVE_INTERVAL_SECONDS
                    ):
                        waited += KEEPALIVE_INTERVAL_SECONDS
                        if (
                            recovery.committed
                            or waited >= UPSTREAM_QUIET_KEEPALIVE_SECONDS
                        ):
                            if not stream.ledger.message_started:
                                stream.ledger.input_tokens = await _get_input_tokens()
                                for event in stream.start():
                                    for held in recovery.push(event):
                                        yield held
                            for event in recovery.flush():
                                yield event
                            message_started_committed = True
                            yield anthropic_ping_frame()
                except BaseException:
                    if (
                        send_task.done()
                        and not send_task.cancelled()
                        and send_task.exception() is None
                    ):
                        response = send_task.result()
                    else:
                        await _discard_pending_task(send_task)
                    raise
                response = send_task.result()
                if response.status_code == 401 and not authentication_recovered:
                    await _read_bounded_body(response)
                    await self._auth.recover_unauthorized(access.access_token)
                    await attempt.retry_immediately()
                    authentication_recovered = True
                    if not message_started_committed:
                        recovery.discard()
                    continue
                if not response.is_success:
                    body_bytes, body_truncated = await _read_bounded_body(response)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        attach_upstream_error_body(
                            exc,
                            body_bytes,
                            truncated=body_truncated,
                        )
                        raise
                content_type = response.headers.get("content-type")
                if content_type and "text/event-stream" not in content_type.lower():
                    body_bytes, body_truncated = await _read_bounded_body(response)
                    error = _TruncatedResponsesStream(
                        "OpenAI returned a non-streaming Responses payload."
                    )
                    attach_upstream_error_body(
                        error,
                        body_bytes,
                        truncated=body_truncated,
                    )
                    raise error
                stream_opened = True
                if not stream.ledger.message_started:
                    stream.ledger.input_tokens = await _get_input_tokens()
                    for event in stream.start():
                        for held in recovery.push(event):
                            yield held

                async with recovery.read_stream(
                    _sse_with_keepalive(_iter_sse(response))
                ) as sse_items:
                    async for sse_item in sse_items:
                        if sse_item is HoldbackSignal.EXPIRED:
                            for event in recovery.flush():
                                yield event
                            message_started_committed = True
                            continue
                        if sse_item is None:
                            for event in recovery.flush():
                                yield event
                            message_started_committed = True
                            yield anthropic_ping_frame()
                            continue
                        event_type, payload = sse_item
                        if not attempt.accepted:
                            await attempt.succeeded()
                        for event in stream.feed(event_type, payload):
                            for held in recovery.push(event):
                                yield held
                if not stream.completed:
                    raise _TruncatedResponsesStream(
                        "OpenAI Responses stream ended without a terminal event."
                    )
                for event in recovery.flush():
                    yield event
                trace_event(
                    stage="provider",
                    event="provider.response.completed",
                    source="provider",
                    provider="openai",
                    request_id=request_id,
                )
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = _effective_error(raw_error)
                post_acceptance_retry_allowed = True
                if attempt is not None:
                    if attempt.accepted:
                        post_acceptance_retry_allowed = (
                            await attempt.retry_after_acceptance(error)
                        )
                    else:
                        await attempt.retry(error)
                retryable = (
                    attempt.failure_retryable
                    if attempt is not None and attempt.failure_retryable is not None
                    else None
                )
                decision = recovery.advance_failure(
                    error,
                    stream_opened=stream_opened,
                    generated_output=stream.generated_output,
                    complete_tool_salvageable=False,
                    attempts_remaining=retry_session.attempts_remaining,
                    retryable_override=retryable,
                )
                if not message_started_committed:
                    repaired_body = history_retry_body(error, body, "responses")
                    if repaired_body is not None and retry_session.can_attempt:
                        body = repaired_body
                        recovery.discard()
                        continue

                should_early_retry = post_acceptance_retry_allowed and (
                    decision.action == RecoveryFailureAction.EARLY_RETRY
                    or (
                        (not decision.committed or not stream.generated_output)
                        and decision.retryable
                        and retry_session.can_attempt
                    )
                )
                if should_early_retry:
                    if decision.committed:
                        message_started_committed = True
                        recovery.flush()
                    else:
                        recovery.discard()
                    trace_event(
                        stage="provider",
                        event="provider.recovery.early_retry",
                        source="provider",
                        provider="openai",
                        request_id=request_id,
                        attempts_started=retry_session.attempts_started,
                        max_attempts=retry_session.max_attempts,
                    )
                    continue

                failure = classify_provider_failure(
                    error,
                    provider_name="OpenAI",
                    read_timeout_s=self._config.http_read_timeout,
                    request_id=request_id,
                )
                self._log_stream_transport_error(
                    "OPENAI",
                    f" request_id={request_id}" if request_id else "",
                    error,
                    request_id=request_id,
                )
                if not decision.committed:
                    recovery.discard()
                    raise failure from raw_error
                for event in stream.ledger.close_unclosed_blocks():
                    yield event
                raise failure from raw_error
            finally:
                if response is not None:
                    await response.aclose()
                if attempt is not None:
                    await attempt.aclose()

        raise RuntimeError("OpenAI retry session ended without a terminal result.")


async def _iter_sse(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_type = ""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if not data_lines:
                event_type = ""
                continue
            raw_data = "\n".join(data_lines)
            data_lines = []
            if not raw_data.strip():
                event_type = ""
                continue
            if raw_data == "[DONE]":
                return
            try:
                payload = orjson.loads(raw_data)
            except orjson.JSONDecodeError:
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise _TruncatedResponsesStream(
                        "OpenAI returned malformed Responses SSE."
                    ) from exc
            if not isinstance(payload, dict):
                raise _TruncatedResponsesStream(
                    "OpenAI returned a non-object Responses event."
                )
            resolved_type = event_type or payload.get("type")
            event_type = ""
            if isinstance(resolved_type, str) and resolved_type:
                yield resolved_type, payload
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            val = line[5:]
            if val.startswith(" "):
                val = val[1:]
            data_lines.append(val)
    if data_lines:
        if "\n".join(data_lines).strip() == "[DONE]":
            return
        raise _TruncatedResponsesStream(
            "OpenAI Responses stream ended during an SSE event."
        )


async def _read_bounded_body(
    response: httpx.Response,
) -> tuple[bytes, bool]:
    limit = ERROR_DETAIL_DISPLAY_CAP_BYTES
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = limit + 1 - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) > limit:
            break
    truncated = len(body) > limit
    return bytes(body[:limit]), truncated


def _auth_headers(access: OpenAIAccess) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access.access_token}",
        "ChatGPT-Account-ID": access.account_id,
    }
    if access.fedramp:
        headers["X-OpenAI-Fedramp"] = "true"
    return headers


def _model_infos(payload: Any) -> frozenset[ProviderModelInfo]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("OpenAI model-list response is missing the models array.")
    infos: set[ProviderModelInfo] = set()
    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("slug")
        visibility = model.get("visibility")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or visibility != "list"
        ):
            continue
        efforts = model.get(
            "supported_reasoning_levels",
            model.get("supported_reasoning_efforts"),
        )
        infos.add(
            ProviderModelInfo(
                model_id=model_id,
                supports_thinking=bool(efforts) if isinstance(efforts, list) else None,
            )
        )
    if not infos:
        raise ValueError("OpenAI did not advertise any visible models.")
    return frozenset(infos)


def _effective_error(error: Exception) -> Exception:
    if isinstance(error, OpenAIReconnectRequired):
        return ExecutionFailure(
            kind=FailureKind.AUTHENTICATION,
            status_code=401,
            message=str(error),
            retryable=False,
        )
    if isinstance(error, ResponsesStreamFailure):
        message = (
            extract_upstream_error_detail(error).exception_text
            or "OpenAI response failed."
        )
        code = (error.code or "").lower()
        if is_context_window_error_code(code):
            return context_window_exceeded_provider_failure(message)
        if "rate" in code or "429" in code:
            return ExecutionFailure(FailureKind.RATE_LIMIT, 429, message, True)
        if any(marker in code for marker in ("overload", "capacity", "529")):
            return ExecutionFailure(FailureKind.OVERLOADED, 529, message, True)
        retryable = any(
            marker in code
            for marker in ("server", "internal", "unavailable", "timeout")
        )
        return ExecutionFailure(FailureKind.UPSTREAM, 502, message, retryable)
    return error

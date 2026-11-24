"""OpenAI Chat Completions API product flow with direct Chat-to-Chat passthrough."""

from typing import Any

from fastapi.responses import JSONResponse

from free_claude_code.api.request_errors import (
    http_status_for_unexpected_api_exception,
    log_unexpected_api_exception,
    require_non_empty_messages,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import (
    FastJSONResponse,
    openai_chat_sse_streaming_response,
    terminal_execution_error_response,
    trace_terminal_execution_error,
)
from free_claude_code.application.errors import ApplicationError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import (
    ChatCompletionsPassthroughProvider,
    ProviderResolver,
)
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
from free_claude_code.core.openai_chat_wire import (
    OPENAI_CHAT_SSE_HEADERS,
    OpenAIChatCompletionsRequest,
    build_passthrough_chat_body,
    chat_request_to_anthropic_messages_request,
    collect_openai_chat_completion_from_sse,
    iter_openai_chat_sse_from_anthropic,
    new_chat_completion_id,
)
from free_claude_code.core.openai_responses import (
    openai_error_payload,
    openai_error_type_for_failure,
    openai_failure_payload,
)
from free_claude_code.core.trace import trace_event


class ChatCompletionsHandler:
    """Handle OpenAI Chat Completions requests (streaming and non-streaming)."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        *,
        model_router: ModelRouter | None = None,
        provider_executor: ProviderExecutor | None = None,
        generation_id: int | None = None,
    ) -> None:
        self._settings = settings
        self._provider_resolver = provider_resolver
        self._model_router = model_router or ModelRouter(settings)
        self._generation_id = generation_id
        self._provider_executor = provider_executor or ProviderExecutor(
            provider_resolver,
            progress_timeout_seconds=settings.provider_progress_timeout,
            generation_id=generation_id,
            log_raw_payloads=settings.log_raw_api_payloads,
        )

    async def create(
        self,
        request_data: OpenAIChatCompletionsRequest,
        *,
        request_id: str | None = None,
    ) -> object:
        """Create a streaming or JSON OpenAI Chat Completions response."""
        request_id = request_id or new_request_id()
        completion_id = new_chat_completion_id()
        try:
            require_non_empty_messages(request_data.messages)
            resolved = self._model_router.resolve(request_data.model)
            provider = self._provider_resolver(resolved.provider_id)

            if isinstance(provider, ChatCompletionsPassthroughProvider):
                # Direct Chat-to-Chat fast path: bypasses Anthropic conversion completely.
                route_trace: dict[str, object] = {
                    "stage": "routing",
                    "event": "free_claude_code.api.route.resolved",
                    "source": "api",
                    "wire_api": "chat_completions",
                    "passthrough": True,
                    "request_id": request_id,
                    "provider_id": resolved.provider_id,
                    "provider_model": resolved.provider_model,
                    "provider_model_ref": resolved.provider_model_ref,
                    "gateway_model": resolved.original_model,
                }
                if self._generation_id is not None:
                    route_trace["generation_id"] = self._generation_id
                trace_event(**route_trace)

                body = build_passthrough_chat_body(
                    request_data,
                    provider_model=resolved.provider_model,
                    force_reasoning_off=(
                        resolved.reasoning_preference is ReasoningPreference.OFF
                    ),
                )
                sse_stream = provider.stream_openai_chat_passthrough(
                    body,
                    request_id=request_id,
                    response_model=resolved.original_model,
                )
            else:
                # Universal fallback for non-Chat upstreams (Codex, Gemini, Bedrock, etc.).
                request_payload: Any = (
                    request_data.model_dump(mode="json", exclude_none=True)
                    if self._settings.log_raw_api_payloads
                    else None
                )
                anthropic_request = chat_request_to_anthropic_messages_request(
                    request_data
                )
                require_non_empty_messages(anthropic_request.messages)
                routed = self._model_router.resolve_messages_request(anthropic_request)
                anthropic_stream = self._provider_executor.stream(
                    routed,
                    wire_api="chat_completions",
                    raw_log_label="FULL_CHAT_COMPLETIONS_PAYLOAD",
                    raw_log_payload=request_payload,
                    request_id=request_id,
                )
                include_usage = bool(
                    not request_data.stream
                    or (
                        isinstance(request_data.stream_options, dict)
                        and request_data.stream_options.get("include_usage")
                    )
                )
                sse_stream = iter_openai_chat_sse_from_anthropic(
                    anthropic_stream,
                    model=resolved.original_model,
                    completion_id=completion_id,
                    include_usage=include_usage,
                )

            if request_data.stream:
                return await openai_chat_sse_streaming_response(
                    sse_stream,
                    headers=OPENAI_CHAT_SSE_HEADERS,
                    pre_start_error_response=lambda exc: self._pre_start_error_response(
                        exc, request_id=request_id
                    ),
                )

            completion = await collect_openai_chat_completion_from_sse(
                sse_stream,
                model=resolved.original_model,
                completion_id=completion_id,
            )
            return FastJSONResponse(content=completion)
        except ApplicationError:
            raise
        except ExecutionFailure as exc:
            return self._execution_failure_response(exc, request_id=request_id)
        except Exception as exc:
            failure = find_execution_failure(exc)
            if failure is not None:
                return self._execution_failure_response(failure, request_id=request_id)
            log_unexpected_api_exception(
                self._settings,
                exc,
                context="CREATE_CHAT_COMPLETION_ERROR",
                request_id=request_id,
            )
            return FastJSONResponse(
                status_code=http_status_for_unexpected_api_exception(exc),
                content=openai_error_payload(
                    message=safe_exception_message(exc),
                    error_type="api_error",
                ),
            )

    def _pre_start_error_response(
        self, exc: BaseException, *, request_id: str
    ) -> JSONResponse:
        failure = find_execution_failure(exc)
        if failure is not None:
            return self._execution_failure_response(failure, request_id=request_id)
        log_unexpected_api_exception(
            self._settings,
            exc,
            context="CREATE_CHAT_COMPLETION_STREAM_START_ERROR",
            request_id=request_id,
        )
        status_code = http_status_for_unexpected_api_exception(exc)
        trace_terminal_execution_error(
            wire_api="chat_completions",
            request_id=request_id,
            status_code=status_code,
            error_type="api_error",
            error=exc,
        )
        return terminal_execution_error_response(
            status_code=status_code,
            content=openai_error_payload(
                message=safe_exception_message(exc),
                error_type="api_error",
            ),
        )

    def _execution_failure_response(
        self,
        failure: ExecutionFailure,
        *,
        request_id: str,
    ) -> JSONResponse:
        error_type = openai_error_type_for_failure(failure)
        trace_terminal_execution_error(
            wire_api="chat_completions",
            request_id=request_id,
            status_code=failure.status_code,
            error_type=error_type,
            error=failure,
        )
        return terminal_execution_error_response(
            status_code=failure.status_code,
            content=openai_failure_payload(failure),
        )

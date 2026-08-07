"""Provider execution shared by inbound API adapters."""

import sys
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Literal

from loguru import logger

from free_claude_code.core.anthropic import (
    Message,
    SystemContent,
    Tool,
    anthropic_request_snapshot,
    get_token_count,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.trace import (
    close_stream_input,
    trace_event,
    traced_async_stream,
)

from .ports import ProviderPort, ProviderResolver
from .routing import ModelRouter, RoutedMessagesRequest

TokenCounter = Callable[
    [list[Message], str | list[SystemContent] | None, list[Tool] | None],
    int,
]
WireApi = Literal["messages", "responses"]

# Failures that mean "this model has no capacity right now" - the only ones
# another provider could plausibly serve. A bad request or a blown context
# window would fail identically everywhere, so retrying those elsewhere would
# just multiply the latency before returning the same error.
_NO_CAPACITY_KINDS = frozenset(
    {
        FailureKind.RATE_LIMIT,
        FailureKind.OVERLOADED,
        FailureKind.UNAVAILABLE,
    }
)


def _has_no_capacity(failure: ExecutionFailure) -> bool:
    """Return whether another provider could plausibly serve this request."""
    return failure.kind in _NO_CAPACITY_KINDS


class ProviderExecutor:
    """Resolve a provider and execute one routed Anthropic Messages stream."""

    def __init__(
        self,
        provider_resolver: ProviderResolver,
        *,
        token_counter: TokenCounter = get_token_count,
        generation_id: int | None = None,
        log_raw_payloads: bool = False,
        model_router: ModelRouter | None = None,
        fallback_refs: Sequence[str] = (),
    ) -> None:
        self._provider_resolver = provider_resolver
        self._token_counter = token_counter
        self._generation_id = generation_id
        self._log_raw_payloads = log_raw_payloads
        self._model_router = model_router
        self._fallback_refs = tuple(fallback_refs)

    def _fallback_attempts(
        self, routed: RoutedMessagesRequest
    ) -> list[tuple[ProviderPort, RoutedMessagesRequest]]:
        """Return the routed request followed by every usable fallback.

        A fallback that cannot be resolved - an unknown provider, or one with no
        credentials - is dropped here rather than raised. The chain exists to
        rescue a request that is already failing, so a bad entry in it must
        never be the reason a good entry is not reached.
        """
        attempts: list[tuple[ProviderPort, RoutedMessagesRequest]] = [
            (self._provider_resolver(routed.resolved.provider_id), routed)
        ]
        if self._model_router is None:
            return attempts

        seen = {routed.resolved.provider_model_ref}
        for ref in self._fallback_refs:
            if ref in seen:
                continue
            seen.add(ref)
            try:
                candidate = self._model_router.resolve_messages_request(
                    routed.request.model_copy(update={"model": ref}, deep=True)
                )
                provider = self._provider_resolver(candidate.resolved.provider_id)
            except Exception as exc:
                logger.warning(
                    "Fallback {} is not usable and was skipped: {}",
                    ref,
                    type(exc).__name__,
                )
                continue
            attempts.append((provider, candidate))
        return attempts

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
        if wire_api == "responses":
            route_trace["wire_api"] = "responses"
        if self._generation_id is not None:
            route_trace["generation_id"] = self._generation_id
        trace_event(**route_trace)

        request_snapshot = anthropic_request_snapshot(routed.request)
        request_snapshot["model"] = gateway_model
        trace_event(
            stage="ingress",
            event=(
                "free_claude_code.api.responses.request.received"
                if wire_api == "responses"
                else "free_claude_code.api.request.received"
            ),
            source="api",
            message_count=len(routed.request.messages),
            snapshot=request_snapshot,
            request_id=request_id,
        )

        if self._log_raw_payloads:
            logger.debug(f"{raw_log_label} [{{}}]: {{}}", request_id, raw_log_payload)

        input_tokens = self._token_counter(
            routed.request.messages,
            routed.request.system,
            routed.request.tools,
        )

        async def provider_body() -> AsyncIterator[str]:
            # Each attempt is (provider, routed request). The first is whatever
            # routing chose; the rest are the configured fallbacks, resolved
            # lazily so a chain naming an unconfigured provider only costs
            # anything if it is actually reached.
            attempts = self._fallback_attempts(routed)
            last_index = len(attempts) - 1
            for index, attempt in enumerate(attempts):
                attempt_provider, attempt_routed = attempt
                provider_stream: AsyncIterator[str] | None = None
                emitted = False
                try:
                    provider_stream = attempt_provider.stream_response(
                        attempt_routed.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        response_model=gateway_model,
                        reasoning=attempt_routed.reasoning,
                    )
                    async for chunk in provider_stream:
                        emitted = True
                        yield chunk
                    return
                except ExecutionFailure as failure:
                    # Once a byte is out the client owns a committed response;
                    # switching models underneath it would splice two different
                    # completions into one message.
                    if emitted or index == last_index or not _has_no_capacity(failure):
                        raise
                    trace_event(
                        stage="routing",
                        event="free_claude_code.api.route.fallback",
                        source="api",
                        request_id=request_id,
                        from_provider_model_ref=(
                            attempt_routed.resolved.provider_model_ref
                        ),
                        to_provider_model_ref=(
                            attempts[index + 1][1].resolved.provider_model_ref
                        ),
                        failure_kind=failure.kind.value,
                        status_code=failure.status_code,
                    )
                    logger.warning(
                        "{} has no capacity ({}); falling back to {}",
                        attempt_routed.resolved.provider_model_ref,
                        failure.kind.value,
                        attempts[index + 1][1].resolved.provider_model_ref,
                    )
                finally:
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

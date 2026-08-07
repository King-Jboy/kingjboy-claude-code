"""Application-owned provider execution contracts."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import (
    ModelRouter,
    ResolvedModel,
    RoutedMessagesRequest,
)
from free_claude_code.config.model_refs import model_fallback_refs
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy


class FakeProvider:
    def __init__(self) -> None:
        self.preflight_calls: list[tuple[MessagesRequest, ReasoningPolicy]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.stream_close_calls = 0

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        self.preflight_calls.append((request, reasoning))

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        try:
            yield "event: message_stop\ndata: {}\n\n"
        finally:
            self.stream_close_calls += 1


class FailingPreflightProvider(FakeProvider):
    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        raise ValueError("invalid provider request")


class FailingStreamConstructionProvider(FakeProvider):
    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream construction failed")


def _routed_request() -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModel(
            original_model="gateway-model",
            provider_id="provider",
            provider_model="provider-model",
            provider_model_ref="provider/provider-model",
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


@pytest.mark.asyncio
async def test_executor_uses_structural_provider_port_and_preflights_eagerly() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload=request.model_dump(),
        request_id="req_application",
    )

    assert provider.preflight_calls == [(request, ReasoningPolicy.on())]
    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert provider.stream_calls == [
        {
            "request": request,
            "input_tokens": 17,
            "request_id": "req_application",
            "response_model": "gateway-model",
            "reasoning": ReasoningPolicy.on(),
        }
    ]
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_closing_executor_stream_closes_provider_stream_once() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_early_close",
    )

    assert await anext(stream) == "event: message_stop\ndata: {}\n\n"
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_stream_construction_failure_remains_deferred_to_iteration() -> None:
    provider = FailingStreamConstructionProvider()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_deferred_construction",
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)


def test_executor_preflight_failure_stays_before_token_count_and_stream() -> None:
    provider = FailingPreflightProvider()
    token_counter = MagicMock(return_value=17)
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=token_counter,
    )

    with pytest.raises(ValueError, match="invalid provider request"):
        executor.stream(
            _routed_request(),
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload={},
            request_id="req_application",
        )

    token_counter.assert_not_called()
    assert provider.stream_calls == []


class RateLimitedProvider(FakeProvider):
    """A provider whose capacity has run out before it emits anything."""

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="account is rate limited",
            retryable=False,
        )


class InvalidRequestProvider(FakeProvider):
    """A provider refusing the request itself, which no peer would accept."""

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        raise ExecutionFailure(
            kind=FailureKind.INVALID_REQUEST,
            status_code=400,
            message="malformed request",
            retryable=False,
        )


class FailsAfterEmittingProvider(FakeProvider):
    """A provider that dies once the client already holds part of a message."""

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        yield "event: message_start\ndata: {}\n\n"
        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="account is rate limited mid-stream",
            retryable=False,
        )


def _nim_routed_request() -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="model-a",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModel(
            original_model="gateway-model",
            provider_id="nvidia_nim",
            provider_model="model-a",
            provider_model_ref="nvidia_nim/model-a",
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


def _executor_with_fallbacks(
    providers: dict[str, FakeProvider],
    fallbacks: str,
) -> ProviderExecutor:
    settings = Settings(MODEL_FALLBACKS=fallbacks)
    return ProviderExecutor(
        lambda provider_id: providers[provider_id],
        token_counter=lambda _messages, _system, _tools: 17,
        model_router=ModelRouter(settings),
        fallback_refs=model_fallback_refs(settings),
    )


async def _drain(
    executor: ProviderExecutor, routed: RoutedMessagesRequest
) -> list[str]:
    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_fallback",
    )
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_a_rate_limited_model_falls_back_to_the_next_provider() -> None:
    # The whole point: a capped account must not surface as an error the client
    # has to back off from, when another provider still has capacity.
    spare = FakeProvider()
    executor = _executor_with_fallbacks(
        {"nvidia_nim": RateLimitedProvider(), "open_router": spare},
        '["open_router/model-b"]',
    )

    assert await _drain(executor, _nim_routed_request()) == [
        "event: message_stop\ndata: {}\n\n"
    ]
    assert len(spare.stream_calls) == 1


@pytest.mark.asyncio
async def test_a_request_level_failure_is_not_retried_elsewhere() -> None:
    # A malformed request fails identically on every provider, so walking the
    # chain would only multiply latency before returning the same error.
    spare = FakeProvider()
    executor = _executor_with_fallbacks(
        {"nvidia_nim": InvalidRequestProvider(), "open_router": spare},
        '["open_router/model-b"]',
    )

    with pytest.raises(ExecutionFailure) as error:
        await _drain(executor, _nim_routed_request())

    assert error.value.kind is FailureKind.INVALID_REQUEST
    assert spare.stream_calls == [], "a fallback must not absorb a bad request"


@pytest.mark.asyncio
async def test_a_committed_response_is_never_spliced_onto_a_fallback() -> None:
    # Once bytes are out the client owns a committed message; continuing it from
    # a different model would stitch two completions into one reply.
    spare = FakeProvider()
    executor = _executor_with_fallbacks(
        {"nvidia_nim": FailsAfterEmittingProvider(), "open_router": spare},
        '["open_router/model-b"]',
    )

    with pytest.raises(ExecutionFailure):
        await _drain(executor, _nim_routed_request())

    assert spare.stream_calls == []


@pytest.mark.asyncio
async def test_an_unusable_fallback_is_skipped_rather_than_fatal() -> None:
    # The chain rescues a request that is already failing, so a broken entry in
    # it must never be the reason a good entry goes unreached.
    spare = FakeProvider()
    providers: dict[str, FakeProvider] = {
        "nvidia_nim": RateLimitedProvider(),
        "open_router": spare,
    }

    def resolve(provider_id: str) -> FakeProvider:
        if provider_id == "mistral":
            raise RuntimeError("mistral has no credentials configured")
        return providers[provider_id]

    settings = Settings(MODEL_FALLBACKS='["mistral/broken", "open_router/model-b"]')
    executor = ProviderExecutor(
        resolve,
        token_counter=lambda _messages, _system, _tools: 17,
        model_router=ModelRouter(settings),
        fallback_refs=model_fallback_refs(settings),
    )

    assert await _drain(executor, _nim_routed_request()) == [
        "event: message_stop\ndata: {}\n\n"
    ]
    assert len(spare.stream_calls) == 1

import pytest
import asyncio
"""Provider stream commit-boundary and recovery policy."""

import httpx
import httpx2
import openai

from free_claude_code.providers.stream_recovery import (
    HoldbackSignal,
    RecoveryController,
    RecoveryFailureAction,
    RecoveryHoldbackBuffer,
    TruncatedProviderStreamError,
    is_retryable_stream_error,
)


def _statusless_openai_api_error(
    message: str, body: object | None = None
) -> openai.APIError:
    return openai.APIError(
        message,
        request=httpx2.Request("POST", "https://provider.test/messages"),
        body=body,
    )


def test_retryable_stream_error_classifies_protocol_transport_and_status() -> None:
    assert is_retryable_stream_error(
        TruncatedProviderStreamError("missing terminal marker")
    )
    assert is_retryable_stream_error(httpx.ReadError("cut off"))

    request = httpx.Request("GET", "https://example.test")
    assert is_retryable_stream_error(
        httpx.HTTPStatusError(
            "server error", request=request, response=httpx.Response(503)
        )
    )
    assert not is_retryable_stream_error(
        httpx.HTTPStatusError(
            "bad request", request=request, response=httpx.Response(400)
        )
    )


def test_stream_retry_preserves_timeout_scope() -> None:
    request = httpx.Request("POST", "https://provider.test/messages")

    assert is_retryable_stream_error(httpx.ReadTimeout("read", request=request))
    assert not is_retryable_stream_error(
        httpx.ConnectTimeout("connect", request=request)
    )
    assert not is_retryable_stream_error(httpx.WriteTimeout("write", request=request))
    assert not is_retryable_stream_error(httpx.PoolTimeout("pool", request=request))


def test_retryable_stream_error_classifies_statusless_api_error_body_status() -> None:
    assert is_retryable_stream_error(
        _statusless_openai_api_error(
            "stream embedded error",
            {"error": {"message": "internal failure", "code": 500}},
        )
    )


def test_retryable_stream_error_classifies_statusless_internal_error_type() -> None:
    assert is_retryable_stream_error(
        _statusless_openai_api_error(
            "stream embedded error",
            {"error": {"message": "internal failure", "type": "internal_server_error"}},
        )
    )


def test_retryable_stream_error_classifies_resource_exhausted_text() -> None:
    assert is_retryable_stream_error(
        _statusless_openai_api_error(
            "ResourceExhausted: limit reached while generating response",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        )
    )


def test_retryable_stream_error_does_not_retry_bad_request_status() -> None:
    request = httpx2.Request("POST", "https://provider.test/messages")
    assert not is_retryable_stream_error(
        openai.BadRequestError(
            "bad request",
            response=httpx2.Response(400, request=request),
            body={"error": {"message": "bad request"}},
        )
    )


def test_early_retry_discards_uncommitted_holdback() -> None:
    controller = RecoveryController()

    assert controller.push("hidden") == []
    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable
    assert decision.has_buffered
    assert not controller.committed
    assert not controller.has_buffered
    assert controller.flush() == []


def test_early_retry_requires_remaining_execution_budget() -> None:
    controller = RecoveryController()
    assert controller.push("hidden") == []

    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable
    assert controller.has_buffered


def test_last_attempt_is_reserved_for_partial_output_recovery() -> None:
    controller = RecoveryController()
    assert controller.push("partial") == []

    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.has_buffered
    assert controller.has_buffered


def test_create_failure_is_owned_by_admission_not_stream_recovery() -> None:
    decision = RecoveryController().advance_failure(
        httpx.ConnectError("connect failed"),
        stream_opened=False,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable


def test_statusless_transient_api_error_allows_early_retry() -> None:
    decision = RecoveryController().advance_failure(
        _statusless_openai_api_error(
            "ResourceExhausted: limit reached while generating response",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable


def test_committed_output_allows_midstream_recovery() -> None:
    controller = RecoveryController()

    assert controller.push("event: content_block_delta\n\n") == []
    assert controller.flush() == ["event: content_block_delta\n\n"]
    decision = controller.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.retryable
    assert decision.committed
    assert controller.flush_uncommitted(decision) == []


def test_uncommitted_complete_tool_can_be_salvaged() -> None:
    controller = RecoveryController()

    assert controller.push("event: content_block_delta\n\n") == []
    decision = controller.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert not decision.committed
    assert decision.has_buffered
    assert controller.flush_uncommitted(decision) == ["event: content_block_delta\n\n"]
    assert controller.committed
    assert not controller.has_buffered


def test_non_retryable_error_is_final() -> None:
    request = httpx.Request("POST", "https://example.test/messages")
    error = httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=httpx.Response(400, request=request),
    )

    decision = RecoveryController().advance_failure(
        error,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert not decision.retryable


def test_holdback_buffers_until_delay_then_commits() -> None:
    now = [10.0]
    holdback = RecoveryHoldbackBuffer(holdback_seconds=0.75, now=lambda: now[0])

    assert holdback.push("event: content_block_start\n\n") == []
    now[0] += 0.74
    assert holdback.push("event: content_block_delta\n\n") == []
    assert not holdback.committed

    now[0] += 0.01
    assert holdback.push("event: content_block_stop\n\n") == [
        "event: content_block_start\n\n",
        "event: content_block_delta\n\n",
        "event: content_block_stop\n\n",
    ]
    assert holdback.committed
    assert holdback.push("event: message_stop\n\n") == ["event: message_stop\n\n"]


def test_holdback_flushes_at_internal_buffer_cap() -> None:
    holdback = RecoveryHoldbackBuffer(max_bytes=5, now=lambda: 1.0)

    assert holdback.push("ab") == []
    assert holdback.push("cde") == ["ab", "cde"]
    assert holdback.committed


def test_holdback_discard_drops_uncommitted_events() -> None:
    holdback = RecoveryHoldbackBuffer(now=lambda: 1.0)

    assert holdback.push("hidden") == []
    holdback.discard()

    assert holdback.flush() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [None, TimeoutError("upstream"), StopAsyncIteration()]
)
@pytest.mark.parametrize("expired_before_read", [False, True])
async def test_deadline_preserves_the_pending_read_and_its_outcome(
    monkeypatch, error, expired_before_read
):
    now = [0.0]
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(
            holdback_seconds=0.01,
            now=(lambda: now[0]) if expired_before_read else None,
        ),
    )
    recovery = RecoveryController()
    recovery.push("buffered")
    if expired_before_read:
        now[0] = 1.0
    release = asyncio.Event()
    reads = []

    class Source:
        async def __anext__(self):
            reads.append("read")
            await release.wait()
            if error is not None:
                raise error
            return "next"

        def __aiter__(self):
            return self

    async with recovery.read_stream(Source()) as stream:
        assert await asyncio.wait_for(anext(stream), 1) is HoldbackSignal.EXPIRED
        assert reads == ["read"]
        assert recovery.flush() == ["buffered"]
        release.set()
        if error is None:
            assert await anext(stream) == "next"
        else:
            with pytest.raises(type(error)) as caught:
                await anext(stream)
            assert caught.value is error
        assert reads == ["read"]


@pytest.mark.asyncio
@pytest.mark.parametrize("expired_before_read", [False, True])
async def test_events_without_output_cannot_extend_a_buffer_deadline(
    monkeypatch, expired_before_read
):
    now = [0.0]
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(now=lambda: now[0]),
    )
    recovery = RecoveryController()
    recovery.push("held")
    if expired_before_read:
        now[0] = 1.0
    reads = []

    async def source():
        reads.append("first")
        now[0] = 1.0
        yield "ignored"
        reads.append("second")
        yield "next"

    upstream = source()
    try:
        async with recovery.read_stream(upstream) as stream:
            assert await anext(stream) == "ignored"
            assert await anext(stream) is HoldbackSignal.EXPIRED
            assert reads == ["first"]
            assert recovery.flush() == ["held"]
            assert await anext(stream) == "next"
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_empty_buffer_does_not_wake_or_read_ahead():
    recovery = RecoveryController()
    entered = asyncio.Event()
    release = asyncio.Event()
    read_count = 0

    async def source():
        nonlocal read_count
        while True:
            read_count += 1
            entered.set()
            await release.wait()
            yield "next"

    upstream = source()
    read = None
    try:
        async with recovery.read_stream(upstream) as stream:
            read = asyncio.create_task(anext(stream))
            await asyncio.wait_for(entered.wait(), 1)
            assert not read.done()
            release.set()
            assert await read == "next"
            assert read_count == 1
    finally:
        release.set()
        if read is not None:
            read.cancel()
            await asyncio.gather(read, return_exceptions=True)
        await upstream.aclose()


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_read_before_leaving_scope():
    recovery = RecoveryController()
    recovery.push("held")
    entered = asyncio.Event()
    closing = asyncio.Event()
    close_release = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            closing.set()
            await close_release.wait()
            closed.set()

    async def consume():
        async with recovery.read_stream(source()) as stream:
            await anext(stream)

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        consumer.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        consumer.cancel()
        await asyncio.sleep(0)
        assert not consumer.done()
        assert not closed.is_set()
    finally:
        close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, 1)
    assert closed.is_set()

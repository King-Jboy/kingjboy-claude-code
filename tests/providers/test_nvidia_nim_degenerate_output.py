"""NVIDIA NIM degenerate "!!!!" output is retried before the client sees it."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.admission import UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
from free_claude_code.providers.nvidia_nim.degenerate_output import (
    NimDegenerateOutputError,
    guard_degenerate_output,
)
from tests.providers.request_factory import make_messages_request


def _chunk(
    *,
    reasoning: str | None = None,
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        role="assistant",
        reasoning_content=reasoning,
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=None,
    )


class _Stream:
    """Upstream stream double that records whether it was closed."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


def _garbage() -> list[SimpleNamespace]:
    # Recorded from moonshotai/kimi-k3: one word, one "!" per chunk, blank answer.
    return [
        _chunk(reasoning="The"),
        *[_chunk(reasoning="!") for _ in range(32)],
        _chunk(content=" ", finish_reason="stop"),
    ]


def _normal() -> list[SimpleNamespace]:
    return [
        _chunk(reasoning="The"),
        _chunk(reasoning=" user"),
        _chunk(reasoning=" wants"),
        _chunk(reasoning=" pong."),
        _chunk(content="pong", finish_reason="stop"),
    ]


async def _drain(stream) -> list[SimpleNamespace]:
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_guard_passes_normal_output_through_unchanged():
    chunks = _normal()

    assert await _drain(guard_degenerate_output(_Stream(chunks))) == chunks


@pytest.mark.asyncio
async def test_guard_passes_a_short_complete_answer_through():
    chunks = [_chunk(content="p"), _chunk(content="ong", finish_reason="stop")]

    assert await _drain(guard_degenerate_output(_Stream(chunks))) == chunks


@pytest.mark.asyncio
async def test_guard_releases_exclamations_that_are_not_a_run():
    chunks = [
        _chunk(content="Hi!! "),
        _chunk(content="Great to see you again, here is the plan for today."),
        _chunk(content=" Done.", finish_reason="stop"),
    ]

    assert await _drain(guard_degenerate_output(_Stream(chunks))) == chunks


@pytest.mark.asyncio
async def test_guard_releases_tool_calls_immediately():
    tool_call = SimpleNamespace(index=0, id="call_1", function=None)
    chunks = [_chunk(tool_calls=[tool_call]), _chunk(finish_reason="tool_calls")]

    assert await _drain(guard_degenerate_output(_Stream(chunks))) == chunks


@pytest.mark.asyncio
async def test_guard_rejects_a_run_of_exclamation_tokens_before_releasing_any():
    guard = guard_degenerate_output(_Stream(_garbage()))

    # The very first read fails, so no garbage chunk is ever released.
    with pytest.raises(NimDegenerateOutputError):
        await anext(guard)


@pytest.mark.asyncio
async def test_guard_close_closes_the_upstream_stream_even_unstarted():
    upstream = _Stream(_normal())

    await guard_degenerate_output(upstream).aclose()

    assert upstream.closed


def _stream_of(chunks: list[SimpleNamespace]):
    async def gen():
        for chunk in chunks:
            yield chunk

    return gen()


def _answer_text(events: list[str]) -> str:
    text = ""
    for event in events:
        for line in event.splitlines():
            if line.startswith("data: ") and '"text_delta"' in line:
                text += json.loads(line[6:])["delta"]["text"]
    return text


@pytest.mark.asyncio
async def test_degenerate_output_is_retried_invisibly(nim_provider):
    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [_stream_of(_garbage()), _stream_of(_normal())]
        events = [
            e
            async for e in nim_provider.stream_response(
                make_messages_request(model="moonshotai/kimi-k3")
            )
        ]

    assert mock_create.await_count == 2
    assert _answer_text(events) == "pong"
    assert not any("!!" in event for event in events)


@pytest.mark.asyncio
async def test_degenerate_output_on_every_attempt_fails_cleanly(nim_provider):
    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [
            _stream_of(_garbage()) for _ in range(UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS)
        ]
        with pytest.raises(ExecutionFailure) as exc_info:
            _ = [
                e
                async for e in nim_provider.stream_response(
                    make_messages_request(model="moonshotai/kimi-k3")
                )
            ]

    assert mock_create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert "degenerate output" in exc_info.value.message
    assert exc_info.value.retryable is False

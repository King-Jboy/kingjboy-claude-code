"""Folding an Anthropic Messages SSE stream into one JSON Message body."""

from collections.abc import AsyncIterator

import pytest

from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.streaming import format_sse_event


def _body(events: list[str]) -> AsyncIterator[str]:
    async def generate() -> AsyncIterator[str]:
        for event in events:
            yield event

    return generate()


def _message_start() -> str:
    return format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_agg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test-model",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )


def _stop_events() -> list[str]:
    return [
        format_sse_event("message_delta", {"type": "message_delta", "delta": {}}),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


@pytest.mark.asyncio
async def test_a_signature_split_across_deltas_is_concatenated() -> None:
    # Every other delta type accumulates fragment by fragment; a signature
    # split across chunks must reassemble the same way instead of keeping
    # only the last fragment.
    events = [
        _message_start(),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig-part-one."},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig-part-two."},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        *_stop_events(),
    ]

    message, error = await aggregate_anthropic_sse_to_message(_body(events))

    assert error is None
    assert message["content"][0]["signature"] == "sig-part-one.sig-part-two."


@pytest.mark.asyncio
async def test_trailing_buffer_without_final_double_newline_is_aggregated() -> None:
    # A stream that terminates without a trailing double newline on its final
    # event must still aggregate the last event instead of dropping it.
    last_event = format_sse_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    ).rstrip("\n")
    events = [
        _message_start(),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello"},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        last_event,
    ]

    message, error = await aggregate_anthropic_sse_to_message(_body(events))
    assert error is None
    assert message["content"][0]["text"] == "Hello"
    assert message["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_crlf_delimited_sse_events_are_aggregated() -> None:
    # Servers or proxies emitting \r\n\r\n framing must be parsed identically to \n\n.
    events = [
        _message_start().replace("\n\n", "\r\n\r\n"),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "CRLF World"},
            },
        ).replace("\n\n", "\r\n\r\n"),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ).replace("\n\n", "\r\n\r\n"),
        *[e.replace("\n\n", "\r\n\r\n") for e in _stop_events()],
    ]

    message, error = await aggregate_anthropic_sse_to_message(_body(events))
    assert error is None
    assert message["content"][0]["text"] == "CRLF World"

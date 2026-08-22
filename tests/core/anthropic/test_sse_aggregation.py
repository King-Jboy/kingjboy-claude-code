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

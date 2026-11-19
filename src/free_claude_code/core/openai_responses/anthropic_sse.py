"""Anthropic SSE parsing used by the Responses stream adapter."""

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from free_claude_code.core.anthropic.sse_parse import (
    AnthropicSseEvent,
    parse_sse_event,
)
from free_claude_code.core.anthropic.sse_parse import (
    iter_sse_events as _iter_sse_events,
)

from .errors import ResponsesConversionError

__all__ = ["AnthropicSseEvent", "iter_sse_events", "parse_sse_event"]


def iter_sse_events(
    chunks: AsyncIterable[Any],
) -> AsyncGenerator[AnthropicSseEvent]:
    """Yield parsed Anthropic SSE frames, raising ResponsesConversionError on overflow."""
    return _iter_sse_events(
        chunks,
        owner="openai_responses.anthropic_sse",
        overflow_error_factory=ResponsesConversionError,
    )

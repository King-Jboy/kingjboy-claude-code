"""Trace-safe snapshots of Anthropic protocol requests."""

from typing import Any

from free_claude_code.core.trace import sanitize_trace_value

from .models import MessagesRequest, TokenCountRequest


def anthropic_request_snapshot(
    request: MessagesRequest | TokenCountRequest,
) -> dict[str, Any]:
    """Return the traceable public fields of an Anthropic request."""
    snapshot: dict[str, Any] = {
        "model": request.model,
        "message_count": len(request.messages),
    }
    if isinstance(request, MessagesRequest):
        if request.max_tokens is not None:
            snapshot["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            snapshot["temperature"] = request.temperature
        if request.top_p is not None:
            snapshot["top_p"] = request.top_p
        if request.top_k is not None:
            snapshot["top_k"] = request.top_k
        if request.stop_sequences is not None:
            snapshot["stop_sequences"] = request.stop_sequences
        if request.metadata is not None:
            snapshot["metadata"] = request.metadata
        if request.stream:
            snapshot["stream"] = request.stream
        if request.tool_choice is not None:
            snapshot["tool_choice"] = (
                request.tool_choice.model_dump(exclude_none=True)
                if hasattr(request.tool_choice, "model_dump")
                else request.tool_choice
            )
    if request.tools:
        snapshot["tool_count"] = len(request.tools)
    if request.thinking is not None:
        snapshot["thinking"] = (
            request.thinking.model_dump(exclude_none=True)
            if hasattr(request.thinking, "model_dump")
            else str(request.thinking)
        )
    sanitized = sanitize_trace_value(snapshot)
    return sanitized if isinstance(sanitized, dict) else {}

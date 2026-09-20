"""OpenAI Responses SSE event formatting."""

from collections.abc import Mapping
from typing import Any

from free_claude_code.core.json_utils import fast_json_dumps

OPENAI_RESPONSES_SSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def format_response_sse_event(event_type: str, data: Mapping[str, Any]) -> str:
    """Format one OpenAI Responses SSE event."""

    return f"event: {event_type}\ndata: {fast_json_dumps(data)}\n\n"

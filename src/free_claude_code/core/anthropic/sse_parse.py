"""Fast Anthropic SSE frame parser shared by OpenAI wire adapters."""

import codecs
import re
import sys
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from dataclasses import dataclass
from typing import Any

from free_claude_code.core.json_utils import fast_json_loads
from free_claude_code.core.trace import close_stream_input

_SSE_EVENT_BOUNDARY = re.compile(r"\r?\n\r?\n")
_MAX_SSE_BUFFER_LEN = 16 * 1024 * 1024  # 16 MB


@dataclass(slots=True)
class AnthropicSseEvent:
    """One parsed Anthropic SSE frame."""

    event: str
    data: dict[str, Any]


async def iter_sse_events(
    chunks: AsyncIterable[Any],
    *,
    owner: str = "anthropic.sse_parse",
    overflow_error_factory: Callable[[str], Exception] = ValueError,
) -> AsyncGenerator[AnthropicSseEvent]:
    """Yield parsed Anthropic SSE frames from an async stream of chunks.

    Fast-paths the common single-frame ``str`` chunks emitted by
    ``AnthropicStreamLedger`` without regex scanning or buffer copying.
    """
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    iterator = aiter(chunks)
    try:
        async for chunk in iterator:
            if (
                not buffer
                and isinstance(chunk, str)
                and chunk.endswith("\n\n")
                and "\r" not in chunk
            ):
                sep_idx = chunk.find("\n\n")
                if sep_idx == len(chunk) - 2:
                    event = parse_sse_event(chunk[:-2])
                    if event is not None:
                        yield event
                    continue

            if isinstance(chunk, bytes):
                buffer += decoder.decode(chunk, final=False)
            else:
                buffer += str(chunk)

            if len(buffer) > _MAX_SSE_BUFFER_LEN:
                raise overflow_error_factory(
                    "SSE stream event exceeded maximum buffer capacity"
                )

            while (boundary := _SSE_EVENT_BOUNDARY.search(buffer)) is not None:
                raw = buffer[: boundary.start()]
                buffer = buffer[boundary.end() :]
                event = parse_sse_event(raw)
                if event is not None:
                    yield event

        buffer += decoder.decode(b"", final=True)
        if buffer.strip():
            event = parse_sse_event(buffer)
            if event is not None:
                yield event
    finally:
        await close_stream_input(
            iterator,
            owner=owner,
            source="core",
            preserved_error=sys.exception(),
        )


def parse_sse_event(raw: str) -> AnthropicSseEvent | None:
    """Parse one raw SSE frame into ``AnthropicSseEvent``."""
    # Fast path for standard two-line ``event: <name>\ndata: <json>`` frames.
    if raw.startswith("event:") and "\r" not in raw:
        nl = raw.find("\n")
        if nl != -1:
            rest = raw[nl + 1 :]
            if rest.startswith("data:") and "\n" not in rest:
                event_type = raw[6:nl].strip()
                data_text = rest[5:].strip()
                if data_text == "[DONE]":
                    return None
                return _build_sse_event(event_type, data_text)

    event_type = ""
    data_parts: list[str] = []
    for line in raw.splitlines():
        stripped = line.rstrip("\r")
        if stripped.startswith("event:"):
            event_type = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("data:"):
            data_parts.append(stripped.split(":", 1)[1].strip())
    if not event_type and not data_parts:
        return None
    data_text = "\n".join(data_parts)
    if data_text == "[DONE]":
        return None
    return _build_sse_event(event_type, data_text)


def _build_sse_event(event_type: str, data_text: str) -> AnthropicSseEvent:
    try:
        parsed = fast_json_loads(data_text) if data_text else {}
    except Exception:
        parsed = {"raw": data_text}
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    return AnthropicSseEvent(event=event_type, data=parsed)

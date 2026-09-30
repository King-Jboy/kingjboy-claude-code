"""Catch NVIDIA NIM's degenerate "!!!!" output before any of it is shown."""

from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from free_claude_code.providers.failure_policy import RetryableProviderProtocolError
from free_claude_code.providers.http import maybe_await_aclose

# Some NIM deployments intermittently stream one word and then token 0 ("!")
# over and over, ending with a blank answer. Measured on moonshotai/kimi-k3:
# 3 of 16 identical requests.
_DEGENERATE_RUN = "!" * 8
# Output is released once this many characters arrive without "!!".
_CLEAR_AFTER_CHARS = 12
# Output that has "!!" but no degenerate run is released at this length.
_RELEASE_AFTER_CHARS = 64


class NimDegenerateOutputError(RetryableProviderProtocolError):
    """NIM streamed a run of "!" tokens instead of real output."""


def _chunk_text(chunk: Any) -> tuple[str, bool]:
    """Return one chunk's generated text and whether it carries tool calls."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return "", False
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return "", False
    text = "".join(
        value
        for name in ("reasoning_content", "reasoning", "content")
        if isinstance(value := getattr(delta, name, None), str)
    )
    tool_calls = getattr(delta, "tool_calls", None)
    return text, isinstance(tool_calls, list) and bool(tool_calls)


class _DegenerateOutputGuard(AsyncIterator[Any]):
    """Withhold a stream's opening chunks until they are known not to be garbage."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._iterator = aiter(stream)
        self._held: deque[Any] = deque()
        self._text = ""
        self._decided = False
        self._exhausted = False

    def __aiter__(self) -> _DegenerateOutputGuard:
        return self

    async def __anext__(self) -> Any:
        while not self._decided:
            try:
                chunk = await anext(self._iterator)
            except StopAsyncIteration:
                self._exhausted = True
                self._decide(stream_ended=True, tool_calls=False)
                break
            self._held.append(chunk)
            text, tool_calls = _chunk_text(chunk)
            self._text += text
            self._decide(stream_ended=False, tool_calls=tool_calls)
        if self._held:
            return self._held.popleft()
        if self._exhausted:
            raise StopAsyncIteration
        return await anext(self._iterator)

    def _decide(self, *, stream_ended: bool, tool_calls: bool) -> None:
        if _DEGENERATE_RUN in self._text:
            self._held.clear()
            raise NimDegenerateOutputError(
                "NVIDIA NIM returned degenerate output (a run of '!' tokens)."
            )
        self._decided = (
            stream_ended
            or tool_calls
            or len(self._text) >= _RELEASE_AFTER_CHARS
            or (len(self._text) >= _CLEAR_AFTER_CHARS and "!!" not in self._text)
        )

    async def aclose(self) -> None:
        await maybe_await_aclose(self._stream)


def guard_degenerate_output(stream: Any) -> _DegenerateOutputGuard:
    """Raise a retryable error when NIM opens with a run of "!" tokens."""
    return _DegenerateOutputGuard(stream)

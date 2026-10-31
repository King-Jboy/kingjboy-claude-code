"""Adapt OpenAI SDK response streams to FCC's cleanup contract."""

from collections import deque
from collections.abc import AsyncIterable, AsyncIterator

from free_claude_code.providers.http import maybe_await_aclose


def _chunk_has_meaningful_delta(chunk: object) -> bool:
    """Return whether an upstream SSE chunk carries actual token or finish progress."""
    choices = getattr(chunk, "choices", None)
    if choices is None:
        return True
    if not choices:
        return getattr(chunk, "usage", None) is not None
    choice = choices[0]
    if getattr(choice, "finish_reason", None):
        return True
    delta = getattr(choice, "delta", None)
    if delta is None:
        return False
    return bool(
        getattr(delta, "content", None)
        or getattr(delta, "reasoning_content", None)
        or getattr(delta, "reasoning", None)
        or getattr(delta, "tool_calls", None)
        or getattr(delta, "reasoning_details", None)
    )


class OpenAIStreamAdapter[EventT](AsyncIterator[EventT]):
    """Close one SDK response stream without closing its reusable client."""

    def __init__(self, stream: AsyncIterable[EventT]) -> None:
        self._stream = stream
        self._iterator = aiter(stream)
        self._primed: deque[EventT] = deque()
        self._exhausted = False

    def __aiter__(self) -> OpenAIStreamAdapter[EventT]:
        return self

    async def prime_first_chunk(self, max_prologue_chunks: int = 8) -> None:
        """Buffer initial chunks until the first meaningful token/delta arrives."""
        for _ in range(max_prologue_chunks):
            try:
                chunk = await anext(self._iterator)
            except StopAsyncIteration:
                self._exhausted = True
                return
            self._primed.append(chunk)
            if _chunk_has_meaningful_delta(chunk):
                return

    async def __anext__(self) -> EventT:
        if self._primed:
            return self._primed.popleft()
        if self._exhausted:
            raise StopAsyncIteration
        return await anext(self._iterator)

    async def aclose(self) -> None:
        self._primed.clear()
        await maybe_await_aclose(self._stream)

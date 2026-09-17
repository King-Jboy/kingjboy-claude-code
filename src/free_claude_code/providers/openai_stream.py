"""Adapt OpenAI SDK response streams to FCC's cleanup contract."""

from collections.abc import AsyncIterable, AsyncIterator

from free_claude_code.providers.http import maybe_await_aclose


class OpenAIStreamAdapter[EventT](AsyncIterator[EventT]):
    """Close one SDK response stream without closing its reusable client."""

    def __init__(self, stream: AsyncIterable[EventT]) -> None:
        self._stream = stream
        self._iterator = aiter(stream)

    def __aiter__(self) -> OpenAIStreamAdapter[EventT]:
        return self

    async def __anext__(self) -> EventT:
        return await anext(self._iterator)

    async def aclose(self) -> None:
        await maybe_await_aclose(self._stream)

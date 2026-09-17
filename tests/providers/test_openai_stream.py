"""OpenAI SDK stream ownership contracts."""

from collections.abc import AsyncIterator, Sequence

import pytest

from free_claude_code.providers.openai_stream import OpenAIStreamAdapter


class _SdkStream(AsyncIterator[str]):
    """Minimal shape exposed by the OpenAI SDK for a streaming response."""

    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = iter(chunks)
        self.close_calls = 0

    def __aiter__(self) -> _SdkStream:
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def close(self) -> None:
        self.close_calls += 1


class _LegacyStream:
    """Existing FCC test-double shape: iterable with an ``aclose`` method."""

    def __init__(self, chunks: Sequence[str]) -> None:
        self._chunks = chunks
        self.close_calls = 0

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_openai_stream_adapter_iterates_and_closes_the_sdk_stream() -> None:
    sdk_stream = _SdkStream(("first", "second"))
    stream = OpenAIStreamAdapter(sdk_stream)

    assert [chunk async for chunk in stream] == ["first", "second"]

    await stream.aclose()

    assert sdk_stream.close_calls == 1


@pytest.mark.asyncio
async def test_openai_stream_adapter_preserves_legacy_stream_cleanup() -> None:
    source = _LegacyStream(("first", "second"))
    stream = OpenAIStreamAdapter(source)

    assert [chunk async for chunk in stream] == ["first", "second"]

    await stream.aclose()

    assert source.close_calls == 1

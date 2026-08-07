"""Keepalives for an upstream that has gone quiet mid-request.

Claude Code aborts a pending response stream that delivers no data for 20
seconds and shows "Waiting for API response - will retry in ...". A reasoning
model regularly thinks for longer than that before its first token, so without
these frames ordinary upstream behaviour reads to the client as a dead
connection.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from free_claude_code.providers.openai_chat.provider import (
    _KEEPALIVE,
    _chunks_with_keepalive,
)


async def _quiet_then(*, silence: float, chunks: tuple[str, ...]) -> AsyncIterator[str]:
    """An upstream that says nothing for ``silence`` seconds, then speaks."""
    await asyncio.sleep(silence)
    for chunk in chunks:
        yield chunk


async def _collect(stream, *, quiet_after: float, interval: float) -> list[object]:
    return [
        item
        async for item in _chunks_with_keepalive(
            stream, quiet_after=quiet_after, interval=interval
        )
    ]


@pytest.mark.asyncio
async def test_a_quiet_upstream_produces_keepalives_before_its_first_chunk() -> None:
    items = await _collect(
        _quiet_then(silence=0.25, chunks=("a", "b")),
        quiet_after=0.05,
        interval=0.02,
    )

    assert _KEEPALIVE in items, "a long silence must not reach the client as nothing"
    assert [item for item in items if item is not _KEEPALIVE] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_prompt_upstream_produces_no_keepalives() -> None:
    # Committing early costs the typed non-2xx path, so a response that never
    # stalls must not pay for a problem it does not have.
    items = await _collect(
        _quiet_then(silence=0.0, chunks=("a", "b")),
        quiet_after=5.0,
        interval=0.02,
    )

    assert items == ["a", "b"]


@pytest.mark.asyncio
async def test_the_quiet_timer_resets_between_chunks() -> None:
    # Otherwise a stream that is merely slow overall, rather than stalled, would
    # accumulate its way into a keepalive on every subsequent chunk.
    async def steady() -> AsyncIterator[str]:
        for chunk in ("a", "b", "c"):
            await asyncio.sleep(0.03)
            yield chunk

    items = await _collect(steady(), quiet_after=0.08, interval=0.02)

    assert items == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_an_upstream_failure_still_propagates() -> None:
    # The keepalive wrapper sits in the failure path; swallowing an error here
    # would strand the request instead of letting recovery see it.
    async def fails() -> AsyncIterator[str]:
        yield "a"
        raise RuntimeError("upstream died")

    with pytest.raises(RuntimeError, match="upstream died"):
        await _collect(fails(), quiet_after=0.05, interval=0.02)

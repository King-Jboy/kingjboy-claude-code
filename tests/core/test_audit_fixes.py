"""Regression tests for concurrency, persistence, parser, and hot-path fixes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import free_claude_code.providers.key_pool as key_pool_module
from free_claude_code.config.context_windows import load_context_window_entries
from free_claude_code.core.anthropic.tools import HeuristicToolParser
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.state import ResponsesStore
from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.core.trace import traced_async_stream
from free_claude_code.providers.admission import _ConcurrencyBulkhead
from free_claude_code.providers.key_pool import ApiKeyPool


@pytest.mark.asyncio
async def test_sliding_window_wakes_immediately_on_capacity_increase_and_respects_cancel() -> (
    None
):
    limiter = StrictSlidingWindowLimiter(rate_limit=1, rate_window=30.0)
    await limiter.acquire()

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0.02)
    assert not waiter.done()

    # Raising capacity wakes the sleeping waiter without waiting for the 30s window.
    limiter.set_rate_limit(2)
    await asyncio.wait_for(waiter, timeout=0.5)

    # Outer cancellation while blocked on full window propagates CancelledError cleanly.
    cancelled_waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0.02)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter


@pytest.mark.asyncio
async def test_api_key_pool_notifies_capacity_listener_on_cooldown_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(key_pool_module, "_RATE_LIMIT_COOLDOWN_SECONDS", 0.05)
    observed_limits: list[int] = []
    pool = ApiKeyPool(
        ("key-a", "key-b"),
        rate_limit=10,
        rate_window=60.0,
        on_capacity_change=observed_limits.append,
    )
    try:
        pool.mark_rate_limited("key-a")
        assert observed_limits == [10]
        await asyncio.sleep(0.12)
        assert observed_limits[-1] == 20
    finally:
        pool.close()


@pytest.mark.asyncio
async def test_concurrency_bulkhead_try_acquire_and_waiter_wakeup_invariants() -> None:
    bulkhead = _ConcurrencyBulkhead(1)
    assert bulkhead.try_acquire() is True
    assert bulkhead.try_acquire() is False

    # Normal waiter blocks until the slot releases, and a cancelled waiter
    # hands off its wakeup cleanly to the next waiter.
    first_waiter = asyncio.create_task(bulkhead.acquire())
    second_waiter = asyncio.create_task(bulkhead.acquire())
    await asyncio.sleep(0.02)
    assert not first_waiter.done()
    assert not second_waiter.done()

    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    bulkhead.release()
    await asyncio.wait_for(second_waiter, timeout=0.5)
    bulkhead.release()
    assert bulkhead.try_acquire() is True
    bulkhead.release()


def test_responses_store_cached_serialization_roundtrips(tmp_path: Path) -> None:
    state_path = tmp_path / "responses.json"
    store = ResponsesStore(path=state_path)
    req1 = OpenAIResponsesRequest(model="m1", input="hello")
    store.record(
        req1,
        {
            "id": "resp_1",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": "hi"}],
        },
    )
    req2 = OpenAIResponsesRequest(
        model="m1",
        input="follow up",
        previous_response_id="resp_1",
    )
    store.record(
        req2,
        {
            "id": "resp_2",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": "done"}],
        },
    )

    reloaded = ResponsesStore(path=state_path)
    resolved = reloaded.resolve(
        OpenAIResponsesRequest(
            model="m1",
            input="third",
            previous_response_id="resp_2",
        )
    )
    assert resolved.previous_response_id is None
    assert isinstance(resolved.input, list)
    assert len(resolved.input) >= 3


def test_heuristic_tool_parser_scopes_web_tools_to_declared_tools() -> None:
    prose = 'Use WebSearch {"query": "python asyncio"} to verify.'

    unscoped = HeuristicToolParser()
    text_unscoped, tools_unscoped = unscoped.feed(prose)
    assert len(tools_unscoped) == 1
    assert tools_unscoped[0]["name"] == "WebSearch"
    assert "WebSearch" not in text_unscoped

    scoped = HeuristicToolParser(allowed_web_tools=frozenset({"Read", "Bash"}))
    text_scoped, tools_scoped = scoped.feed(prose)
    assert tools_scoped == []
    assert text_scoped == prose


def test_context_windows_cache_invalidates_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "context.md"
    path.write_text(
        "## openai\n| Model | Context | Source |\n| --- | --- | --- |\n| `gpt-4o` | 128000 | manual |\n",
        encoding="utf-8",
    )

    first = load_context_window_entries(path)
    second = load_context_window_entries(path)
    assert first == second
    assert first["openai/gpt-4o"].context == 128000

    # Ensure size and/or mtime changes invalidate the cached entries.
    time.sleep(0.01)
    path.write_text(
        "## openai\n| Model | Context | Source |\n| --- | --- | --- |\n| `gpt-4o` | 200000 | manual |\n",
        encoding="utf-8",
    )
    updated = load_context_window_entries(path)
    assert updated["openai/gpt-4o"].context == 200000


@pytest.mark.asyncio
async def test_traced_async_stream_returns_original_iterator_when_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import free_claude_code.core.trace as trace_module

    monkeypatch.setattr(trace_module, "_debug_trace_enabled", lambda: False)

    async def sample() -> AsyncIterator[str]:
        yield "ok"

    gen = sample()
    wrapped = traced_async_stream(
        gen,
        stage="egress",
        source="api",
        complete_event="complete",
        interrupted_event="interrupted",
    )
    assert wrapped is gen
    items = [item async for item in wrapped]
    assert items == ["ok"]

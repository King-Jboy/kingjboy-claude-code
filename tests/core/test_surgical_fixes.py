import time
from typing import cast

import httpx
import pytest

from free_claude_code.core.openai_responses.tools import parse_arguments
from free_claude_code.core.trace import sanitize_trace_value
from free_claude_code.providers.failure_policy import _reports_context_window_exceeded
from free_claude_code.providers.key_pool import ApiKeyInfo
from free_claude_code.providers.openai_codex.provider import _iter_sse


def test_parse_arguments_null():
    assert parse_arguments("null") == {}
    assert parse_arguments(" null ") == {}
    assert parse_arguments(None) == {}
    assert parse_arguments("") == {}
    assert parse_arguments('{"key": "value"}') == {"key": "value"}


def test_sanitize_trace_value_recursion_and_cycles():
    d = {}
    d["self"] = d
    res = sanitize_trace_value(d)
    assert res == {"self": "<cycle>"}

    curr = {}
    root = curr
    for _ in range(25):
        curr["child"] = {}
        curr = curr["child"]
    res_deep = sanitize_trace_value(root)
    assert "<truncated>" in str(res_deep)


def test_reports_context_window_exceeded():
    class CustomErr(Exception):
        pass

    err1 = CustomErr("Error: maximum context tokens exceeded")
    assert _reports_context_window_exceeded(err1) is True

    err2 = CustomErr("Your request exceeded the context length of 128000")
    assert _reports_context_window_exceeded(err2) is True

    err_unrelated = CustomErr("Something went wrong with connection")
    assert _reports_context_window_exceeded(err_unrelated) is False


def test_key_pool_mark_rate_limited_monotonic():
    ki = ApiKeyInfo(key="test-key-12345678")
    now = time.monotonic()
    ki.mark_rate_limited(cooldown_seconds=60.0)
    first_until = ki.rate_limited_until
    assert first_until >= now + 59.0

    ki.mark_rate_limited(cooldown_seconds=10.0)
    assert ki.rate_limited_until >= first_until


@pytest.mark.asyncio
async def test_codex_iter_sse_whitespace_and_empty():
    lines = [
        "event: response.output_item.added",
        'data: {"type": "response.output_item.added", "item": {"id": "1"}}',
        "",
        "data: ",
        "",
        'data:  {"type": "response.content_part.added"}',
        "",
        "data: [DONE]",
        "",
    ]

    class MockResponse:
        async def aiter_lines(self):
            for line in lines:
                yield line

    events = []
    async for ev_type, payload in _iter_sse(cast(httpx.Response, MockResponse())):
        events.append((ev_type, payload))

    assert len(events) == 2
    assert events[0][0] == "response.output_item.added"
    assert events[1][0] == "response.content_part.added"

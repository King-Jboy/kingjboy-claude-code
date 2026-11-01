import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.providers.openai_chat.stream_create import create_chat_stream

SSE = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    "data: [DONE]\n\n"
)

BODY: dict[str, Any] = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "héllo"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "print(1)"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
    "max_tokens": 64,
    "stream_options": {"include_usage": True},
    "extra_body": {"chat_template_kwargs": {"thinking": False}, "top_k": 20},
}


def _client(captured: list[httpx2.Request]) -> AsyncOpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=SSE
        )

    return AsyncOpenAI(
        api_key="secret",
        base_url="https://provider.invalid/v1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


async def _chunks(stream) -> list[str]:
    return [chunk.choices[0].delta.content async for chunk in stream]


@pytest.mark.asyncio
async def test_fast_path_sends_the_same_request_as_sdk_create() -> None:
    fast: list[httpx2.Request] = []
    sdk: list[httpx2.Request] = []
    fast_client, sdk_client = _client(fast), _client(sdk)
    try:
        assert await _chunks(await create_chat_stream(fast_client, BODY)) == ["hi"]
        assert await _chunks(
            await sdk_client.chat.completions.create(**BODY, stream=True)
        ) == ["hi"]
    finally:
        await fast_client.close()
        await sdk_client.close()

    (fast_request,) = fast
    (sdk_request,) = sdk
    assert fast_request.url == sdk_request.url
    assert json.loads(fast_request.content) == json.loads(sdk_request.content)
    for header in ("authorization", "content-type", "accept"):
        assert fast_request.headers[header] == sdk_request.headers[header]
    assert fast_request.headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_extra_body_keys_override_body_fields_like_sdk_create() -> None:
    captured: list[httpx2.Request] = []
    client = _client(captured)
    try:
        stream = await create_chat_stream(
            client, {**BODY, "extra_body": {"max_tokens": 9}}
        )
        await stream.close()
    finally:
        await client.close()

    assert json.loads(captured[0].content)["max_tokens"] == 9
    assert json.loads(captured[0].content)["stream"] is True


@pytest.mark.asyncio
async def test_a_patched_create_still_receives_the_request() -> None:
    client = _client([])
    sentinel = object()
    try:
        with patch.object(
            client.chat.completions, "create", AsyncMock(return_value=sentinel)
        ) as create:
            assert await create_chat_stream(client, BODY) is sentinel
    finally:
        await client.close()

    create.assert_awaited_once_with(**BODY, stream=True)


@pytest.mark.asyncio
async def test_non_sdk_clients_and_request_options_use_create() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value="stream")
    assert await create_chat_stream(mock_client, BODY) == "stream"

    captured: list[httpx2.Request] = []
    client = _client(captured)
    try:
        with_headers = {**BODY, "extra_headers": {"X-Trace": "1"}}
        await (await create_chat_stream(client, with_headers)).close()
        # orjson rejects non-string keys; create()'s encoder stringifies them.
        int_keys = {**BODY, "extra_body": {"logit_bias": {50256: -100}}}
        await (await create_chat_stream(client, int_keys)).close()
    finally:
        await client.close()

    assert captured[0].headers["x-trace"] == "1"
    assert json.loads(captured[1].content)["logit_bias"] == {"50256": -100}

"""Tests for POST /v1/chat/completions (direct passthrough & universal fallback)."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx2
from fastapi.testclient import TestClient

from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.openai_responses import (
    OpenAIResponsesAdapter,
    OpenAIResponsesRequest,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.deepseek.client import DeepSeekProvider
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission


class _AnthropicOnlyFakeProvider:
    """Fake non-Chat provider (like Bedrock/Codex/Gemini) emitting Anthropic SSE."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.preflight_stream = MagicMock()
        self.requests: list[Any] = []

    async def stream_response(self, request_data, **_kwargs):
        self.requests.append(request_data)
        for chunk in self.chunks:
            yield chunk


def _parse_chat_sse(text: str) -> tuple[list[dict[str, Any]], bool]:
    frames: list[dict[str, Any]] = []
    saw_done = False
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        frames.append(json.loads(payload))
    return frames, saw_done


def _anthropic_thinking_and_tool_stream() -> list[str]:
    return [
        format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "test-model",
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "step by step"},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Checking weather."},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 1}
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_weather_1",
                    "name": "get_weather",
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city":"Paris"}',
                },
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 2}
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 17},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _make_mock_deepseek_provider(
    captured_bodies: list[dict[str, Any]],
    sse_payload: bytes,
    *,
    status_code: int = 200,
) -> DeepSeekProvider:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured_bodies.append(json.loads(request.content))
        if status_code != 200:
            return httpx2.Response(
                status_code,
                json={"error": {"message": "upstream rejected"}},
                request=request,
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_payload,
            request=request,
        )

    provider = DeepSeekProvider(
        ProviderConfig(api_key="test-ds-key", base_url="https://api.deepseek.test/v1"),
        admission=immediate_admission(),
    )
    provider._client = provider._client.copy(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    return provider


def test_chat_completions_direct_passthrough_streaming_and_non_streaming() -> None:
    captured: list[dict[str, Any]] = []
    upstream_sse = (
        b'data: {"id":"ds-1","object":"chat.completion.chunk","created":1700000000,'
        b'"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"reasoning_content":"pondering"},"finish_reason":null}]}\n\n'
        b'data: {"id":"ds-1","object":"chat.completion.chunk","created":1700000000,'
        b'"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"hello world"},'
        b'"finish_reason":null}]}\n\n'
        b'data: {"id":"ds-1","object":"chat.completion.chunk","created":1700000000,'
        b'"model":"deepseek-reasoner","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}\n\n'
        b"data: [DONE]\n\n"
    )
    provider = _make_mock_deepseek_provider(captured, upstream_sse)
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        # Streaming request
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-reasoner",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        frames, saw_done = _parse_chat_sse(res.text)
        assert saw_done is True
        assert len(frames) == 3
        assert frames[0]["model"] == "deepseek/deepseek-reasoner"
        assert frames[0]["choices"][0]["delta"]["reasoning_content"] == "pondering"
        assert frames[1]["choices"][0]["delta"]["content"] == "hello world"
        assert frames[2]["choices"][0]["finish_reason"] == "stop"
        assert captured[0]["model"] == "deepseek-reasoner"
        assert captured[0]["stream"] is True

        # Non-streaming request
        res_json = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-reasoner",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert res_json.status_code == 200
        body = res_json.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "deepseek/deepseek-reasoner"
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "hello world"
        assert msg["reasoning_content"] == "pondering"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        }


def test_chat_completions_fallback_translates_anthropic_provider_streams() -> None:
    provider = _AnthropicOnlyFakeProvider(_anthropic_thinking_and_tool_stream())
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Weather in Paris?"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert res.status_code == 200
        frames, saw_done = _parse_chat_sse(res.text)
        assert saw_done is True
        joined_reasoning = "".join(
            f["choices"][0]["delta"].get("reasoning_content", "")
            for f in frames
            if f.get("choices")
        )
        joined_text = "".join(
            f["choices"][0]["delta"].get("content", "")
            for f in frames
            if f.get("choices")
        )
        tool_chunks = [
            tc
            for f in frames
            if f.get("choices")
            for tc in f["choices"][0]["delta"].get("tool_calls", [])
        ]
        assert joined_reasoning == "step by step"
        assert joined_text == "Checking weather."
        assert tool_chunks[0]["id"] == "call_weather_1"
        assert tool_chunks[0]["function"]["name"] == "get_weather"
        assert tool_chunks[1]["function"]["arguments"] == '{"city":"Paris"}'
        assert any(
            f.get("choices") and f["choices"][0]["finish_reason"] == "tool_calls"
            for f in frames
        )
        usage_frames = [f["usage"] for f in frames if "usage" in f]
        assert usage_frames == [
            {"prompt_tokens": 11, "completion_tokens": 17, "total_tokens": 28}
        ]

        # Non-streaming fallback
        res_json = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "stream": False,
            },
        )
        assert res_json.status_code == 200
        payload = res_json.json()
        assert payload["object"] == "chat.completion"
        msg = payload["choices"][0]["message"]
        assert msg["content"] == "Checking weather."
        assert msg["reasoning_content"] == "step by step"
        assert msg["tool_calls"][0]["function"] == {
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
        assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_wire_formats_stay_strictly_isolated_across_all_three_endpoints() -> None:
    anthropic_chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "isolated ok"},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
    provider = _AnthropicOnlyFakeProvider(anthropic_chunks)
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        # 1. Claude Code -> /v1/messages (Anthropic)
        r_msg = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert r_msg.status_code == 200
        assert r_msg.json()["type"] == "message"
        assert r_msg.json()["content"][0]["text"] == "isolated ok"

        # 2. Codex / Responses harness -> /v1/responses (OpenAI Responses)
        r_resp = client.post(
            "/v1/responses",
            json={"model": "claude-sonnet-4-5", "input": "hi", "stream": True},
        )
        assert r_resp.status_code == 200
        assert "event: response.completed" in r_resp.text

        # 3. Chat Completions harness -> /v1/chat/completions (OpenAI Chat)
        r_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert r_chat.status_code == 200
        assert r_chat.json()["object"] == "chat.completion"
        assert r_chat.json()["choices"][0]["message"]["content"] == "isolated ok"

        # Probe endpoint
        r_opt = client.options("/v1/chat/completions")
        assert r_opt.status_code == 204
        assert "POST" in r_opt.headers["allow"]

        # Empty messages returns OpenAI error shape
        r_err = client.post(
            "/v1/chat/completions",
            json={"model": "claude-sonnet-4-5", "messages": []},
        )
        assert r_err.status_code == 400
        assert "error" in r_err.json()


def test_chat_completions_passthrough_upstream_error_returns_openai_error() -> None:
    captured: list[dict[str, Any]] = []
    provider = _make_mock_deepseek_provider(captured, b"", status_code=401)
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-chat",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert res.status_code == 401
        assert res.headers.get("x-should-retry") == "false"
        assert "error" in res.json()


def test_responses_adapter_fast_messages_request_construction() -> None:
    adapter = OpenAIResponsesAdapter()
    req = OpenAIResponsesRequest(
        model="claude-sonnet-4-5",
        instructions="Be helpful",
        input="Hello",
        max_output_tokens=256,
        temperature=0.5,
    )
    built = adapter.to_anthropic_messages_request(req)
    assert built.model == "claude-sonnet-4-5"
    assert built.system == "Be helpful"
    assert built.max_tokens == 256
    assert built.temperature == 0.5
    assert len(built.messages) == 1
    assert built.messages[0].role == "user"


def test_chat_completions_non_streaming_propagates_mid_stream_error() -> None:
    anthropic_chunks = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_err","model":"claude-sonnet-4-5","usage":{"input_tokens":5}}}\n\n',
        'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"upstream broke mid-stream"}}\n\n',
    ]
    provider = _AnthropicOnlyFakeProvider(anthropic_chunks)
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert res.status_code == 502
        body = res.json()
        assert "upstream broke mid-stream" in body["error"]["message"]


def test_chat_completions_sets_openai_request_id_and_tracks_inference_lifetime() -> (
    None
):
    from free_claude_code.api.request_lifetime import _INFERENCE_PATHS
    from free_claude_code.core.anthropic.models import (
        ContentBlockGeneric,
        ContentBlockText,
        MessagesRequest,
    )

    assert "/v1/chat/completions" in _INFERENCE_PATHS

    # Verify tagged ContentBlock discriminator routes known and unknown block types.
    req = MessagesRequest.model_validate(
        {
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "future_custom_block", "payload": 42},
                    ],
                }
            ],
        }
    )
    assert isinstance(req.messages[0].content, list)
    assert isinstance(req.messages[0].content[0], ContentBlockText)
    assert isinstance(req.messages[0].content[1], ContentBlockGeneric)
    assert req.messages[0].content[1].type == "future_custom_block"

    anthropic_chunks = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_hdr","model":"claude-sonnet-4-5","usage":{"input_tokens":3}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    provider = _AnthropicOnlyFakeProvider(anthropic_chunks)
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert res.status_code == 200
        assert res.headers.get("request-id")
        assert res.headers.get("x-request-id") == res.headers.get("request-id")

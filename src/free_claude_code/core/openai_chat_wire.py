"""Inbound OpenAI Chat Completions wire protocol models and translators."""

import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any

from pydantic import BaseModel, ConfigDict

from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
    Tool,
    construct_content_block,
)
from free_claude_code.core.anthropic.sse_parse import iter_sse_events
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_utils import (
    fast_json_dumps,
    fast_json_loads,
    safe_parse_tool_arguments,
)
from free_claude_code.core.openai_responses import openai_failure_payload

OPENAI_CHAT_SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
OPENAI_CHAT_SSE_DONE = "data: [DONE]\n\n"

_ANTHROPIC_STOP_REASON_TO_OPENAI: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class OpenAIChatCompletionsRequest(BaseModel):
    """Inbound ``POST /v1/chat/completions`` request payload."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None


def new_chat_completion_id() -> str:
    """Return a standard OpenAI Chat Completions identifier."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def format_chat_completion_sse_chunk(payload: Mapping[str, Any]) -> str:
    """Format one OpenAI Chat Completions SSE data frame."""
    return f"data: {fast_json_dumps(payload)}\n\n"


def format_chat_completion_sse_error(failure: ExecutionFailure) -> str:
    """Format a post-start execution failure as an OpenAI SSE error frame."""
    return format_chat_completion_sse_chunk(openai_failure_payload(failure))


def build_passthrough_chat_body(
    request: OpenAIChatCompletionsRequest,
    *,
    provider_model: str,
    force_reasoning_off: bool = False,
) -> dict[str, Any]:
    """Build an upstream Chat Completions body directly without Anthropic conversion."""
    body: dict[str, Any] = {
        "model": provider_model,
        "messages": request.messages,
        "stream": True,
    }
    if request.stream_options is not None:
        body["stream_options"] = dict(request.stream_options)
    elif not request.stream:
        # Non-streaming callers still expect usage in the aggregated completion JSON.
        body["stream_options"] = {"include_usage": True}

    if request.tools is not None:
        body["tools"] = request.tools
    if request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    if request.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = request.parallel_tool_calls
    if request.max_completion_tokens is not None:
        body["max_completion_tokens"] = request.max_completion_tokens
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop"] = request.stop
    if request.reasoning_effort is not None and not force_reasoning_off:
        body["reasoning_effort"] = request.reasoning_effort
    if request.extra_body is not None:
        body["extra_body"] = dict(request.extra_body)

    if request.model_extra:
        for key, value in request.model_extra.items():
            if key not in body and value is not None:
                body[key] = value

    if force_reasoning_off:
        body.pop("reasoning_effort", None)
        body.pop("thinking", None)
        extra = body.get("extra_body")
        if isinstance(extra, dict):
            extra = dict(extra)
            extra["thinking"] = {"type": "disabled"}
            body["extra_body"] = extra
        else:
            body["extra_body"] = {"thinking": {"type": "disabled"}}

    return body


def serialize_upstream_chat_chunk(
    chunk: Any,
    *,
    response_model: str,
    completion_id: str,
    created: int,
) -> dict[str, Any] | None:
    """Extract a JSON-ready OpenAI ChatCompletionChunk dict from an upstream chunk."""
    raw: dict[str, Any] | None = None
    extra = getattr(chunk, "model_extra", None)
    if isinstance(extra, dict) and ("choices" in extra or "usage" in extra):
        raw = dict(extra)
    elif hasattr(chunk, "model_dump"):
        dumped = chunk.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            raw = dumped
    elif isinstance(chunk, Mapping):
        raw = dict(chunk)

    if raw is None:
        return None

    raw.setdefault("id", completion_id)
    raw["object"] = "chat.completion.chunk"
    raw.setdefault("created", created)
    raw["model"] = response_model
    if "choices" not in raw:
        raw["choices"] = []
    return raw


def _convert_chat_content_to_anthropic(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": str(part.get("text", ""))})
        elif ptype == "image_url":
            image_url = part.get("image_url")
            url = (
                image_url.get("url", "")
                if isinstance(image_url, dict)
                else str(image_url or "")
            )
            if url.startswith("data:") and ";base64," in url:
                header, b64_data = url.split(";base64,", 1)
                media_type = header[5:] or "image/png"
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    }
                )
            elif url:
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                )
        elif ptype:
            blocks.append(dict(part))
    return blocks or ""


def _convert_chat_tool_content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(fast_json_dumps(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return fast_json_dumps(content)
    return str(content)


def chat_request_to_anthropic_messages_request(
    request: OpenAIChatCompletionsRequest,
) -> MessagesRequest:
    """Convert an OpenAI Chat Completions request to an Anthropic MessagesRequest.

    Used when a ``/v1/chat/completions`` caller targets a non-OpenAI-Chat
    upstream (such as ``openai_codex``, Gemini native, or Bedrock).
    """
    system_parts: list[str] = []
    anthropic_messages: list[Message] = []
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if pending_tool_results:
            anthropic_messages.append(
                Message.model_construct(
                    role="user",
                    content=[
                        construct_content_block(block) for block in pending_tool_results
                    ],
                    reasoning_content=None,
                )
            )
            pending_tool_results.clear()

    for raw_msg in request.messages:
        role = str(raw_msg.get("role", "")).strip().lower()
        if role in ("system", "developer"):
            text = _convert_chat_tool_content_to_str(raw_msg.get("content"))
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(raw_msg.get("tool_call_id") or ""),
                    "content": _convert_chat_tool_content_to_str(
                        raw_msg.get("content")
                    ),
                }
            )
            continue

        _flush_tool_results()

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            reasoning = raw_msg.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = raw_msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                blocks.append({"type": "thinking", "thinking": reasoning})

            converted_content = _convert_chat_content_to_anthropic(
                raw_msg.get("content")
            )
            if isinstance(converted_content, str):
                if converted_content:
                    blocks.append({"type": "text", "text": converted_content})
            elif isinstance(converted_content, list):
                blocks.extend(converted_content)

            raw_tool_calls = raw_msg.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for idx, tc in enumerate(raw_tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    fn = (
                        tc.get("function")
                        if isinstance(tc.get("function"), dict)
                        else {}
                    )
                    tc_id = str(tc.get("id") or f"call_{idx}")
                    tc_name = str(fn.get("name") or "")
                    tc_args = safe_parse_tool_arguments(fn.get("arguments"))
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": tc_args,
                        }
                    )

            anthropic_messages.append(
                Message.model_construct(
                    role="assistant",
                    content=[construct_content_block(block) for block in blocks]
                    if blocks
                    else "",
                    reasoning_content=reasoning if isinstance(reasoning, str) else None,
                )
            )
            continue

        anthropic_messages.append(
            Message.model_construct(
                role="user",
                content=(
                    [construct_content_block(block) for block in converted]
                    if isinstance(
                        converted := _convert_chat_content_to_anthropic(
                            raw_msg.get("content")
                        ),
                        list,
                    )
                    else converted
                ),
                reasoning_content=None,
            )
        )

    _flush_tool_results()

    tools: list[Tool] | None = None
    if request.tools:
        converted_tools: list[Tool] = []
        for raw_tool in request.tools:
            if not isinstance(raw_tool, dict):
                continue
            if raw_tool.get("type") == "function" and isinstance(
                raw_tool.get("function"), dict
            ):
                fn = raw_tool["function"]
                name = str(fn.get("name") or "").strip()
                if not name:
                    continue
                schema = fn.get("parameters")
                converted_tools.append(
                    Tool.model_construct(
                        name=name,
                        type=None,
                        description=fn.get("description"),
                        input_schema=schema
                        if isinstance(schema, dict)
                        else {"type": "object", "properties": {}},
                    )
                )
        if converted_tools:
            tools = converted_tools

    tool_choice: dict[str, Any] | None = None
    raw_choice = request.tool_choice
    if isinstance(raw_choice, str):
        if raw_choice == "auto":
            tool_choice = {"type": "auto"}
        elif raw_choice == "required":
            tool_choice = {"type": "any"}
        elif raw_choice == "none":
            tool_choice = {"type": "none"}
            tools = None
    elif (
        isinstance(raw_choice, dict)
        and raw_choice.get("type") == "function"
        and isinstance(raw_choice.get("function"), dict)
    ):
        fn_name = raw_choice["function"].get("name")
        if isinstance(fn_name, str) and fn_name:
            tool_choice = {"type": "tool", "name": fn_name}

    stop_sequences: list[str] | None = None
    if isinstance(request.stop, str):
        stop_sequences = [request.stop]
    elif isinstance(request.stop, list):
        stop_sequences = [str(item) for item in request.stop]

    output_config: dict[str, Any] | None = None
    if isinstance(request.reasoning_effort, str) and request.reasoning_effort.strip():
        output_config = {"effort": request.reasoning_effort.strip().lower()}

    max_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )

    return MessagesRequest.model_construct(
        model=request.model,
        original_model=None,
        resolved_provider_model=None,
        max_tokens=max_tokens,
        messages=anthropic_messages,
        system="\n\n".join(system_parts) if system_parts else None,
        stop_sequences=stop_sequences,
        stream=True,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=None,
        metadata=None,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        thinking=None,
        context_management=None,
        output_config=output_config,
        mcp_servers=None,
        extra_body=dict(request.extra_body) if request.extra_body is not None else None,
        betas=None,
    )


async def iter_openai_chat_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    *,
    model: str,
    completion_id: str,
    include_usage: bool = False,
    created: int | None = None,
) -> AsyncIterator[str]:
    """Translate an Anthropic SSE stream into OpenAI Chat Completions SSE frames."""
    created_ts = created if created is not None else int(time.time())
    sent_role = False
    finish_reason: str | None = None
    input_tokens = 0
    output_tokens = 0
    tool_index_by_block: dict[int, int] = {}
    next_tool_index = 0

    def _chunk(
        *,
        delta: dict[str, Any] | None = None,
        finish: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": []
            if delta is None and finish is None and usage is not None
            else [
                {
                    "index": 0,
                    "delta": delta if delta is not None else {},
                    "finish_reason": finish,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return format_chat_completion_sse_chunk(payload)

    async with aclosing(
        iter_sse_events(chunks, owner="openai_chat_wire.stream")
    ) as events:
        async for event in events:
            ev_type = event.event or str(event.data.get("type") or "")
            data = event.data

            if ev_type == "message_start":
                message = data.get("message")
                if isinstance(message, dict):
                    usage = message.get("usage")
                    if isinstance(usage, dict):
                        input_tokens = (
                            int(usage.get("input_tokens") or 0)
                            + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0)
                        )
                if not sent_role:
                    sent_role = True
                    yield _chunk(delta={"role": "assistant", "content": ""})
                continue

            if ev_type == "content_block_start":
                block = data.get("content_block")
                block_idx = data.get("index")
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if not sent_role:
                        sent_role = True
                        yield _chunk(delta={"role": "assistant"})
                    tc_idx = next_tool_index
                    next_tool_index += 1
                    if isinstance(block_idx, int):
                        tool_index_by_block[block_idx] = tc_idx
                    initial_args = ""
                    raw_input = block.get("input")
                    if isinstance(raw_input, dict) and raw_input:
                        initial_args = fast_json_dumps(raw_input)
                    yield _chunk(
                        delta={
                            "tool_calls": [
                                {
                                    "index": tc_idx,
                                    "id": str(block.get("id") or f"call_{tc_idx}"),
                                    "type": "function",
                                    "function": {
                                        "name": str(block.get("name") or ""),
                                        "arguments": initial_args,
                                    },
                                }
                            ]
                        }
                    )
                continue

            if ev_type == "content_block_delta":
                delta = data.get("delta")
                if not isinstance(delta, dict):
                    continue
                dtype = delta.get("type")
                if dtype == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        if not sent_role:
                            sent_role = True
                            yield _chunk(delta={"role": "assistant", "content": text})
                        else:
                            yield _chunk(delta={"content": text})
                elif dtype == "thinking_delta":
                    thinking = delta.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        if not sent_role:
                            sent_role = True
                            yield _chunk(
                                delta={
                                    "role": "assistant",
                                    "reasoning_content": thinking,
                                }
                            )
                        else:
                            yield _chunk(delta={"reasoning_content": thinking})
                elif dtype == "input_json_delta":
                    partial = delta.get("partial_json")
                    block_idx = data.get("index")
                    tc_idx = (
                        tool_index_by_block.get(block_idx, 0)
                        if isinstance(block_idx, int)
                        else 0
                    )
                    if isinstance(partial, str) and partial:
                        yield _chunk(
                            delta={
                                "tool_calls": [
                                    {
                                        "index": tc_idx,
                                        "function": {"arguments": partial},
                                    }
                                ]
                            }
                        )
                continue

            if ev_type == "message_delta":
                delta = data.get("delta")
                if isinstance(delta, dict):
                    raw_stop = delta.get("stop_reason")
                    if isinstance(raw_stop, str) and raw_stop:
                        finish_reason = _ANTHROPIC_STOP_REASON_TO_OPENAI.get(
                            raw_stop, "stop"
                        )
                usage = data.get("usage")
                if isinstance(usage, dict):
                    if usage.get("output_tokens") is not None:
                        output_tokens = int(usage["output_tokens"])
                    if usage.get("input_tokens") is not None and input_tokens == 0:
                        input_tokens = int(usage["input_tokens"])
                continue

            if ev_type == "error":
                err = data.get("error")
                err_msg = (
                    str(err.get("message"))
                    if isinstance(err, dict) and err.get("message")
                    else "Upstream stream error."
                )
                yield format_chat_completion_sse_chunk(
                    {"error": {"message": err_msg, "type": "api_error"}}
                )
                yield OPENAI_CHAT_SSE_DONE
                return

    if not sent_role:
        yield _chunk(delta={"role": "assistant", "content": ""})
    yield _chunk(delta={}, finish=finish_reason or "stop")
    if include_usage:
        yield _chunk(
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        )
    yield OPENAI_CHAT_SSE_DONE


async def collect_openai_chat_completion_from_sse(
    sse_chunks: AsyncIterable[str],
    *,
    model: str,
    completion_id: str,
    created: int | None = None,
) -> dict[str, Any]:
    """Aggregate OpenAI Chat Completions SSE frames into one JSON completion object."""
    created_ts = created if created is not None else int(time.time())
    resolved_id = completion_id
    resolved_model = model
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    async for frame in sse_chunks:
        for line in frame.splitlines():
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                parsed = fast_json_loads(payload_text)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            raw_error = parsed.get("error")
            if isinstance(raw_error, dict):
                err_msg = (
                    str(raw_error.get("message"))
                    if raw_error.get("message")
                    else "Upstream stream error."
                )
                raise ExecutionFailure(
                    kind=FailureKind.UPSTREAM,
                    status_code=502,
                    message=err_msg,
                    retryable=False,
                )
            if isinstance(parsed.get("id"), str) and parsed["id"]:
                resolved_id = parsed["id"]
            if isinstance(parsed.get("model"), str) and parsed["model"]:
                resolved_model = parsed["model"]
            if isinstance(parsed.get("created"), int):
                created_ts = parsed["created"]
            if isinstance(parsed.get("usage"), dict):
                usage = parsed["usage"]
            choices = parsed.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            text = delta.get("content")
            if isinstance(text, str) and text:
                content_parts.append(text)
            reasoning = delta.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)
            raw_tcs = delta.get("tool_calls")
            if isinstance(raw_tcs, list):
                for tc in raw_tcs:
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index")
                    tc_index = idx if isinstance(idx, int) else len(tool_calls_by_index)
                    entry = tool_calls_by_index.setdefault(
                        tc_index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if isinstance(tc.get("id"), str) and tc["id"]:
                        entry["id"] = tc["id"]
                    if isinstance(tc.get("type"), str) and tc["type"]:
                        entry["type"] = tc["type"]
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        if isinstance(fn.get("name"), str) and fn["name"]:
                            entry["function"]["name"] += fn["name"]
                        if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                            entry["function"]["arguments"] += fn["arguments"]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls_by_index:
        message["tool_calls"] = [
            tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index)
        ]
    elif message["content"] is None:
        message["content"] = ""

    result: dict[str, Any] = {
        "id": resolved_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": resolved_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
                or ("tool_calls" if tool_calls_by_index else "stop"),
            }
        ],
    }
    if usage is not None:
        result["usage"] = usage
    return result

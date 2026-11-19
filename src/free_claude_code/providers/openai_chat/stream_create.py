"""Open Chat Completions streams without the OpenAI SDK's body or chunk overhead."""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import orjson
from openai import APIError, AsyncOpenAI, AsyncStream
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# AsyncCompletions.create() deep-walks the body against its TypedDicts and then
# encodes it with the stdlib json module: ~40 ms for a 1 MiB Claude Code
# history on a small VM, paid before the request leaves. FCC bodies are
# already plain JSON, so the walk changes nothing and orjson encodes the same
# bytes in well under a millisecond.
#
# Similarly, the default AsyncStream instantiates a full Pydantic
# ChatCompletionChunk tree on every SSE token. FastOrjsonChatStream parses the
# same SSE byte stream with orjson into lightweight slotted objects while
# keeping AsyncOpenAI's auth headers, HTTP/2 pooling, and error handling intact.
_SDK_CREATE = AsyncCompletions.create
# create() keywords that are request options rather than body fields.
_REQUEST_OPTION_KEYS = frozenset({"extra_headers", "extra_query", "timeout"})
_BUF_COMPACT_THRESHOLD = 65_536


class _FastToolCallFunction:
    __slots__ = ("arguments", "name")

    def __init__(self, name: Any, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _FastToolCall:
    __slots__ = ("extra_content", "function", "id", "index", "model_extra", "type")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.index = raw.get("index")
        self.id = raw.get("id")
        self.type = raw.get("type")
        fn = raw.get("function")
        self.function = (
            _FastToolCallFunction(fn.get("name"), fn.get("arguments"))
            if isinstance(fn, dict)
            else None
        )
        self.extra_content = raw.get("extra_content")
        self.model_extra = raw

    def __getattr__(self, name: str) -> Any:
        return self.model_extra.get(name)


class _FastDelta:
    __slots__ = (
        "content",
        "extra_content",
        "model_extra",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "role",
        "tool_calls",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.content = raw.get("content")
        self.role = raw.get("role")
        self.reasoning_content = raw.get("reasoning_content")
        self.reasoning = raw.get("reasoning")
        self.reasoning_details = raw.get("reasoning_details")
        self.extra_content = raw.get("extra_content")
        raw_tcs = raw.get("tool_calls")
        if isinstance(raw_tcs, list) and raw_tcs:
            self.tool_calls: list[_FastToolCall] | None = [
                _FastToolCall(tc) for tc in raw_tcs if isinstance(tc, dict)
            ] or None
        else:
            self.tool_calls = None
        self.model_extra = raw

    def __getattr__(self, name: str) -> Any:
        return self.model_extra.get(name)


class _FastChoice:
    __slots__ = ("delta", "finish_reason", "index", "model_extra")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.index = raw.get("index", 0)
        self.finish_reason = raw.get("finish_reason")
        raw_delta = raw.get("delta")
        self.delta = _FastDelta(raw_delta) if isinstance(raw_delta, dict) else None
        self.model_extra = raw

    def __getattr__(self, name: str) -> Any:
        return self.model_extra.get(name)


class _FastUsageDetails:
    __slots__ = (
        "cache_write_tokens",
        "cached_tokens",
        "model_extra",
        "reasoning_tokens",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.cached_tokens = raw.get("cached_tokens")
        self.cache_write_tokens = raw.get("cache_write_tokens")
        self.reasoning_tokens = raw.get("reasoning_tokens")
        self.model_extra = raw

    def __getattr__(self, name: str) -> Any:
        return self.model_extra.get(name)


class _FastUsage:
    __slots__ = (
        "completion_tokens",
        "completion_tokens_details",
        "model_extra",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "prompt_tokens",
        "prompt_tokens_details",
        "total_tokens",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.prompt_tokens = raw.get("prompt_tokens")
        self.completion_tokens = raw.get("completion_tokens")
        self.total_tokens = raw.get("total_tokens")
        prompt_details = raw.get("prompt_tokens_details")
        self.prompt_tokens_details = (
            _FastUsageDetails(prompt_details)
            if isinstance(prompt_details, dict)
            else prompt_details
        )
        completion_details = raw.get("completion_tokens_details")
        self.completion_tokens_details = (
            _FastUsageDetails(completion_details)
            if isinstance(completion_details, dict)
            else completion_details
        )
        self.prompt_cache_hit_tokens = raw.get("prompt_cache_hit_tokens")
        self.prompt_cache_miss_tokens = raw.get("prompt_cache_miss_tokens")
        self.model_extra = raw

    def __getattr__(self, name: str) -> Any:
        return self.model_extra.get(name)


class _FastChunk:
    __slots__ = ("choices", "id", "model", "model_extra", "usage")

    def __init__(self, raw: dict[str, Any]) -> None:
        self.id = raw.get("id")
        self.model = raw.get("model")
        raw_choices = raw.get("choices")
        if isinstance(raw_choices, list) and raw_choices:
            self.choices: list[_FastChoice] = [
                _FastChoice(choice)
                for choice in raw_choices
                if isinstance(choice, dict)
            ]
        else:
            self.choices = []
        raw_usage = raw.get("usage")
        self.usage = _FastUsage(raw_usage) if isinstance(raw_usage, dict) else None
        self.model_extra = raw

    def __getattr__(self, name: str) -> Any:
        return self.model_extra.get(name)


def _decode_sse_json(raw: bytes) -> Any:
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        return json.loads(raw.decode("utf-8", errors="replace"))


class FastOrjsonChatStream[ResponseT](AsyncStream[ResponseT]):
    """Zero-Pydantic orjson SSE stream plugged into AsyncOpenAI.post()."""

    def __init__(self, *, cast_to: type[ResponseT], response: Any, client: Any) -> None:
        self._buf = bytearray()
        self._pos = 0
        self._event_name = ""
        self._data_lines: list[bytes] = []
        self._done = False
        super().__init__(cast_to=cast_to, response=response, client=client)

    async def aclose(self) -> None:
        await self.close()

    def pop_ready_nowait(self) -> tuple[bool, Any]:
        """Synchronously parse and pop one already-buffered SSE chunk if available."""
        if self._done:
            return False, None
        return self._consume_buffered_event()

    def _dispatch_event(
        self, event_name: str, data_lines: list[bytes]
    ) -> tuple[bool, Any]:
        raw = data_lines[0] if len(data_lines) == 1 else b"\n".join(data_lines)
        stripped = raw.strip()
        if not stripped:
            return False, None
        if stripped == b"[DONE]":
            self._done = True
            return False, None
        payload = _decode_sse_json(stripped)
        if event_name == "error" or (
            isinstance(payload, dict)
            and payload.get("error") is not None
            and not payload.get("choices")
        ):
            err = payload.get("error") if isinstance(payload, dict) else payload
            if isinstance(err, dict):
                message = err.get("message") or str(err)
            else:
                message = str(err) if err else "An error occurred during streaming"
            raise APIError(
                message=str(message),
                request=self.response.request,
                body=payload,
            )
        if isinstance(payload, dict):
            return True, _FastChunk(payload)
        return False, None

    def _consume_line(self, line: bytes) -> tuple[bool, Any]:
        if not line:
            if not self._data_lines:
                self._event_name = ""
                return False, None
            event_name = self._event_name
            data_lines = self._data_lines
            self._event_name = ""
            self._data_lines = []
            return self._dispatch_event(event_name, data_lines)
        if line.startswith(b":"):
            return False, None
        if line.startswith(b"data:"):
            val = line[5:]
            if val.startswith(b" "):
                val = val[1:]
            self._data_lines.append(val)
            return False, None
        if line.startswith(b"event:"):
            self._event_name = line[6:].strip().decode("utf-8", errors="replace")
        return False, None

    def _consume_buffered_event(self) -> tuple[bool, Any]:
        buf = self._buf
        while not self._done:
            nl = buf.find(b"\n", self._pos)
            if nl == -1:
                break
            end = nl - 1 if nl > self._pos and buf[nl - 1] == 13 else nl
            line = bytes(buf[self._pos : end])
            self._pos = nl + 1
            if self._pos >= len(buf):
                buf.clear()
                self._pos = 0
            elif self._pos >= _BUF_COMPACT_THRESHOLD:
                del buf[: self._pos]
                self._pos = 0
            has_event, chunk = self._consume_line(line)
            if has_event:
                return True, chunk
        return False, None

    def _flush_eof(self) -> Any | None:
        if self._pos < len(self._buf):
            trailing = bytes(self._buf[self._pos :]).rstrip(b"\r")
            self._buf.clear()
            self._pos = 0
            if trailing:
                self._consume_line(trailing)
        if self._data_lines:
            event_name = self._event_name
            data_lines = self._data_lines
            self._event_name = ""
            self._data_lines = []
            has_event, chunk = self._dispatch_event(event_name, data_lines)
            if has_event:
                return chunk
        return None

    async def __stream__(self) -> AsyncIterator[Any]:
        try:
            async for raw in self.response.aiter_bytes():
                if self._done:
                    break
                if raw:
                    self._buf.extend(raw)
                while not self._done:
                    has_event, chunk = self._consume_buffered_event()
                    if not has_event:
                        break
                    yield chunk
                if self._done:
                    break
            if not self._done:
                final_chunk = self._flush_eof()
                if final_chunk is not None:
                    yield final_chunk
        finally:
            self._done = True
            self._buf.clear()
            self._pos = 0
            self._data_lines.clear()
            await self.response.aclose()


async def create_chat_stream(client: Any, body: Mapping[str, Any]) -> Any:
    """Open one streaming chat completion with the same wire request as create()."""
    content = _encoded_body(client, body)
    if content is None:
        return await client.chat.completions.create(**body, stream=True)
    return await client.post(
        "/chat/completions",
        cast_to=ChatCompletion,
        content=content,
        options={"security": {"bearer_auth": True}},
        stream=True,
        stream_cls=FastOrjsonChatStream[ChatCompletionChunk],
    )


def _encoded_body(client: Any, body: Mapping[str, Any]) -> bytes | None:
    """Return the request bytes, or None when create() must build them itself."""
    if not isinstance(client, AsyncOpenAI):
        return None
    completions = client.chat.completions
    # A wrapped or instrumented create() must still see every request.
    if "create" in vars(completions) or type(completions).create is not _SDK_CREATE:
        return None
    if not _REQUEST_OPTION_KEYS.isdisjoint(body):
        return None
    payload = {key: value for key, value in body.items() if key != "extra_body"}
    payload["stream"] = True
    extra_body = body.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, Mapping):
            return None
        # create() merges extra_body over the body, so its keys win.
        payload.update(extra_body)
    try:
        return orjson.dumps(payload)
    except TypeError:
        return None

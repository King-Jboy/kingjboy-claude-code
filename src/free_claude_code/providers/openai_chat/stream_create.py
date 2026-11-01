"""Open Chat Completions streams without the OpenAI SDK's body transform."""

from collections.abc import Mapping
from typing import Any

import orjson
from openai import AsyncOpenAI, AsyncStream
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletion, ChatCompletionChunk

# AsyncCompletions.create() deep-walks the body against its TypedDicts and then
# encodes it with the stdlib json module: ~40 ms for a 1 MiB Claude Code
# history on a small VM, paid before the request leaves. FCC bodies are
# already plain JSON, so the walk changes nothing and orjson encodes the same
# bytes in well under a millisecond.
_SDK_CREATE = AsyncCompletions.create
# create() keywords that are request options rather than body fields.
_REQUEST_OPTION_KEYS = frozenset({"extra_headers", "extra_query", "timeout"})


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
        stream_cls=AsyncStream[ChatCompletionChunk],
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

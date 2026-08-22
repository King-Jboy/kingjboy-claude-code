"""Drive KeyPool through the real OpenAI SDK against recorded upstream replies.

These tests exercise the whole client stack - SDK error mapping included -
rather than a hand-written double. That matters because the pool branches on
the exception subclass the SDK raises, and which subclass that is depends on a
status code only the provider gets to choose.

The SDK's 3.x line speaks httpx2, so the replies are served from an
httpx2.MockTransport instead of respx: respx patches the httpx 0.x transports
the SDK no longer uses.
"""

import httpx2
import pytest
from openai import APIStatusError, AsyncOpenAI

from free_claude_code.providers.key_pool import KeyFailureAction, KeyPool
from tests.providers.provider_responses import (
    NVIDIA_INVALID_KEY,
    OPENROUTER_INVALID_KEY,
    OPENROUTER_RATE_LIMITED,
    UNAUTHENTICATED_MODEL_LIST,
    RecordedResponse,
)

_BASE_URL = "https://recorded.provider.test/v1"


def _reply(recorded: RecordedResponse) -> httpx2.Response:
    return httpx2.Response(
        recorded.status, json=recorded.json, headers=dict(recorded.headers or {})
    )


def _recorded_pool(
    keys: tuple[str, ...],
    *,
    chat_replies: list[RecordedResponse] | None = None,
) -> KeyPool:
    """Return a pool whose clients answer from recorded replies, in order."""
    replies = list(chat_replies or [])

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/models"):
            return _reply(UNAUTHENTICATED_MODEL_LIST)
        if replies:
            return _reply(replies.pop(0))
        return httpx2.Response(500, json={"message": "no scripted reply left"})

    def client_factory(key: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=key,
            base_url=_BASE_URL,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )

    return KeyPool(
        keys,
        provider_name="RECORDED",
        client_factory=client_factory,
    )


async def _complete(client: AsyncOpenAI):
    return await client.chat.completions.create(
        model="recorded/model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )


@pytest.mark.asyncio
async def test_recorded_nvidia_refusal_cools_rather_than_retires() -> None:
    # NVIDIA sends 403 for a dead key. Retiring on it would let one refused
    # request walk the pool and kill every key.
    pool = _recorded_pool(("dead", "spare"), chat_replies=[NVIDIA_INVALID_KEY])
    lease = await pool.acquire()

    with pytest.raises(APIStatusError) as raised:
        await _complete(lease.client)
    action = pool.record_failure(lease, raised.value)

    assert raised.value.status_code == 403
    assert action is KeyFailureAction.HOP_AMBIGUOUS
    assert pool.status().retired == 0
    await pool.aclose()


@pytest.mark.asyncio
async def test_recorded_openrouter_refusal_cools_the_key() -> None:
    # OpenRouter sends 401 for the same condition: unambiguous, so the key sits
    # out for a long authentication cooldown - but never permanently.
    pool = _recorded_pool(("dead", "spare"), chat_replies=[OPENROUTER_INVALID_KEY])
    lease = await pool.acquire()

    with pytest.raises(APIStatusError) as raised:
        await _complete(lease.client)
    action = pool.record_failure(lease, raised.value)

    assert raised.value.status_code == 401
    assert action is KeyFailureAction.HOP
    assert pool.status().retired == 1
    await pool.aclose()


@pytest.mark.asyncio
async def test_recorded_rate_limit_headers_drive_the_cooldown() -> None:
    # No Retry-After, reset expressed as epoch milliseconds. Guessing this
    # wrong would idle a key for hours or hammer it immediately.
    pool = _recorded_pool(
        ("throttled", "spare"), chat_replies=[OPENROUTER_RATE_LIMITED]
    )
    lease = await pool.acquire()

    with pytest.raises(APIStatusError) as raised:
        await _complete(lease.client)
    action = pool.record_failure(lease, raised.value)

    assert raised.value.status_code == 429
    assert action is KeyFailureAction.HOP
    # The reset is a fixed past instant, so the parse floors at zero and the
    # default cooldown carries the key instead.
    assert pool.status().cooling == 1
    await pool.aclose()


@pytest.mark.asyncio
async def test_a_dead_key_is_walked_past_to_serve_the_request() -> None:
    _SUCCESS_JSON = {
        "id": "cmpl-recorded",
        "object": "chat.completion",
        "created": 0,
        "model": "recorded/model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "stop",
            }
        ],
    }
    success = RecordedResponse(200, _SUCCESS_JSON, {}, "recorded success")
    pool = _recorded_pool(("dead", "alive"), chat_replies=[NVIDIA_INVALID_KEY, success])

    completion = await pool.run_key_local(_complete)

    assert completion.choices[0].message.content == "OK"
    await pool.aclose()


@pytest.mark.asyncio
async def test_model_listing_succeeds_on_a_dead_key_and_proves_nothing() -> None:
    # Recorded from both pooled providers: /v1/models ignores the credential.
    # Counting this as recovery would reset a dead key's failure streak on
    # every discovery refresh.
    pool = _recorded_pool(("dead",))
    lease = await pool.acquire()
    key = pool._keys[lease.index]
    pool.record_failure(lease, _unauthorized())
    pool.record_failure(lease, _unauthorized())
    assert key.consecutive_failures == 2

    key.cooling_until = 0.0
    await pool.run_key_local(
        lambda client: client.models.list(), proves_credential=False
    )

    assert key.consecutive_failures == 2
    await pool.aclose()


def _unauthorized() -> httpx2.HTTPStatusError:
    request = httpx2.Request("POST", f"{_BASE_URL}/chat/completions")
    response = httpx2.Response(
        OPENROUTER_INVALID_KEY.status,
        request=request,
        json=OPENROUTER_INVALID_KEY.json,
    )
    return httpx2.HTTPStatusError("401", request=request, response=response)

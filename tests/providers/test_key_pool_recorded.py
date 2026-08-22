"""Drive KeyPool through the real OpenAI SDK against recorded upstream replies.

These tests exercise the whole client stack - SDK error mapping included -
rather than a hand-written double. That matters because the pool branches on
the exception subclass the SDK raises, and which subclass that is depends on a
status code only the provider gets to choose.
"""

import httpx
import pytest
import respx
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
_COMPLETIONS = f"{_BASE_URL}/chat/completions"


def _pool(keys: tuple[str, ...]) -> KeyPool:
    return KeyPool(
        keys,
        provider_name="RECORDED",
        client_factory=lambda key: AsyncOpenAI(
            api_key=key, base_url=_BASE_URL, max_retries=0
        ),
    )


def _reply(recorded: RecordedResponse) -> httpx.Response:
    return httpx.Response(
        recorded.status, json=recorded.json, headers=recorded.headers or None
    )


async def _complete(client: AsyncOpenAI):
    return await client.chat.completions.create(
        model="recorded/model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )


_SUCCESS = {
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


@pytest.mark.asyncio
@respx.mock
async def test_recorded_nvidia_refusal_cools_rather_than_retires() -> None:
    # NVIDIA sends 403 for a dead key. Retiring on it would let one refused
    # request walk the pool and kill every key.
    respx.post(_COMPLETIONS).mock(return_value=_reply(NVIDIA_INVALID_KEY))
    pool = _pool(("dead", "spare"))
    lease = await pool.acquire()

    with pytest.raises(APIStatusError) as error:
        await _complete(lease.client)
    action = pool.record_failure(lease, error.value)

    assert error.value.status_code == 403
    assert action is KeyFailureAction.HOP_AMBIGUOUS
    assert pool.status().retired == 0
    await pool.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_recorded_openrouter_refusal_cools_the_key() -> None:
    # OpenRouter sends 401 for the same condition: unambiguous, so the key sits
    # out for a long authentication cooldown - but never permanently.
    respx.post(_COMPLETIONS).mock(return_value=_reply(OPENROUTER_INVALID_KEY))
    pool = _pool(("dead", "spare"))
    lease = await pool.acquire()

    with pytest.raises(APIStatusError) as error:
        await _complete(lease.client)
    action = pool.record_failure(lease, error.value)

    assert error.value.status_code == 401
    assert action is KeyFailureAction.HOP
    assert pool.status().retired == 1
    await pool.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_recorded_rate_limit_headers_drive_the_cooldown() -> None:
    # No Retry-After, reset expressed as epoch milliseconds. Guessing this
    # wrong would idle a key for hours or hammer it immediately.
    respx.post(_COMPLETIONS).mock(return_value=_reply(OPENROUTER_RATE_LIMITED))
    pool = _pool(("throttled", "spare"))
    lease = await pool.acquire()

    with pytest.raises(APIStatusError) as error:
        await _complete(lease.client)
    action = pool.record_failure(lease, error.value)

    assert error.value.status_code == 429
    assert action is KeyFailureAction.HOP
    # The reset is a fixed past instant, so the parse floors at zero and the
    # default cooldown carries the key instead.
    assert pool.status().cooling == 1
    await pool.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_a_dead_key_is_walked_past_to_serve_the_request() -> None:
    respx.post(_COMPLETIONS).mock(
        side_effect=[
            _reply(NVIDIA_INVALID_KEY),
            httpx.Response(200, json=_SUCCESS),
        ]
    )
    pool = _pool(("dead", "alive"))

    completion = await pool.run_key_local(_complete)

    assert completion.choices[0].message.content == "OK"
    await pool.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_model_listing_succeeds_on_a_dead_key_and_proves_nothing() -> None:
    # Recorded from both pooled providers: /v1/models ignores the credential.
    # Counting this as recovery would reset a dead key's failure streak on
    # every discovery refresh.
    respx.get(f"{_BASE_URL}/models").mock(
        return_value=_reply(UNAUTHENTICATED_MODEL_LIST)
    )
    pool = _pool(("dead",))
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


def _unauthorized() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", _COMPLETIONS)
    response = httpx.Response(
        OPENROUTER_INVALID_KEY.status, request=request, json=OPENROUTER_INVALID_KEY.json
    )
    return httpx.HTTPStatusError("401", request=request, response=response)

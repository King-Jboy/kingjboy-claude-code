"""Tests for the TokenRouter OpenAI-chat provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import TOKENROUTER_DEFAULT_BASE
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def tokenrouter_provider():
    return profiled_provider(
        "tokenrouter",
        ProviderConfig(
            api_key="test_tokenrouter_key",
            base_url=TOKENROUTER_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_openai_chat_endpoint(tokenrouter_provider):
    assert isinstance(tokenrouter_provider, OpenAIChatProvider)
    assert tokenrouter_provider._api_key == "test_tokenrouter_key"
    assert tokenrouter_provider._base_url == "https://api.tokenrouter.com/v1"


def test_build_request_body_openai_chat(tokenrouter_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "openai/gpt-4o",
            "max_tokens": 100,
            "messages": [Message(role="user", content="Hello")],
        }
    )

    body = tokenrouter_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "openai/gpt-4o"
    assert body["max_tokens"] == 100
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_model_discovery(tokenrouter_provider):
    tokenrouter_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="openai/gpt-4o"),
                SimpleNamespace(id="anthropic/claude-3.5-sonnet"),
            ]
        )
    )

    assert await tokenrouter_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("openai/gpt-4o"),
            ProviderModelInfo("anthropic/claude-3.5-sonnet"),
        }
    )

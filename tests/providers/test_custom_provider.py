"""Tests for the Custom OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.support import (
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.fixture
def custom_provider():
    return profiled_provider(
        "custom",
        ProviderConfig(
            api_key="test_custom_key",
            base_url="https://api.zenmux.ai/v1",
            rate_limit=40,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_custom_base_url_and_key(custom_provider):
    assert isinstance(custom_provider, OpenAIChatProvider)
    assert custom_provider._api_key == "test_custom_key"
    assert custom_provider._base_url == "https://api.zenmux.ai/v1"


def test_build_request_body_openai_chat(custom_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "custom/meta-llama/llama-3.3-70b-instruct",
            "messages": [Message(role="user", content="Hello")],
        }
    )

    body = custom_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "custom/meta-llama/llama-3.3-70b-instruct"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert "reasoning_effort" not in body


def test_build_request_body_reasoning_effort(custom_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "custom/deepseek-ai/deepseek-r1",
            "messages": [Message(role="user", content="Hello")],
        }
    )
    policy = ReasoningPolicy(effort=ReasoningEffort.HIGH)

    body = custom_provider._build_request_body(request, reasoning=policy)

    assert body["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_model_discovery(custom_provider):
    custom_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="meta-llama/llama-3.3-70b-instruct"),
                SimpleNamespace(id="deepseek-ai/deepseek-r1"),
            ]
        )
    )

    assert await custom_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("meta-llama/llama-3.3-70b-instruct"),
            ProviderModelInfo("deepseek-ai/deepseek-r1"),
        }
    )

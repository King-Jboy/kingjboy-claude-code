"""Tests for the NaraRoute OpenAI-chat provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.config.provider_catalog import NARAROUTE_DEFAULT_BASE
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
def nararoute_provider():
    return profiled_provider(
        "nararoute",
        ProviderConfig(
            api_key="test_nararoute_key",
            base_url=NARAROUTE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        ),
        admission=immediate_admission(),
    )


def test_init_uses_openai_chat_endpoint(nararoute_provider):
    assert isinstance(nararoute_provider, OpenAIChatProvider)
    assert nararoute_provider._api_key == "test_nararoute_key"
    assert nararoute_provider._base_url == "https://router.bynara.id/v1"


def test_build_request_body_openai_chat(nararoute_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "openai/gpt-4o",
            "messages": [Message(role="user", content="Hello")],
        }
    )

    body = nararoute_provider._build_request_body(
        request, reasoning=reasoning_for(request)
    )

    assert body["model"] == "openai/gpt-4o"
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert "reasoning_effort" not in body


def test_build_request_body_reasoning_effort(nararoute_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "openai/gpt-4o",
            "messages": [Message(role="user", content="Hello")],
        }
    )
    policy = ReasoningPolicy(effort=ReasoningEffort.HIGH)

    body = nararoute_provider._build_request_body(request, reasoning=policy)

    assert body["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_model_discovery(nararoute_provider):
    nararoute_provider._client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="openai/gpt-4o"),
                SimpleNamespace(id="anthropic/claude-3.5-sonnet"),
            ]
        )
    )

    assert await nararoute_provider.list_model_infos() == frozenset(
        {
            ProviderModelInfo("openai/gpt-4o"),
            ProviderModelInfo("anthropic/claude-3.5-sonnet"),
        }
    )

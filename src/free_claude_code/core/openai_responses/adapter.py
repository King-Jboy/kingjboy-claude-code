"""Facade for OpenAI Responses protocol adaptation."""

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any, ClassVar

from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
    Tool,
    construct_content_block,
)

from .errors import ResponsesConversionError, openai_error_payload
from .events import OPENAI_RESPONSES_SSE_HEADERS
from .input import convert_request_to_anthropic_payload
from .models import OpenAIResponsesRequest
from .stream import (
    CompletedResponseObserver,
    PostStartTerminalFailureObserver,
    iter_responses_sse_from_anthropic,
)


def _construct_messages_request(payload: dict[str, Any]) -> MessagesRequest:
    """Construct a MessagesRequest directly from already-normalized payload dicts."""
    messages = [
        Message.model_construct(
            role=msg["role"],
            content=(
                [construct_content_block(block) for block in msg["content"]]
                if isinstance(msg["content"], list)
                else msg["content"]
            ),
            reasoning_content=msg.get("reasoning_content"),
        )
        for msg in payload["messages"]
    ]
    raw_tools = payload.get("tools")
    tools = (
        [
            Tool.model_construct(
                name=tool["name"],
                type=tool.get("type"),
                description=tool.get("description"),
                input_schema=tool.get("input_schema"),
            )
            for tool in raw_tools
        ]
        if isinstance(raw_tools, list)
        else None
    )
    return MessagesRequest.model_construct(
        model=payload["model"],
        original_model=None,
        resolved_provider_model=None,
        max_tokens=payload.get("max_tokens"),
        messages=messages,
        system=payload.get("system"),
        stop_sequences=payload.get("stop_sequences"),
        stream=bool(payload.get("stream", True)),
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        metadata=payload.get("metadata"),
        tools=tools,
        tool_choice=payload.get("tool_choice"),
        parallel_tool_calls=payload.get("parallel_tool_calls"),
        thinking=None,
        context_management=payload.get("context_management"),
        output_config=payload.get("output_config"),
        mcp_servers=None,
        extra_body=payload.get("extra_body"),
        betas=None,
    )


class OpenAIResponsesAdapter:
    """Convert between OpenAI Responses and the proxy's Anthropic core path."""

    ConversionError: ClassVar[type[ResponsesConversionError]] = ResponsesConversionError
    sse_headers: ClassVar[dict[str, str]] = OPENAI_RESPONSES_SSE_HEADERS

    def to_anthropic_payload(self, request: OpenAIResponsesRequest) -> dict[str, Any]:
        return convert_request_to_anthropic_payload(request)

    def to_anthropic_messages_request(
        self, request: OpenAIResponsesRequest
    ) -> MessagesRequest:
        """Convert directly to MessagesRequest without redundant Pydantic re-validation."""
        return _construct_messages_request(self.to_anthropic_payload(request))

    def iter_sse_from_anthropic(
        self,
        chunks: AsyncIterable[Any],
        request: OpenAIResponsesRequest,
        *,
        on_post_start_terminal_failure: PostStartTerminalFailureObserver | None = None,
        on_completed_response: CompletedResponseObserver | None = None,
    ) -> AsyncIterator[str]:
        return iter_responses_sse_from_anthropic(
            chunks,
            request,
            on_post_start_terminal_failure=on_post_start_terminal_failure,
            on_completed_response=on_completed_response,
        )

    def error_payload(self, *, message: str, error_type: str) -> dict[str, Any]:
        return openai_error_payload(message=message, error_type=error_type)

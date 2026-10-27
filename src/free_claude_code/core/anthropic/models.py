"""Pydantic models for the Anthropic Messages protocol."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_KNOWN_CONTENT_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "document",
        "tool_use",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "tool_reference",
        "tool_addition",
        "tool_definition",
        "tool_search_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
    }
)


class _AnthropicBlockBase(BaseModel):
    """Pass through protocol extensions such as ``cache_control``."""

    model_config = ConfigDict(extra="allow")


class ContentBlockText(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class ContentBlockImage(_AnthropicBlockBase):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockDocument(_AnthropicBlockBase):
    """Anthropic document block (e.g. PDF files via the Files API)."""

    type: Literal["document"]
    source: dict[str, Any]


class ContentBlockToolUse(_AnthropicBlockBase):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ContentBlockToolResult(_AnthropicBlockBase):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[Any] | dict[str, Any] | None = ""


class ContentBlockThinking(_AnthropicBlockBase):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ContentBlockRedactedThinking(_AnthropicBlockBase):
    type: Literal["redacted_thinking"]
    data: str


class ContentBlockServerToolUse(_AnthropicBlockBase):
    """Anthropic server-side tool invocation (e.g. ``web_search``, ``web_fetch``)."""

    type: Literal["server_tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ContentBlockWebSearchToolResult(_AnthropicBlockBase):
    type: Literal["web_search_tool_result"]
    tool_use_id: str
    content: Any = None


class ContentBlockWebFetchToolResult(_AnthropicBlockBase):
    type: Literal["web_fetch_tool_result"]
    tool_use_id: str
    content: Any = None


class ContentBlockToolReference(_AnthropicBlockBase):
    """Anthropic deferred tool reference block (e.g. via ``ToolSearch``)."""

    type: Literal["tool_reference"]
    tool_name: str | None = None
    name: str | None = None


class ContentBlockToolAddition(_AnthropicBlockBase):
    """Advisor-style beta block: adds a tool reference inline in a message."""

    type: Literal["tool_addition"]
    tool: dict[str, Any] = Field(default_factory=dict)


class ContentBlockToolDefinition(_AnthropicBlockBase):
    """Inline deferred tool definition block."""

    type: Literal["tool_definition"]
    name: str | None = None
    tool_name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    tool: dict[str, Any] | None = None


class ContentBlockToolSearchToolResult(_AnthropicBlockBase):
    """Anthropic server-side tool search result block."""

    type: Literal["tool_search_tool_result"]
    tool_use_id: str
    content: Any = None


class ContentBlockMcpToolUse(_AnthropicBlockBase):
    """Anthropic MCP connector tool invocation block."""

    type: Literal["mcp_tool_use"]
    id: str
    name: str
    server_name: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)


class ContentBlockMcpToolResult(_AnthropicBlockBase):
    """Anthropic MCP connector tool result block."""

    type: Literal["mcp_tool_result"]
    tool_use_id: str
    content: Any = None
    is_error: bool | None = None


class ContentBlockGeneric(_AnthropicBlockBase):
    """Forward-compatible fallback for unrecognized Anthropic content blocks."""

    type: str

    @field_validator("type")
    @classmethod
    def _reject_known_block_types(cls, value: str) -> str:
        if value in _KNOWN_CONTENT_BLOCK_TYPES:
            raise ValueError(
                f"Known block type {value!r} must match its specific schema."
            )
        return value


class SystemContent(_AnthropicBlockBase):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: (
        str
        | list[
            ContentBlockText
            | ContentBlockImage
            | ContentBlockDocument
            | ContentBlockToolUse
            | ContentBlockToolResult
            | ContentBlockThinking
            | ContentBlockRedactedThinking
            | ContentBlockServerToolUse
            | ContentBlockWebSearchToolResult
            | ContentBlockWebFetchToolResult
            | ContentBlockToolReference
            | ContentBlockToolAddition
            | ContentBlockToolDefinition
            | ContentBlockToolSearchToolResult
            | ContentBlockMcpToolUse
            | ContentBlockMcpToolResult
            | ContentBlockGeneric
        ]
    )
    reasoning_content: str | None = None


class Tool(_AnthropicBlockBase):
    name: str
    type: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class ThinkingConfig(BaseModel):
    enabled: bool | None = True
    type: str | None = None
    budget_tokens: int | None = None


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    thinking: ThinkingConfig | None = None
    context_management: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    extra_body: dict[str, Any] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)


class TokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    original_model: str | None = Field(default=None, exclude=True)
    resolved_provider_model: str | None = Field(default=None, exclude=True)
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None
    context_management: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    betas: list[str] | None = Field(default=None, exclude=True)


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: list[
        ContentBlockText
        | ContentBlockToolUse
        | ContentBlockThinking
        | ContentBlockRedactedThinking
        | dict[str, Any]
    ]
    type: Literal["message"] = "message"
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: Usage

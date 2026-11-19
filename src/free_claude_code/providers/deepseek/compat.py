"""DeepSeek Anthropic-to-OpenAI chat request policy."""

from collections.abc import Mapping
from typing import Any

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import (
    ReasoningReplayMode,
    serialize_tool_result_content,
)
from free_claude_code.core.anthropic.content import get_block_attr, get_block_type
from free_claude_code.core.anthropic.models import (
    ContentBlockText,
    Message,
    MessagesRequest,
)
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

DEEPSEEK_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="DEEPSEEK",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
)

_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_FORWARDABLE_ATTACHMENT_BLOCK_TYPES = frozenset({"image"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}
_OMITTED_ATTACHMENT_TEXT_BLOCK = ContentBlockText.model_construct(
    type="text", text=_OMITTED_ATTACHMENT_TEXT
)

_ASSISTANT_TOOL_USE_TYPES = frozenset({"tool_use", "mcp_tool_use"})
_USER_TOOL_RESULT_TYPES = frozenset(
    {"tool_result", "mcp_tool_result", "tool_search_tool_result"}
)


def _is_vision_capable_model(model: str | None) -> bool:
    """True when the DeepSeek model accepts OpenAI image parts.

    DeepSeek's vision models (e.g. ``deepseek-v4-flash-vision-exp``) support
    image inputs natively. Document blocks are still stripped because the
    shared OpenAI conversion path has no document parts.
    """
    if not model:
        return False
    return "vision" in str(model).rsplit("/", 1)[-1].lower()


def build_deepseek_request_body(
    request_data: MessagesRequest, *, reasoning: ReasoningPolicy
) -> dict:
    """Build a DeepSeek Chat Completions body from an Anthropic request."""
    logger.debug(
        "DEEPSEEK_REQUEST: chat build model={} msgs={}",
        request_data.model,
        len(request_data.messages),
    )

    vision_capable = _is_vision_capable_model(request_data.model)
    messages = _strip_unsupported_attachment_blocks_models(
        request_data.messages, allow_attachments=vision_capable
    )
    _validate_deepseek_request(request_data, messages, allow_attachments=vision_capable)
    tool_choice = _downgrade_forced_tool_choice_value(request_data.tool_choice)
    output_config = request_data.output_config
    context_management = request_data.context_management

    has_tool_history = _has_tool_history_models(messages)
    has_replayable_tool_thinking = _all_tool_calls_have_replayable_thinking_models(
        messages
    )
    unsafe_tool_followup = has_tool_history and not has_replayable_tool_thinking
    effective_reasoning = reasoning
    tool_count = len(request_data.tools) if request_data.tools else 0
    if reasoning.control is not ReasoningControl.OFF:
        if unsafe_tool_followup:
            logger.debug(
                "DEEPSEEK_REQUEST: disabling thinking for tool follow-up without "
                "replayable thinking model={} msgs={} tools={}",
                request_data.model,
                len(messages),
                tool_count,
            )
            output_config, context_management = _clean_deepseek_thinking_hints(
                output_config, context_management
            )
            effective_reasoning = ReasoningPolicy.off()
        elif has_tool_history:
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for tool follow-up with "
                "replayable thinking model={} msgs={} tools={}",
                request_data.model,
                len(messages),
                tool_count,
            )
        elif request_data.tools or tool_choice:
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for initial tool request "
                "model={} msgs={} tools={}",
                request_data.model,
                len(messages),
                tool_count,
            )

    sanitized_messages = _sanitize_and_normalize_messages_models(messages)

    sanitized_request = MessagesRequest.model_construct(
        model=request_data.model,
        original_model=request_data.original_model,
        resolved_provider_model=request_data.resolved_provider_model,
        max_tokens=request_data.max_tokens,
        messages=sanitized_messages,
        system=request_data.system,
        stop_sequences=request_data.stop_sequences,
        stream=request_data.stream,
        temperature=request_data.temperature,
        top_p=request_data.top_p,
        top_k=request_data.top_k,
        metadata=request_data.metadata,
        tools=request_data.tools,
        tool_choice=tool_choice,
        parallel_tool_calls=request_data.parallel_tool_calls,
        thinking=request_data.thinking,
        context_management=context_management,
        output_config=output_config,
        mcp_servers=request_data.mcp_servers,
        extra_body=request_data.extra_body,
        betas=request_data.betas,
    )
    body = build_openai_chat_request_body(
        sanitized_request,
        reasoning=effective_reasoning,
        policy=DEEPSEEK_REQUEST_POLICY,
        postprocessors=(_apply_deepseek_chat_extras,),
    )
    if "max_tokens" not in body or body.get("max_tokens") is None:
        body["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def sanitize_deepseek_messages_for_openai(messages: Any) -> Any:
    """Keep only DeepSeek-required reasoning history on assistant tool calls."""
    if not isinstance(messages, list):
        return messages

    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue
        replay_tool_reasoning = _assistant_has_tool_use(message)
        new_msg = dict(message)
        if not replay_tool_reasoning:
            new_msg.pop("reasoning_content", None)
        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(new_msg)
            continue

        filtered = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and (
                    block.get("type") == "redacted_thinking"
                    or (block.get("type") == "thinking" and not replay_tool_reasoning)
                )
            )
        ]
        new_msg["content"] = filtered or ""
        sanitized.append(new_msg)
    return sanitized


def _copy_block_with_content(block: Any, content: Any) -> Any:
    if isinstance(block, dict):
        copied = dict(block)
        copied["content"] = content
        return copied
    if hasattr(block, "model_copy"):
        return block.model_copy(update={"content": content}, deep=False)
    return block


def _strip_unsupported_attachment_blocks_models(
    messages: list[Message], *, allow_attachments: bool = False
) -> list[Message]:
    stripped: list[Message] = []
    top_level_dropped: dict[str, int] = {}
    nested_dropped: dict[str, int] = {}
    placeholder_replacements = 0
    any_changed = False

    for message in messages:
        content = message.content
        if not isinstance(content, list):
            stripped.append(message)
            continue

        new_content: list[Any] = []
        message_changed = False
        message_dropped_attachment = False
        for block in content:
            btype = get_block_type(block)
            if btype in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                if allow_attachments and btype in _FORWARDABLE_ATTACHMENT_BLOCK_TYPES:
                    new_content.append(block)
                    continue
                top_level_dropped[btype] = top_level_dropped.get(btype, 0) + 1
                message_dropped_attachment = True
                message_changed = True
                continue
            if btype == "tool_result":
                inner = get_block_attr(block, "content")
                if isinstance(inner, list):
                    filtered_inner: list[Any] = []
                    inner_changed = False
                    for sub in inner:
                        sub_type = get_block_type(sub)
                        if sub_type in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                            if (
                                allow_attachments
                                and sub_type in _FORWARDABLE_ATTACHMENT_BLOCK_TYPES
                            ):
                                filtered_inner.append(sub)
                                continue
                            nested_dropped[sub_type] = (
                                nested_dropped.get(sub_type, 0) + 1
                            )
                            inner_changed = True
                            continue
                        filtered_inner.append(sub)
                    if inner_changed:
                        if not filtered_inner:
                            filtered_inner = [_OMITTED_ATTACHMENT_BLOCK]
                            placeholder_replacements += 1
                        new_content.append(
                            _copy_block_with_content(block, filtered_inner)
                        )
                        message_changed = True
                        continue
            new_content.append(block)

        if not new_content and message_dropped_attachment:
            new_content = [_OMITTED_ATTACHMENT_TEXT_BLOCK]
            placeholder_replacements += 1
            message_changed = True

        if message_changed:
            any_changed = True
            stripped.append(
                Message.model_construct(
                    role=message.role,
                    content=new_content,
                    reasoning_content=message.reasoning_content,
                )
            )
        else:
            stripped.append(message)

    if top_level_dropped or nested_dropped:
        logger.warning(
            "DEEPSEEK_REQUEST: stripped unsupported attachment blocks "
            "(top_level={} nested_in_tool_result={} placeholder_tool_results={}). "
            "DeepSeek has no vision/document support; the model will not see this content.",
            dict(top_level_dropped),
            dict(nested_dropped),
            placeholder_replacements,
        )
    return stripped if any_changed else messages


def _is_server_listed_tool(tool: Any) -> bool:
    name = (get_block_attr(tool, "name") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = get_block_attr(tool, "type")
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(
    blocks: Any, *, where: str, allow_attachments: bool = False
) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        btype = get_block_type(block)
        if btype is None:
            continue
        if btype in _UNSUPPORTED_MESSAGE_BLOCK_TYPES:
            if allow_attachments and btype in _FORWARDABLE_ATTACHMENT_BLOCK_TYPES:
                continue
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result":
            inner = get_block_attr(block, "content")
            if inner is not None:
                _walk_block_list_for_unsupported(
                    inner,
                    where=f"{where} (tool_result content)",
                    allow_attachments=allow_attachments,
                )


def _validate_deepseek_request(
    request_data: MessagesRequest,
    messages: list[Message],
    *,
    allow_attachments: bool = False,
) -> None:
    if request_data.mcp_servers:
        raise InvalidRequestError("DeepSeek does not support mcp_servers on requests.")

    for tool in request_data.tools or ():
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(messages):
        if isinstance(message.content, list):
            _walk_block_list_for_unsupported(
                message.content,
                where=f"messages[{i}].content",
                allow_attachments=allow_attachments,
            )

    if isinstance(request_data.system, list):
        _walk_block_list_for_unsupported(
            request_data.system, where="system", allow_attachments=allow_attachments
        )


def _has_tool_history_blocks_model(message: Message) -> bool:
    role = message.role
    content = message.content
    if not isinstance(content, list):
        return False

    for block in content:
        btype = get_block_type(block)
        if role == "assistant" and btype in _ASSISTANT_TOOL_USE_TYPES:
            return True
        if role == "user" and btype in _USER_TOOL_RESULT_TYPES:
            return True
    return False


def _has_replayable_thinking_before_tool_use_model(message: Message) -> bool:
    if message.role != "assistant":
        return False
    content = message.content
    if not isinstance(content, list):
        return False

    has_thinking = isinstance(message.reasoning_content, str)
    for block in content:
        btype = get_block_type(block)
        if btype == "thinking" and isinstance(get_block_attr(block, "thinking"), str):
            has_thinking = True
            continue
        if btype in _ASSISTANT_TOOL_USE_TYPES:
            return has_thinking
    return False


def _assistant_has_tool_use(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in _ASSISTANT_TOOL_USE_TYPES
        for block in content
    )


def _assistant_has_tool_use_model(message: Message) -> bool:
    if message.role != "assistant":
        return False
    content = message.content
    return isinstance(content, list) and any(
        get_block_type(block) in _ASSISTANT_TOOL_USE_TYPES for block in content
    )


def _has_tool_history_models(messages: list[Message]) -> bool:
    return any(_has_tool_history_blocks_model(message) for message in messages)


def _all_tool_calls_have_replayable_thinking_models(messages: list[Message]) -> bool:
    found_tool_call = False
    for message in messages:
        if not _assistant_has_tool_use_model(message):
            continue
        found_tool_call = True
        if not _has_replayable_thinking_before_tool_use_model(message):
            return False
    return found_tool_call


def _clean_deepseek_thinking_hints(
    output_config: dict[str, Any] | None,
    context_management: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    cleaned_output = output_config
    if isinstance(output_config, dict) and "effort" in output_config:
        tmp = dict(output_config)
        tmp.pop("effort", None)
        cleaned_output = tmp or None

    cleaned_ctx = context_management
    if isinstance(context_management, dict):
        edits = context_management.get("edits")
        if isinstance(edits, list):
            filtered_edits = [
                edit
                for edit in edits
                if not (
                    isinstance(edit, dict)
                    and isinstance(edit.get("type"), str)
                    and edit["type"].startswith("clear_thinking_")
                )
            ]
            if len(filtered_edits) != len(edits):
                tmp_ctx = dict(context_management)
                if filtered_edits:
                    tmp_ctx["edits"] = filtered_edits
                    cleaned_ctx = tmp_ctx
                else:
                    tmp_ctx.pop("edits", None)
                    cleaned_ctx = tmp_ctx or None

    return cleaned_output, cleaned_ctx


def _sanitize_and_normalize_messages_models(
    messages: list[Message],
) -> list[Message]:
    result: list[Message] = []
    any_changed = False

    for message in messages:
        content = message.content
        reasoning_content = message.reasoning_content
        changed = False

        if message.role == "assistant":
            replay_tool_reasoning = _assistant_has_tool_use_model(message)
            if not replay_tool_reasoning and reasoning_content is not None:
                reasoning_content = None
                changed = True
            if isinstance(content, list):
                filtered: list[Any] = []
                for block in content:
                    btype = get_block_type(block)
                    if btype == "redacted_thinking" or (
                        btype == "thinking" and not replay_tool_reasoning
                    ):
                        changed = True
                        continue
                    if btype in _USER_TOOL_RESULT_TYPES:
                        raw_inner = get_block_attr(block, "content")
                        normalized_inner = serialize_tool_result_content(raw_inner)
                        if normalized_inner is not raw_inner:
                            filtered.append(
                                _copy_block_with_content(block, normalized_inner)
                            )
                            changed = True
                            continue
                    filtered.append(block)
                new_content: Any = filtered or ""
                if not filtered and content != "":
                    changed = True
            else:
                new_content = content
        else:
            if isinstance(content, list):
                new_blocks: list[Any] = []
                for block in content:
                    if get_block_type(block) in _USER_TOOL_RESULT_TYPES:
                        raw_inner = get_block_attr(block, "content")
                        normalized_inner = serialize_tool_result_content(raw_inner)
                        if normalized_inner is not raw_inner:
                            new_blocks.append(
                                _copy_block_with_content(block, normalized_inner)
                            )
                            changed = True
                            continue
                    new_blocks.append(block)
                new_content = new_blocks if changed else content
            else:
                new_content = content

        if changed:
            any_changed = True
            result.append(
                Message.model_construct(
                    role=message.role,
                    content=new_content,
                    reasoning_content=reasoning_content,
                )
            )
        else:
            result.append(message)

    return result if any_changed else messages


def _downgrade_forced_tool_choice_value(
    tool_choice: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "tool" or not isinstance(
        tool_choice.get("name"), str
    ):
        return tool_choice
    logger.debug(
        "DEEPSEEK_REQUEST: downgrading forced tool_choice to auto for unsupported "
        "native request shape tool={}",
        tool_choice["name"],
    )
    return {"type": "auto"}


def _apply_deepseek_chat_extras(
    body: dict[str, Any], _request_data: MessagesRequest, policy: ReasoningPolicy
) -> None:
    extra_body = body.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return
    if policy.control is ReasoningControl.OFF:
        extra_body["thinking"] = {"type": "disabled"}
        return
    if policy.effort in {ReasoningEffort.XHIGH, ReasoningEffort.MAX}:
        body["reasoning_effort"] = "max"
    elif policy.effort is not None:
        body["reasoning_effort"] = "high"
    elif policy.requests_reasoning:
        extra_body["thinking"] = {"type": "enabled"}

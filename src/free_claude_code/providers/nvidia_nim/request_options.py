"""NVIDIA NIM request option injection."""

from typing import Any

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic import ReasoningReplayMode, set_if_not_none
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

from .tool_schema import sanitize_nim_tool_schemas

NIM_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="NIM",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
)


def build_nim_request_body(
    request_data: MessagesRequest, nim: NimSettings, *, reasoning: ReasoningPolicy
) -> dict[str, Any]:
    """Build OpenAI-format request body from Anthropic request plus NIM settings."""
    return build_openai_chat_request_body(
        request_data,
        reasoning=reasoning,
        policy=NIM_REQUEST_POLICY,
        postprocessors=(
            lambda body, request, policy: apply_nim_request_options(
                body,
                request,
                policy,
                nim=nim,
            ),
        ),
    )


def apply_nim_request_options(
    body: dict[str, Any],
    request_data: MessagesRequest,
    reasoning: ReasoningPolicy,
    *,
    nim: NimSettings,
) -> None:
    """Apply NIM schema repairs and configured request defaults."""
    sanitize_nim_tool_schemas(body)

    max_tokens = body.get("max_tokens") or request_data.max_tokens
    if isinstance(max_tokens, int) and nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    if body.get("temperature") is None and nim.temperature is not None:
        body["temperature"] = nim.temperature
    if body.get("top_p") is None and nim.top_p is not None:
        body["top_p"] = nim.top_p

    if "stop" not in body and nim.stop:
        body["stop"] = nim.stop

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    if body.get("tools"):
        body["parallel_tool_calls"] = (
            nim.parallel_tool_calls
            if request_data.parallel_tool_calls is None
            else request_data.parallel_tool_calls
        )

    extra_body: dict[str, Any] = {}
    request_extra = request_data.extra_body
    if request_extra:
        extra_body.update(request_extra)
    for key in (
        "reasoning",
        "reasoning_budget",
        "reasoning_effort",
        "reasoning_tokens",
        "thinking",
        "thinking_budget_tokens",
    ):
        extra_body.pop(key, None)
    request_template_kwargs = extra_body.get("chat_template_kwargs")
    if isinstance(request_template_kwargs, dict):
        request_template_kwargs = dict(request_template_kwargs)
        for key in ("thinking", "enable_thinking", "reasoning_budget"):
            request_template_kwargs.pop(key, None)
        if request_template_kwargs:
            extra_body["chat_template_kwargs"] = request_template_kwargs
        else:
            extra_body.pop("chat_template_kwargs", None)

    model_name = str(body.get("model") or request_data.model or "").strip().lower()
    is_nim_hybrid_reasoning = _is_nim_hybrid_reasoning_model(model_name)

    if (
        reasoning.control is ReasoningControl.OFF
        or reasoning.requests_reasoning
        or is_nim_hybrid_reasoning
    ):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if isinstance(chat_template_kwargs, dict):
            enabled = (
                reasoning.control is not ReasoningControl.OFF
                and reasoning.requests_reasoning
            )
            chat_template_kwargs["thinking"] = enabled
            chat_template_kwargs["enable_thinking"] = enabled
            if enabled:
                budget = reasoning.numeric_budget_tokens
                if budget is None and (
                    (
                        request_data.thinking is not None
                        and request_data.thinking.type == "adaptive"
                    )
                    or is_nim_hybrid_reasoning
                ):
                    # An invented 1k-2k budget is spent before the answer starts.
                    budget = ReasoningEffort.MINIMAL.budget_tokens
                if budget is not None:
                    chat_template_kwargs["reasoning_budget"] = budget
            elif is_nim_hybrid_reasoning:
                off_budget = 16
                if isinstance(max_tokens, int) and max_tokens > 0:
                    off_budget = min(off_budget, max(1, max_tokens // 2))
                chat_template_kwargs["reasoning_budget"] = off_budget

    req_top_k = request_data.top_k
    top_k = req_top_k if req_top_k is not None else nim.top_k
    _set_extra(extra_body, "top_k", top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)

    if extra_body:
        body["extra_body"] = extra_body


_NIM_HYBRID_REASONING_PREFIXES: tuple[str, ...] = (
    "stepfun-ai/",
    "minimaxai/",
    "mistralai/mistral-large-4",
    "meta/muse-",
    "deepseek-ai/",
    "z-ai/",
    "nvidia/nemotron-",
    "moonshotai/",
    "qwen/",
)


def _is_nim_hybrid_reasoning_model(model_name: str) -> bool:
    """Return whether a NIM model defaults to unbounded thinking if omitted."""
    return model_name.startswith(_NIM_HYBRID_REASONING_PREFIXES)


def _set_extra(
    extra_body: dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value

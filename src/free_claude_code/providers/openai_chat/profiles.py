"""Declarative profiles for ordinary OpenAI-compatible providers."""

from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy

from .base_url import openai_v1_base_url
from .extra_body import (
    validate_extra_body_does_not_override_reasoning_fields,
)
from .reasoning import (
    NO_REASONING,
    NamedEffortReasoning,
    ReasoningEncoder,
    ThinkingObjectReasoning,
)
from .request_policy import OpenAIChatPostprocessor, OpenAIChatRequestPolicy

_LOW_MEDIUM_HIGH = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "high"),
    (ReasoningEffort.MAX, "high"),
)
_LOW_TO_MAX = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "max"),
    (ReasoningEffort.MAX, "max"),
)


@dataclass(frozen=True, slots=True)
class OpenAIChatProfile:
    """Immutable transport and reasoning behavior for one provider."""

    request_policy: OpenAIChatRequestPolicy
    reasoning: ReasoningEncoder
    postprocessors: tuple[OpenAIChatPostprocessor, ...] = ()
    model_ids_are_routable: bool = True
    normalize_base_url: bool = False
    reasoning_delta_field: Literal["reasoning_content", "reasoning"] = (
        "reasoning_content"
    )
    structured_reasoning_details: bool = False
    user_agent: str | None = None

    @property
    def provider_name(self) -> str:
        return self.request_policy.provider_name

    def base_url(self, configured: str) -> str:
        return openai_v1_base_url(configured) if self.normalize_base_url else configured

    def reasoning_delta(self, delta: Any) -> str | None:
        value = getattr(delta, self.reasoning_delta_field, None)
        return value if isinstance(value, str) else None

    def apply_reasoning(
        self,
        body: dict[str, Any],
        _request: MessagesRequest,
        policy: ReasoningPolicy,
    ) -> None:
        self.reasoning.encode(body, policy)

    @property
    def request_postprocessors(self) -> tuple[OpenAIChatPostprocessor, ...]:
        return (*self.postprocessors, self.apply_reasoning)


def _policy(
    provider_name: str,
    replay: ReasoningReplayMode,
    **kwargs: Any,
) -> OpenAIChatRequestPolicy:
    return OpenAIChatRequestPolicy(
        provider_name=provider_name,
        reasoning_replay=replay,
        **kwargs,
    )


OPENAI_CHAT_PROFILES: dict[str, OpenAIChatProfile] = {
    "huggingface": OpenAIChatProfile(
        _policy(
            "HUGGINGFACE",
            ReasoningReplayMode.DISABLED,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        NO_REASONING,
    ),
    "kimi": OpenAIChatProfile(
        _policy(
            "KIMI",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Kimi Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ThinkingObjectReasoning(
            enabled={"type": "enabled"},
            disabled={"type": "disabled"},
        ),
    ),
    "groq": OpenAIChatProfile(
        _policy(
            "GROQ",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            max_tokens_field="max_completion_tokens",
            strip_message_names=True,
            unsupported_body_keys=frozenset({"logprobs", "logit_bias", "top_logprobs"}),
            normalize_n_to_one=True,
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            disabled_value="none",
            enabled_value="medium",
        ),
    ),
    "zai": OpenAIChatProfile(
        _policy(
            "ZAI",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Z.ai Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ThinkingObjectReasoning(
            enabled={"type": "enabled", "clear_thinking": False},
            disabled={"type": "disabled"},
        ),
    ),
    "ollama": OpenAIChatProfile(
        _policy(
            "OLLAMA",
            ReasoningReplayMode.REASONING,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
        normalize_base_url=True,
        reasoning_delta_field="reasoning",
    ),
}

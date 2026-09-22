"""OpenRouter provider implementation."""

from collections.abc import Mapping

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import extract_tool_capable_model_infos
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ReasoningObject,
    apply_reasoning_details_replay,
    validate_extra_body_does_not_override_canonical_fields,
)

_OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://claude.ai/code",
    "X-Title": "Claude Code",
}

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENROUTER",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        default_headers: Mapping[str, str] | None = None,
    ):
        headers = dict(_OPENROUTER_DEFAULT_HEADERS)
        if default_headers:
            headers.update(default_headers)
        super().__init__(
            config,
            profile=_PROFILE,
            admission=admission,
            default_headers=headers,
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Advertise OpenRouter tool models with reasoning capability metadata."""
        payload = await self._list_models_payload()
        return extract_tool_capable_model_infos(
            payload, provider_name=self._provider_name
        )

    def _provider_failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Translate OpenRouter image-unsupported 404 into non-retryable 400."""
        if getattr(error, "status_code", None) != 404:
            return None
        body = getattr(error, "body", None)
        if isinstance(body, Mapping) and "error" in body:
            if any(field in body for field in ("code", "message", "metadata")):
                return None
            body = body["error"]
        if not isinstance(body, Mapping):
            return None
        code = body.get("code")
        if not isinstance(code, int) or code != 404:
            return None
        if not isinstance(body.get("message"), str):
            return None
        metadata = body.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("failed_routing_step") != "Filter by Image Support"
        ):
            return None
        return ExecutionFailure(
            FailureKind.INVALID_REQUEST,
            400,
            "No OpenRouter endpoint for this request supports image input. "
            "Remove the image or choose an image-capable model.",
            False,
        )


_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    ReasoningObject(tuple((effort, effort.value) for effort in ReasoningEffort)),
    postprocessors=(apply_reasoning_details_replay,),
    structured_reasoning_details=True,
)

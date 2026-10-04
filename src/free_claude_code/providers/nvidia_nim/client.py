"""NVIDIA NIM provider implementation."""

import json
import re
from collections.abc import Mapping
from typing import Any

import httpx
import httpx2
import openai
from loguru import logger

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    context_window_exceeded_provider_failure,
    overloaded_provider_failure,
)
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
)

from .native_tool_stream import normalize_nim_native_tool_stream
from .request_options import NIM_REQUEST_POLICY, build_nim_request_body
from .retry import (
    _strip_chat_template_fields,
    _strip_message_reasoning_content,
    _strip_reasoning_budget_fields,
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
)
from .tool_schema import (
    body_without_nim_tool_argument_aliases,
    nim_tool_argument_aliases_from_body,
)

_DEGRADED_FUNCTION_STATE = "degraded function cannot be invoked"
_NEGATIVE_MAX_TOKENS_PATTERN = re.compile(
    r"\bmax_tokens must be at least 1,\s*got\s+-[1-9]\d*\b",
    re.IGNORECASE,
)
_OPAQUE_STREAM_INTERNAL_ERROR_ATTEMPTS = 2
_PROFILE = OpenAIChatProfile(
    NIM_REQUEST_POLICY,
    NO_REASONING,
)


class NvidiaNimProvider(OpenAIChatProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        nim_settings: NimSettings,
        admission: ProviderAdmissionController,
    ):
        super().__init__(
            config,
            profile=_PROFILE,
            admission=admission,
        )
        self._nim_settings = nim_settings
        self._unsupported_reasoning_content_models: set[str] = set()
        self._unsupported_chat_template_models: set[str] = set()
        self._unsupported_reasoning_budget_models: set[str] = set()

    def _build_request_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_nim_request_body(
            request,
            self._nim_settings,
            reasoning=reasoning,
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        body = body_without_nim_tool_argument_aliases(body)
        model = _model_id(body)
        if model in self._unsupported_reasoning_content_models:
            body = dict(body)
            messages = body.get("messages")
            if isinstance(messages, list):
                body["messages"] = [
                    dict(msg) if isinstance(msg, dict) else msg for msg in messages
                ]
            _strip_message_reasoning_content(body)
        if model in self._unsupported_reasoning_budget_models:
            extra = body.get("extra_body")
            if isinstance(extra, dict):
                body = dict(body)
                extra = dict(extra)
                body["extra_body"] = extra
                _strip_reasoning_budget_fields(extra)
                if not extra:
                    body.pop("extra_body", None)
        if model in self._unsupported_chat_template_models:
            extra = body.get("extra_body")
            if isinstance(extra, dict):
                body = dict(body)
                extra = dict(extra)
                body["extra_body"] = extra
                _strip_chat_template_fields(extra)
                if not extra:
                    body.pop("extra_body", None)
        return body

    def _normalize_stream(self, stream: Any, _body: Mapping[str, Any]) -> Any:
        """Repair model-native MiniMax tool markup leaked by NVIDIA NIM."""
        return normalize_nim_native_tool_stream(stream, _body)

    def _rotate_on_permission_denied(self) -> bool:
        """Treat NIM 403s as model/request denials, not pool-key failures."""
        return False

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        status_code = getattr(error, "status_code", None)
        bad_request_like = isinstance(error, openai.BadRequestError) or (
            status_code == 400
        )

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()
        model = _model_id(body)

        if _is_reasoning_budget_rejection(error_text) and (
            bad_request_like or status_code == 500
        ):
            self._unsupported_reasoning_budget_models.add(model)
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning budget after upstream rejection"
            )
            return retry_body

        opaque_internal_error = _is_opaque_internal_server_error(error)
        if (bad_request_like and "chat_template" in error_text) or (
            opaque_internal_error and _has_chat_template_controls(body)
        ):
            retry_body = clone_body_without_chat_template(body)
            if retry_body is None:
                return None
            if bad_request_like and "chat_template" in error_text:
                self._unsupported_chat_template_models.add(model)
            logger.warning(
                "NIM_STREAM: retrying without chat_template controls after {}",
                "opaque 500" if opaque_internal_error else "400 error",
            )
            return retry_body

        if not bad_request_like:
            return None

        if "reasoning_content" in error_text:
            self._unsupported_reasoning_content_models.add(model)
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_content after 400 error"
            )
            return retry_body

        return None

    def _provider_failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Classify NVIDIA-specific 400/500 responses by their actual semantics."""
        if not isinstance(error, openai.BadRequestError | openai.InternalServerError):
            return None
        status = getattr(error, "status_code", None)
        if status not in (400, 500):
            return None
        bodies = _nim_error_bodies(error)
        if any(_is_context_window_exhaustion(body) for body in bodies):
            return context_window_exceeded_provider_failure()
        if isinstance(error, openai.BadRequestError) and any(
            _is_degraded_function(body) for body in bodies
        ):
            return overloaded_provider_failure()
        return None

    def _open_failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Do not re-queue a request that timed out waiting in NIM's queue.

        NIM withholds response headers while a request is queued. A timeout
        there means the queue outlasted HTTP_READ_TIMEOUT, and a retry rejoins
        at the back, so it would only time out again. It also says nothing
        about other NIM models, so it must not pause the whole provider.
        """
        if _is_queue_timeout(error):
            return ExecutionFailure(
                kind=FailureKind.TIMEOUT,
                status_code=504,
                message=(
                    "NVIDIA NIM request timed out after "
                    f"{self._config.http_read_timeout:g}s waiting in the model's "
                    "queue. It was not retried: a retry rejoins the back of the "
                    "queue. Raise HTTP_READ_TIMEOUT if this model's queue is "
                    "usually longer."
                ),
                retryable=False,
            )
        return self._provider_failure_override(error)

    def _stream_failure_override(
        self,
        error: Exception,
        *,
        attempts_started: int,
        stream_opened: bool,
        accepted: bool,
    ) -> ExecutionFailure | None:
        """Bound opaque pre-output NIM failures without changing the request body."""
        override = self._provider_failure_override(error)
        if override is not None:
            return override
        if (
            stream_opened
            and not accepted
            and attempts_started >= _OPAQUE_STREAM_INTERNAL_ERROR_ATTEMPTS
            and _is_opaque_internal_server_error(error)
        ):
            return ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=500,
                message="NVIDIA NIM returned an internal server error.",
                retryable=False,
            )
        return None


def _is_queue_timeout(error: Exception) -> bool:
    """Whether NIM never sent response headers, i.e. the request stayed queued.

    A connect timeout never reached the queue, so it stays retryable.
    """
    if isinstance(error, openai.APITimeoutError):
        cause = error.__cause__
        return cause is None or isinstance(
            cause, httpx.ReadTimeout | httpx2.ReadTimeout
        )
    return isinstance(error, httpx.ReadTimeout | httpx2.ReadTimeout)


def _nim_error_bodies(error: Exception) -> tuple[Mapping[str, Any], ...]:
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return ()
    nested = body.get("error")
    if isinstance(nested, Mapping):
        return body, nested
    return (body,)


def _model_id(body: Mapping[str, Any]) -> str:
    """Return a model identifier suitable for a capability-cache key."""
    model = body.get("model")
    return model if isinstance(model, str) else ""


def _is_context_window_exhaustion(body: Mapping[str, Any]) -> bool:
    message = body.get("message")
    return (
        body.get("param") == "max_tokens"
        and isinstance(message, str)
        and _NEGATIVE_MAX_TOKENS_PATTERN.search(message) is not None
    )


def _is_degraded_function(body: Mapping[str, Any]) -> bool:
    detail = body.get("detail")
    if not isinstance(detail, str):
        return False
    function_ref, separator, state = detail.lower().partition(": ")
    function_id = function_ref.removeprefix("function id ").strip(" '\"")
    return bool(
        separator
        and function_ref.startswith("function id ")
        and function_id
        and state.strip() == _DEGRADED_FUNCTION_STATE
    )


def _is_opaque_internal_server_error(error: Exception) -> bool:
    """Return whether NIM provided no request-shape correction to make."""
    status_code = getattr(error, "status_code", None)
    if isinstance(error, openai.InternalServerError) and status_code == 500:
        pass
    elif isinstance(error, openai.APIError) and status_code is None:
        bodies = _nim_error_bodies(error)
        is_500 = any(
            body.get("code") in (500, "500")
            or body.get("type") == "internal_server_error"
            for body in bodies
        )
        if not is_500:
            return False
    else:
        return False
    error_text = str(error)
    error_body = getattr(error, "body", None)
    if error_body is not None:
        error_text = f"{error_text} {json.dumps(error_body, default=str)}"
    error_text = error_text.lower()
    return (
        not _is_reasoning_budget_rejection(error_text)
        and "reasoning_content" not in error_text
        and "chat_template" not in error_text
    )


def _has_chat_template_controls(body: Mapping[str, Any]) -> bool:
    """Return whether the request carries optional NIM chat-template controls."""
    extra_body = body.get("extra_body")
    return isinstance(extra_body, Mapping) and (
        "chat_template" in extra_body or "chat_template_kwargs" in extra_body
    )


def _is_reasoning_budget_rejection(error_text: str) -> bool:
    """Return whether NIM rejected optional thinking budget control."""
    if "reasoning_budget" in error_text:
        return True
    if "thinking_token_budget" not in error_text:
        return False
    return "reasoning_config" in error_text or (
        "not yet supported" in error_text and "v2 model runner" in error_text
    )

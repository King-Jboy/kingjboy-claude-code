"""LM Studio provider implementation (OpenAI-compatible chat completions).

Switched from LM Studio's native Anthropic Messages endpoint (2026-07-04):
the newer ``/v1/messages`` path renders Claude Code conversations through the
model's jinja chat template with strict role-alternation rules and a fragile
``[TOOL_CALLS]`` parser — observed leaking control tokens into tool names
(``[TOOL_CALLS]Read``) and dumping whole tool calls into text
(``Read[ARGS]{...}``), which ends agent runs silently. The OpenAI
``/v1/chat/completions`` path is LM Studio's mature parsing route, and fcc's
OpenAI provider layers its own tool-call assembly, think-tag parsing, and
heuristic recovery on top.
"""

import asyncio
import time

import httpx
from loguru import logger

from free_claude_code.core.anthropic import ReasoningReplayMode, get_token_count
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    context_window_exceeded_provider_failure,
)
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)

_PROFILE = OpenAIChatProfile(
    OpenAIChatRequestPolicy(
        provider_name="LMSTUDIO",
        reasoning_replay=ReasoningReplayMode.DISABLED,
    ),
    NamedEffortReasoning(
        (
            (ReasoningEffort.MINIMAL, "low"),
            (ReasoningEffort.LOW, "low"),
            (ReasoningEffort.MEDIUM, "medium"),
            (ReasoningEffort.HIGH, "high"),
            (ReasoningEffort.XHIGH, "high"),
            (ReasoningEffort.MAX, "high"),
        ),
        disabled_value="none",
        enabled_value="high",
        budget_field="reasoning_tokens",
    ),
)


class LMStudioProvider(OpenAIChatProvider):
    """LM Studio via its OpenAI-compatible chat completions endpoint."""

    # LM Studio truncates the stream silently (no terminal event) when the
    # prompt exceeds the loaded context. Refuse clearly over-budget prompts
    # up front as a context-window failure so protocol adapters can tell their
    # clients to compact/retry instead of letting the stream die silently.
    _CONTEXT_CACHE_TTL_S = 30.0

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        super().__init__(
            config,
            profile=_PROFILE,
            admission=admission,
        )
        self._loaded_context_cache: tuple[float, int | None] = (0.0, None)
        self._context_refresh: asyncio.Task[None] | None = None

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        super().preflight_stream(request, reasoning=reasoning)
        self._preflight_context_budget(request)

    def _preflight_context_budget(self, request: MessagesRequest) -> None:
        loaded_context = self._loaded_context_length()
        if loaded_context is None:
            return
        estimate = get_token_count(
            request.messages,
            request.system,
            request.tools,
        )
        # The estimate is cl100k-based and undercounts local tokenizers
        # (observed ~8% low vs devstral); a request above 90% of the loaded
        # context is already past where client-side compaction should have
        # fired, and letting it through risks a silent LM Studio truncation.
        budget = int(loaded_context * 0.9)
        if estimate > budget:
            raise context_window_exceeded_provider_failure(
                f"Estimated provider input ({estimate} tokens) exceeds the safe "
                f"LM Studio context budget ({budget} tokens; 90% of loaded "
                f"context {loaded_context})."
            )

    def _loaded_context_length(self) -> int | None:
        """Best-effort loaded context length from LM Studio's REST API, cached.

        Preflight runs synchronously on the event loop, so a stale cache is
        refreshed in the background and this call never waits on the network.
        """
        cached_at, cached_value = self._loaded_context_cache
        stale = time.monotonic() - cached_at >= self._CONTEXT_CACHE_TTL_S
        if stale and (self._context_refresh is None or self._context_refresh.done()):
            self._context_refresh = asyncio.get_running_loop().create_task(
                self._refresh_loaded_context()
            )
        return cached_value

    async def _refresh_loaded_context(self) -> None:
        value = await self._fetch_loaded_context()
        self._loaded_context_cache = (time.monotonic(), value)

    async def _fetch_loaded_context(self) -> int | None:
        value: int | None = None
        try:
            root = self._base_url
            root = root[: -len("/v1")] if root.endswith("/v1") else root
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{root}/api/v0/models")
            response.raise_for_status()
            loaded = [
                model.get("loaded_context_length")
                for model in response.json().get("data", [])
                if model.get("state") == "loaded"
                and isinstance(model.get("loaded_context_length"), int)
            ]
            # ponytail: single-model setups in practice; with several loaded
            # models the most generous ceiling still makes a valid backstop.
            value = max(loaded) if loaded else None
        except Exception as error:  # backstop only — never block the request
            logger.debug(
                "LMSTUDIO context preflight unavailable: {}", type(error).__name__
            )
            value = None
        return value

    async def cleanup(self) -> None:
        if self._context_refresh is not None:
            self._context_refresh.cancel()
        await super().cleanup()

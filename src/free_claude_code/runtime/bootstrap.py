"""Single production composition root for the FCC server."""

import os
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.config.logging_config import configure_logging
from free_claude_code.config.paths import responses_state_path, server_log_path
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.openai_responses import ResponsesStore
from free_claude_code.messaging.transcription import TranscriptionService
from free_claude_code.messaging.voice import Transcriber
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.key_pool import ApiKeyPool
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.nvidia_nim.voice import NvidiaNimTranscriber
from free_claude_code.providers.openai_codex import (
    OpenAIAuthManager,
    OpenAICodexProvider,
)
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.providers.runtime.config import provider_credentials
from free_claude_code.providers.runtime.factory import create_provider

from .application import ApplicationRuntime, RestartCallback, StopCallback
from .asgi import RuntimeASGIApp
from .codex_catalog import CodexModelCatalogPublisher
from .provider_manager import ProviderRuntimeManager


def build_asgi_app(
    settings: Settings,
    restart_callback: RestartCallback | None = None,
    stop_callback: StopCallback | None = None,
) -> RuntimeASGIApp:
    """Construct the complete server application and its resource owner."""
    log_path = Path(os.getenv("LOG_FILE", server_log_path()))
    configure_logging(
        log_path,
        level=settings.log_level,
        verbose_third_party=settings.log_raw_api_payloads,
    )
    openai_auth = OpenAIAuthManager(proxy=settings.openai_proxy)
    openai_factory = partial(_create_openai_provider, auth=openai_auth)
    provider_constructor = partial(
        create_provider,
        injected_factories={"openai": openai_factory},
    )
    runtime_factory = partial(
        ProviderRuntime,
        provider_constructor=provider_constructor,
    )
    provider_manager = ProviderRuntimeManager(
        settings,
        runtime_factory=runtime_factory,
        connected_provider_ids=openai_auth.connected_provider_ids,
        model_catalog_publisher=CodexModelCatalogPublisher(),
    )
    runtime = ApplicationRuntime(
        provider_manager,
        transcriber=_create_transcriber(
            settings,
            nvidia_nim_key_pool_provider=partial(
                _current_nvidia_nim_key_pool, provider_manager
            ),
        ),
        restart_callback=restart_callback,
        stop_callback=stop_callback,
        connected_accounts={"openai": openai_auth},
    )
    services = ApiServices(
        requests=provider_manager,
        admin=runtime,
        tasks=runtime,
        responses_store=ResponsesStore(responses_state_path()),
    )
    return RuntimeASGIApp(create_app(services), runtime)


def _create_openai_provider(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
    *,
    auth: OpenAIAuthManager,
) -> BaseProvider:
    return OpenAICodexProvider(config, auth=auth, admission=admission)


async def _current_nvidia_nim_key_pool(
    provider_manager: ProviderRuntimeManager,
) -> ApiKeyPool | None:
    """Return the NIM pool from the current provider generation."""
    lease = await provider_manager.acquire()
    try:
        provider = lease.resolve_provider("nvidia_nim")
        if not isinstance(provider, NvidiaNimProvider):
            raise RuntimeError("NVIDIA NIM provider did not expose its key pool.")
        return provider.api_key_pool
    finally:
        await lease.release()


def _create_transcriber(
    settings: Settings,
    *,
    nvidia_nim_key_pool_provider: Callable[[], Awaitable[ApiKeyPool | None]] | None = None,
) -> Transcriber | None:
    if not settings.voice_note_enabled:
        return None
    if settings.whisper_device == "nvidia_nim":
        api_keys = provider_credentials(PROVIDER_CATALOG["nvidia_nim"], settings)
        if nvidia_nim_key_pool_provider is None:
            return NvidiaNimTranscriber(
                model=settings.whisper_model,
                api_keys=api_keys,
                key_rate_limit=settings.nvidia_nim_key_rate_limit,
            )
        return NvidiaNimTranscriber(
            model=settings.whisper_model,
            api_keys=api_keys,
            key_rate_limit=settings.nvidia_nim_key_rate_limit,
            key_pool_provider=nvidia_nim_key_pool_provider,
        )
    return TranscriptionService(
        model=settings.whisper_model,
        device=settings.whisper_device,
        huggingface_api_key=settings.huggingface_api_key,
    )

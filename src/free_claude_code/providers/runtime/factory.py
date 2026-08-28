"""Provider construction from declarative profiles and exceptional adapters."""

from collections.abc import Callable, Mapping

from free_claude_code.application.errors import (
    ApplicationUnavailableError,
    UnknownProviderError,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    create_openai_chat_provider,
)

from .config import build_provider_config, numeric_setting

ProviderFactory = Callable[
    [ProviderConfig, Settings, ProviderAdmissionController], BaseProvider
]

# Pooled keys multiply provider quota, but concurrency is bounded by local
# sockets and event-loop work rather than by quota. Cap the bulkhead so a large
# pool cannot open an unbounded number of simultaneous upstream streams.
#
# This ceiling, not the rate limit, is what bounds throughput for streaming
# work: a response holds its slot open for tens of seconds, so a pool serves
# roughly ``ceiling / response_seconds`` requests per second no matter how much
# quota its keys add up to. Set too low it silently strands most of a large
# pool's capacity and queues callers until they time out, which reads as a
# network error rather than as the throttle it is. Operators can retune it with
# PROVIDER_MAX_POOLED_CONCURRENCY.
MAX_POOLED_CONCURRENCY = 64


def _create_nvidia_nim(
    config: ProviderConfig,
    settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(
        config,
        nim_settings=settings.nim,
        admission=admission,
    )


def _create_open_router(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config, admission=admission)


def _create_deepseek(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config, admission=admission)


def _create_lmstudio(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config, admission=admission)


def _create_gemini(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.gemini import GeminiProvider

    return GeminiProvider(config, admission=admission)


_SPECIAL_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "gemini": _create_gemini,
}
_INJECTED_PROVIDER_IDS = {"openai"}

_profiled_ids = set(OPENAI_CHAT_PROFILES)
_special_ids = set(_SPECIAL_PROVIDER_FACTORIES)
_construction_ids = _profiled_ids | _special_ids | _INJECTED_PROVIDER_IDS
if (
    _profiled_ids & _special_ids
    or _profiled_ids & _INJECTED_PROVIDER_IDS
    or _special_ids & _INJECTED_PROVIDER_IDS
    or _construction_ids != set(PROVIDER_CATALOG)
):
    raise AssertionError(
        "Every provider must have exactly one construction owner: "
        f"profiles={_profiled_ids!r} special={_special_ids!r} "
        f"injected={_INJECTED_PROVIDER_IDS!r} catalog={set(PROVIDER_CATALOG)!r}"
    )


def create_provider(
    provider_id: str,
    settings: Settings,
    *,
    injected_factories: Mapping[str, ProviderFactory] | None = None,
) -> BaseProvider:
    """Create a provider instance for a supported provider id."""
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if descriptor is None:
        raise UnknownProviderError.for_provider(provider_id, PROVIDER_CATALOG)

    config = build_provider_config(descriptor, settings)
    # A pool serves the combined quota of its keys, so the provider-wide gate
    # admits the pooled total; otherwise it would cap the pool at one key's rate.
    pool_scale = max(1, len(config.api_keys))
    max_concurrency = config.max_concurrency
    if pool_scale > 1:
        # Concurrency is bounded by local sockets rather than by quota, so a
        # pool scales it per key - capped by the pooled ceiling, which an
        # operator may set anywhere (including below one key's figure; a
        # streaming response holds its slot for tens of seconds, so a low cap
        # trades stranded quota for socket safety by choice). Single-credential
        # providers never feel the pooled ceiling.
        pooled_ceiling = max(
            1,
            int(
                numeric_setting(
                    settings, "provider_max_pooled_concurrency", MAX_POOLED_CONCURRENCY
                )
            ),
        )
        max_concurrency = min(config.max_concurrency * pool_scale, pooled_ceiling)
    admission = ProviderAdmissionController(
        provider_name=provider_id,
        rate_limit=(config.rate_limit or 40) * pool_scale,
        rate_window=config.rate_window or 60.0,
        max_concurrency=max_concurrency,
    )
    factory = (injected_factories or {}).get(provider_id)
    if provider_id in _INJECTED_PROVIDER_IDS and factory is None:
        raise ApplicationUnavailableError(
            f"Provider {provider_id!r} is unavailable in this runtime."
        )
    factory = factory or _SPECIAL_PROVIDER_FACTORIES.get(provider_id)
    if factory is not None:
        return factory(config, settings, admission)
    return create_openai_chat_provider(provider_id, config, admission)

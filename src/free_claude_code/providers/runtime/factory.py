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


def _create_mistral(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.mistral import MistralProvider

    return MistralProvider(config, admission=admission)


def _create_kilo(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.kilo import KiloProvider

    return KiloProvider(config, admission=admission)


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


def _create_cloudflare(
    config: ProviderConfig,
    settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.cloudflare import CloudflareProvider

    return CloudflareProvider(
        config,
        account_id=settings.cloudflare_account_id,
        admission=admission,
    )


def _create_gemini(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.gemini import GeminiProvider

    return GeminiProvider(config, admission=admission)


def _create_vertex(
    config: ProviderConfig,
    settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.vertex import VertexProvider

    return VertexProvider(
        config,
        project_id=settings.vertex_project_id,
        location=settings.vertex_location,
        admission=admission,
    )


def _create_github_models(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.github_models import GitHubModelsProvider

    return GitHubModelsProvider(config, admission=admission)


_SPECIAL_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "mistral": _create_mistral,
    "kilo": _create_kilo,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "cloudflare": _create_cloudflare,
    "gemini": _create_gemini,
    "vertex": _create_vertex,
    "github_models": _create_github_models,
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
    # Each pooled key enforces its own window, so the provider-wide gate must
    # admit the pooled total; otherwise it would cap the pool at one key's rate.
    pool_scale = max(1, len(config.api_keys))
    pooled_ceiling = int(
        numeric_setting(
            settings, "provider_max_pooled_concurrency", MAX_POOLED_CONCURRENCY
        )
    )
    admission = ProviderAdmissionController(
        provider_name=provider_id,
        rate_limit=(config.rate_limit or 40) * pool_scale,
        rate_window=config.rate_window or 60.0,
        max_concurrency=min(
            config.max_concurrency * pool_scale,
            max(config.max_concurrency, pooled_ceiling),
        ),
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

"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.runtime.factory`; this module stays free of
provider implementation imports (see contract tests).

This fork carries a deliberately small catalog: the cloud providers its operator
routes to, plus local runtimes. Adding a provider back is a catalog entry, a
settings field, and a profile or factory - see git history for the shape.
"""

from dataclasses import dataclass
from enum import StrEnum

# Default upstream base URLs are owned here with the provider catalog.
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
# Moonshot Kimi OpenAI-compatible Chat Completions API.
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/v1"
# DeepSeek Chat Completions API; cache usage is reported on this endpoint.
DEEPSEEK_DEFAULT_BASE = "https://api.deepseek.com"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"
HUGGINGFACE_DEFAULT_BASE = "https://router.huggingface.co/v1"
# Z.ai GLM Coding Plan OpenAI-compatible Chat Completions API.
ZAI_DEFAULT_BASE = "https://api.z.ai/api/coding/paas/v4"
GROQ_DEFAULT_BASE = "https://api.groq.com/openai/v1"
OPENAI_CODEX_DEFAULT_BASE = "https://chatgpt.com/backend-api/codex"


class ProviderAuthKind(StrEnum):
    """How a customer makes one provider available."""

    CONFIGURATION = "configuration"
    CONNECTED_ACCOUNT = "connected_account"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
    display_name: str
    auth_kind: ProviderAuthKind = ProviderAuthKind.CONFIGURATION
    local: bool = False
    credential_env: str | None = None
    credential_url: str | None = None
    credential_attr: str | None = None
    # Optional settings attribute holding a JSON list of interchangeable keys.
    # A configured pool satisfies ``credential_attr`` on its own.
    credential_pool_attr: str | None = None
    static_credential: str | None = None
    default_base_url: str | None = None
    base_url_attr: str | None = None
    proxy_attr: str | None = None
    required_settings_attrs: tuple[str, ...] = ()
    # Published per-key request quota, when this provider states one. Rate limits
    # are a property of the provider, not of the installation, so one global
    # setting cannot serve two providers with different ceilings: pacing every
    # provider at the highest one guarantees refusals from the lowest. An
    # explicitly configured PROVIDER_RATE_LIMIT still overrides this.
    rate_limit: int | None = None
    rate_window: float | None = None

    def configuration_attrs(self) -> tuple[str, ...]:
        """Return settings fields whose non-empty values configure this provider."""
        if self.required_settings_attrs:
            return self.required_settings_attrs
        if self.credential_attr is not None:
            return (self.credential_attr,)
        if self.base_url_attr is not None:
            return (self.base_url_attr,)
        return ()


PROVIDER_CATALOG: dict[str, ProviderDescriptor] = {
    "nvidia_nim": ProviderDescriptor(
        provider_id="nvidia_nim",
        display_name="NVIDIA NIM",
        credential_env="NVIDIA_NIM_API_KEY",
        credential_url="https://build.nvidia.com/settings/api-keys",
        credential_attr="nvidia_nim_api_key",
        credential_pool_attr="nvidia_nim_api_keys",
        default_base_url=NVIDIA_NIM_DEFAULT_BASE,
        proxy_attr="nvidia_nim_proxy",
        rate_limit=40,
        rate_window=60.0,
    ),
    "openai": ProviderDescriptor(
        provider_id="openai",
        display_name="OpenAI / ChatGPT",
        auth_kind=ProviderAuthKind.CONNECTED_ACCOUNT,
        default_base_url=OPENAI_CODEX_DEFAULT_BASE,
        proxy_attr="openai_proxy",
    ),
    "open_router": ProviderDescriptor(
        provider_id="open_router",
        display_name="OpenRouter",
        credential_env="OPENROUTER_API_KEY",
        credential_url="https://openrouter.ai/keys",
        credential_attr="open_router_api_key",
        credential_pool_attr="open_router_api_keys",
        default_base_url=OPENROUTER_DEFAULT_BASE,
        proxy_attr="open_router_proxy",
        rate_limit=20,
        rate_window=60.0,
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        display_name="DeepSeek",
        credential_env="DEEPSEEK_API_KEY",
        credential_url="https://platform.deepseek.com/api_keys",
        credential_attr="deepseek_api_key",
        default_base_url=DEEPSEEK_DEFAULT_BASE,
    ),
    "huggingface": ProviderDescriptor(
        provider_id="huggingface",
        display_name="Hugging Face",
        credential_env="HUGGINGFACE_API_KEY",
        credential_url="https://huggingface.co/settings/tokens",
        credential_attr="huggingface_api_key",
        default_base_url=HUGGINGFACE_DEFAULT_BASE,
        proxy_attr="huggingface_proxy",
    ),
    "kimi": ProviderDescriptor(
        provider_id="kimi",
        display_name="Kimi",
        credential_env="KIMI_API_KEY",
        credential_url="https://platform.moonshot.cn/console/api-keys",
        credential_attr="kimi_api_key",
        default_base_url=KIMI_DEFAULT_BASE,
        proxy_attr="kimi_proxy",
    ),
    "groq": ProviderDescriptor(
        provider_id="groq",
        display_name="Groq",
        credential_env="GROQ_API_KEY",
        credential_url="https://console.groq.com/keys",
        credential_attr="groq_api_key",
        default_base_url=GROQ_DEFAULT_BASE,
        proxy_attr="groq_proxy",
    ),
    "zai": ProviderDescriptor(
        provider_id="zai",
        display_name="Z.ai",
        credential_env="ZAI_API_KEY",
        credential_attr="zai_api_key",
        default_base_url=ZAI_DEFAULT_BASE,
        proxy_attr="zai_proxy",
    ),
    "lmstudio": ProviderDescriptor(
        provider_id="lmstudio",
        display_name="LM Studio",
        static_credential="lm-studio",
        default_base_url=LMSTUDIO_DEFAULT_BASE,
        base_url_attr="lm_studio_base_url",
        proxy_attr="lmstudio_proxy",
        local=True,
    ),
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        display_name="Ollama",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        local=True,
    ),
}

# Key order: NVIDIA NIM first (README default), then the kept cloud providers,
# with the local runtimes last. ``SUPPORTED_PROVIDER_IDS`` inherits this
# insertion order for UI and error-message listing.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())

if len(set(SUPPORTED_PROVIDER_IDS)) != len(SUPPORTED_PROVIDER_IDS):
    raise AssertionError("Duplicate provider ids in PROVIDER_CATALOG key order")

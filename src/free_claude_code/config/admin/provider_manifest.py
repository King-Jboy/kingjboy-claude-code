"""Catalog-derived Admin provider fields."""

from typing import Any

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings

_PROVIDER_FIELD_OVERRIDES: dict[str, dict[str, Any]] = {
    "OPENAI_PROXY": {
        "description": (
            "Optional proxy used for OpenAI sign-in and ChatGPT Codex requests. "
            "Changing it restarts FCC."
        ),
        "restart_required": True,
    },
    "NVIDIA_NIM_API_KEY": {
        "label": "NVIDIA NIM API Key",
        "description": "Used by NVIDIA NIM chat and optional NIM voice transcription.",
    },
    "HUGGINGFACE_API_KEY": {
        "label": "Hugging Face API Key",
        "description": (
            "Hugging Face token with Inference Providers permission; also used "
            "for local Whisper model downloads when voice notes need gated models."
        ),
    },
    "ZAI_API_KEY": {
        "label": "Z.ai API Key",
        "description": "Z.ai Coding Plan API key.",
    },
    "GROQ_API_KEY": {
        "label": "Groq API Key",
        "description": (
            "GroqCloud OpenAI-compatible API key ([console.groq.com/keys]("
            "https://console.groq.com/keys)); see Groq "
            "[OpenAI compatibility docs](https://console.groq.com/docs/openai)."
        ),
    },
}


def provider_field_specs() -> tuple[dict[str, Any], ...]:
    """Return provider fields generated from the provider catalog."""

    return (
        *_credential_field_specs(),
        *_credential_pool_field_specs(),
        *_base_url_field_specs(),
        *_proxy_field_specs(),
    )


def _credential_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    seen_env_keys: set[str] = set()
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_env is None:
            continue
        if descriptor.credential_env in seen_env_keys:
            continue
        seen_env_keys.add(descriptor.credential_env)
        spec = {
            "key": descriptor.credential_env,
            "label": f"{descriptor.display_name} API Key",
            "section_id": "providers",
            "field_type": "secret",
            "settings_attr": descriptor.credential_attr,
            "secret": True,
        }
        spec.update(_PROVIDER_FIELD_OVERRIDES.get(descriptor.credential_env, {}))
        specs.append(spec)
    return tuple(specs)


def _credential_pool_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_pool_attr is None:
            continue
        specs.append(
            {
                "key": _settings_env_key(descriptor.credential_pool_attr),
                "label": f"{descriptor.display_name} API Key Pool",
                "section_id": "providers",
                "field_type": "textarea",
                "settings_attr": descriptor.credential_pool_attr,
                "secret": True,
                "description": (
                    "Optional JSON list of interchangeable API keys, for example "
                    '["key-one", "key-two"]. When two or more keys are set this '
                    "replaces the single key above: each key gets its own rate "
                    "window, and a rejected or rate-limited key is skipped "
                    "automatically."
                ),
            }
        )
    return tuple(specs)


def _base_url_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.base_url_attr is None:
            continue
        key = _settings_env_key(descriptor.base_url_attr)
        spec = {
            "key": key,
            "label": f"{descriptor.display_name} Base URL",
            "section_id": "providers",
            "settings_attr": descriptor.base_url_attr,
            "default": descriptor.default_base_url or "",
        }
        spec.update(_PROVIDER_FIELD_OVERRIDES.get(key, {}))
        specs.append(spec)
    return tuple(specs)


def _proxy_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.proxy_attr is None:
            continue
        specs.append(
            {
                "key": _settings_env_key(descriptor.proxy_attr),
                "label": f"{descriptor.display_name} Proxy",
                "section_id": "providers",
                "field_type": "secret",
                "settings_attr": descriptor.proxy_attr,
                "secret": True,
                "advanced": True,
            }
        )
    return tuple(specs)


def _settings_env_key(settings_attr: str) -> str:
    model_field = Settings.model_fields[settings_attr]
    alias = model_field.validation_alias
    return str(alias) if alias is not None else settings_attr

"""Provider configuration status for the Admin UI."""

import os
from collections.abc import Mapping
from typing import Any

from free_claude_code.config.api_key_pool import parse_api_key_pool
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)

from .manifest import FIELDS
from .sources import configured_env_files, dotenv_values_from_file

_POOL_ENV_BY_PROVIDER: dict[str, str] = {
    "nvidia_nim": "NVIDIA_NIM_API_KEYS",
    "open_router": "OPENROUTER_API_KEYS",
    "custom": "CUSTOM_API_KEYS",
}


def provider_config_status(
    state: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return provider configuration status without making network calls."""
    statuses: list[dict[str, Any]] = []
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT:
            statuses.append(
                {
                    "provider_id": provider_id,
                    "display_name": descriptor.display_name,
                    "kind": "connected_account",
                    "status": "disconnected",
                    "label": "Not connected",
                }
            )
            continue
        if descriptor.local:
            base_url = ""
            if descriptor.base_url_attr is not None:
                base_url = _value_for_settings_attr(state, descriptor.base_url_attr)
            statuses.append(
                {
                    "provider_id": provider_id,
                    "display_name": descriptor.display_name,
                    "kind": "local",
                    "status": "missing_url" if not base_url.strip() else "unknown",
                    "label": "Missing URL" if not base_url.strip() else "Not checked",
                    "base_url": base_url or descriptor.default_base_url or "",
                }
            )
            continue

        configuration_attrs = descriptor.configuration_attrs()
        missing_attrs = tuple(
            attr
            for attr in configuration_attrs
            if not _attr_has_value(state, provider_id, descriptor.credential_attr, attr)
        )
        configured = not missing_attrs
        configuration = " + ".join(
            _field_key_for_settings_attr(attr) for attr in configuration_attrs
        )
        missing_key = descriptor.credential_attr in missing_attrs
        statuses.append(
            {
                "provider_id": provider_id,
                "display_name": descriptor.display_name,
                "kind": "remote",
                "status": (
                    "configured"
                    if configured
                    else "missing_key"
                    if missing_key
                    else "missing_config"
                ),
                "label": (
                    "Configured"
                    if configured
                    else "Missing key"
                    if missing_key
                    else "Missing configuration"
                ),
                "configuration": configuration,
            }
        )
    return statuses


def _attr_has_value(
    state: Mapping[str, Mapping[str, Any]],
    provider_id: str,
    credential_attr: str | None,
    settings_attr: str,
) -> bool:
    if _value_for_settings_attr(state, settings_attr).strip():
        return True
    if settings_attr == credential_attr:
        pool_env = _POOL_ENV_BY_PROVIDER.get(provider_id)
        if pool_env and _has_configured_key_pool(state, pool_env):
            return True
    return False


def _has_configured_key_pool(
    state: Mapping[str, Mapping[str, Any]], pool_env: str
) -> bool:
    raw = str(state.get(pool_env, {}).get("value", ""))
    if not raw.strip():
        raw = os.environ.get(pool_env, "")
    if not raw.strip():
        for _source, path in configured_env_files():
            file_val = dotenv_values_from_file(path).get(pool_env, "")
            if file_val.strip():
                raw = file_val
    if not raw.strip():
        return False
    try:
        return bool(parse_api_key_pool(raw))
    except ValueError:
        return False


def _value_for_settings_attr(
    state: Mapping[str, Mapping[str, Any]], settings_attr: str
) -> str:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            return str(state.get(field.key, {}).get("value", field.default))
    return ""


def _field_key_for_settings_attr(settings_attr: str) -> str:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            return field.key
    raise AssertionError(f"No admin field owns settings attribute {settings_attr!r}")

"""Provider configuration construction from neutral catalog metadata."""

from math import ceil

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.api_key_pool import parse_api_key_pool
from free_claude_code.config.provider_catalog import ProviderDescriptor
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig

_POOL_SETTINGS_BY_PROVIDER = {
    "nvidia_nim": (
        "nvidia_nim_api_keys",
        "nvidia_nim_key_rate_limit",
        40,
    ),
    "open_router": (
        "open_router_api_keys",
        "open_router_key_rate_limit",
        20,
    ),
    "custom": (
        "custom_api_keys",
        "custom_key_rate_limit",
        40,
    ),
}


def string_setting(settings: Settings, attr_name: str | None, default: str = "") -> str:
    """Return a string-valued settings attribute, ignoring non-string mocks."""
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def numeric_setting(settings: Settings, attr_name: str, default: float) -> float:
    """Return a numeric settings attribute, ignoring non-numeric mocks."""
    value = getattr(settings, attr_name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def operator_configured(settings: Settings, attr_name: str) -> bool:
    """Return whether the operator set this field rather than inheriting a default.

    Two signals are needed. Pydantic records explicitly supplied fields in
    ``model_fields_set``, which is exact but absent on the plain objects that
    stand in for settings elsewhere; comparing against the declared default
    works for any object, and a value equal to the default is indistinguishable
    from an unset one anyway.
    """
    fields_set = getattr(settings, "model_fields_set", None)
    if isinstance(fields_set, set | frozenset) and attr_name in fields_set:
        return True
    field = Settings.model_fields.get(attr_name)
    if field is None:
        return False
    value = getattr(settings, attr_name, field.default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return value != field.default


def rate_with_margin(limit: int, margin: float) -> int:
    """Hold back a fraction of a quota, never less than one whole request.

    The cushion covers the gap between counting a request when we send it and
    the provider counting it on arrival. One request is the smallest cushion
    that means anything, so a small quota still gets a real one.
    """
    if margin <= 0.0 or limit <= 1:
        return max(1, limit)
    return max(1, limit - max(1, ceil(limit * margin)))


def resolve_rate_policy(
    descriptor: ProviderDescriptor, settings: Settings
) -> tuple[int, float]:
    """Return the request quota and window to pace this provider at.

    Precedence is operator, then provider, then global default: an explicitly
    configured setting always wins, otherwise the provider's own published quota
    beats a shared default that cannot be right for every provider at once.

    The margin is applied once at this provider-wide gate so the local limit
    leaves room for transport latency and clock skew.
    """
    limit = int(numeric_setting(settings, "provider_rate_limit", 40))
    window = numeric_setting(settings, "provider_rate_window", 60.0)
    if descriptor.rate_limit is not None and not operator_configured(
        settings, "provider_rate_limit"
    ):
        limit = descriptor.rate_limit
    if descriptor.rate_window is not None and not operator_configured(
        settings, "provider_rate_window"
    ):
        window = descriptor.rate_window
    margin = numeric_setting(settings, "provider_rate_margin", 0.05)
    return rate_with_margin(limit, margin), window


def provider_credential(descriptor: ProviderDescriptor, settings: Settings) -> str:
    """Return the configured credential for a provider descriptor."""
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    credentials = provider_credentials(descriptor, settings)
    if credentials:
        return credentials[0]
    if descriptor.credential_attr:
        credential = string_setting(settings, descriptor.credential_attr)
        if credential.strip():
            return credential
    return ""


def provider_credentials(
    descriptor: ProviderDescriptor, settings: Settings
) -> tuple[str, ...]:
    """Return configured NIM/OpenRouter credentials in rotation order."""
    pool_settings = _POOL_SETTINGS_BY_PROVIDER.get(descriptor.provider_id)
    if pool_settings is None:
        return ()
    pool_value = string_setting(settings, pool_settings[0])
    if pool_value.strip():
        return parse_api_key_pool(pool_value)
    if descriptor.credential_attr:
        credential = string_setting(settings, descriptor.credential_attr)
        if credential.strip():
            return (credential,)
    return ()


def has_provider_configuration(
    descriptor: ProviderDescriptor, settings: Settings
) -> bool:
    """Return whether all provider-defining settings are present."""
    attrs = descriptor.configuration_attrs()
    if attrs:
        return all(_attr_configured(descriptor, settings, attr) for attr in attrs)
    return descriptor.static_credential is not None


def _attr_configured(
    descriptor: ProviderDescriptor, settings: Settings, attr: str
) -> bool:
    if (
        attr == descriptor.credential_attr
        and descriptor.provider_id in _POOL_SETTINGS_BY_PROVIDER
    ):
        return bool(provider_credentials(descriptor, settings))
    return bool(string_setting(settings, attr).strip())


def require_provider_credential(
    descriptor: ProviderDescriptor, credential: str
) -> None:
    """Raise a user-facing configuration error when a required key is missing."""
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    env_names = [descriptor.credential_env]
    if pool_settings := _POOL_SETTINGS_BY_PROVIDER.get(descriptor.provider_id):
        pool_env = Settings.model_fields[pool_settings[0]].validation_alias
        if isinstance(pool_env, str):
            env_names.insert(0, pool_env)
    message = f"{' or '.join(env_names)} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise ApplicationUnavailableError(message)


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings
) -> ProviderConfig:
    """Build shared provider configuration for one provider descriptor."""
    api_keys = provider_credentials(descriptor, settings)
    credential = api_keys[0] if api_keys else provider_credential(descriptor, settings)
    require_provider_credential(descriptor, credential)
    base_url = string_setting(
        settings, descriptor.base_url_attr, descriptor.default_base_url or ""
    )
    resolved_base_url = base_url or descriptor.default_base_url
    if not resolved_base_url:
        if descriptor.base_url_attr is None:
            raise AssertionError(
                f"Provider {descriptor.provider_id!r} has no base URL owner."
            )
        field = Settings.model_fields[descriptor.base_url_attr]
        env_name = field.validation_alias or descriptor.base_url_attr
        raise ApplicationUnavailableError(
            f"{env_name} is not set. Add it to your .env file."
        )
    proxy = string_setting(settings, descriptor.proxy_attr)
    rate_limit, rate_window = resolve_rate_policy(descriptor, settings)
    http_read_timeout = settings.http_read_timeout
    key_rate_limit: int | None = None
    if pool_settings := _POOL_SETTINGS_BY_PROVIDER.get(descriptor.provider_id):
        key_rate_limit = int(
            numeric_setting(settings, pool_settings[1], pool_settings[2])
        )
        rate_limit = key_rate_limit * len(api_keys)
        rate_window = 60.0
    return ProviderConfig(
        api_key=credential,
        base_url=resolved_base_url,
        api_keys=api_keys,
        key_rate_limit=key_rate_limit,
        key_rate_window=60.0,
        rate_limit=rate_limit,
        rate_window=rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
        key_hedge_delay_seconds=float(settings.key_hedge_delay_seconds),
    )

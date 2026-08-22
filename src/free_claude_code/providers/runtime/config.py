"""Provider configuration construction from neutral catalog metadata."""

from math import ceil

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.api_keys import parse_api_key_list
from free_claude_code.config.provider_catalog import ProviderDescriptor
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig


def string_setting(settings: Settings, attr_name: str | None, default: str = "") -> str:
    """Return a string-valued settings attribute, ignoring non-string mocks."""
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def provider_credential_pool(
    descriptor: ProviderDescriptor, settings: Settings
) -> tuple[str, ...]:
    """Return the configured pool of interchangeable keys, if this provider has one."""
    if descriptor.credential_pool_attr is None:
        return ()
    raw = string_setting(settings, descriptor.credential_pool_attr)
    env_name = descriptor.credential_pool_attr.upper()
    return parse_api_key_list(raw, env_name=env_name)


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
    """Return the per-key request quota and window to pace this provider at.

    Precedence is operator, then provider, then global default: an explicitly
    configured setting always wins, otherwise the provider's own published quota
    beats a shared default that cannot be right for every provider at once.

    The margin is applied here, to the per-key figure, because the provider-wide
    gate is built as ``per_key * pool_size``. Shaving once at this level keeps
    both tiers agreeing on capacity instead of cushioning the total twice.
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


def resolve_key_usage_policy(
    descriptor: ProviderDescriptor, settings: Settings
) -> tuple[int, float | None]:
    """Return the usage budget per pooled key and its rolling window.

    Providers meter keys differently, so the budget follows the provider: a
    provider whose free tier caps calls per key on a schedule (OpenRouter's
    daily quota) counts uses against a window; one rate-limited per minute
    instead (NVIDIA NIM) needs no local budget at all because the reactive
    ``429`` cooldowns already model it. Zero meters nothing.
    """
    limits: dict[str, tuple[str, float | None]] = {
        # OpenRouter: 1000 calls per key per day on the free tier.
        "open_router": ("open_router_key_usage_limit", 86400.0),
        # NIM's free tier is rate-limited per minute, not by a consumable
        # budget, so the reactive cooldowns carry it alone.
        "nvidia_nim": ("nvidia_nim_key_usage_limit", None),
    }
    entry = limits.get(descriptor.provider_id)
    if entry is None:
        return 0, None
    attr, window = entry
    limit = numeric_setting(settings, attr, 0.0)
    if limit <= 0:
        return 0, None
    return int(limit), window


def provider_credential(descriptor: ProviderDescriptor, settings: Settings) -> str:
    """Return the configured credential for a provider descriptor.

    A pool stands in for the single credential when only the pool is configured,
    so the shared client and preflight paths always hold a usable key.
    """
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    if descriptor.credential_attr:
        credential = string_setting(settings, descriptor.credential_attr)
        if credential.strip():
            return credential
    pool = provider_credential_pool(descriptor, settings)
    return pool[0] if pool else ""


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
    if string_setting(settings, attr).strip():
        return True
    # A configured key pool makes its singular credential attribute redundant.
    return attr == descriptor.credential_attr and bool(
        provider_credential_pool(descriptor, settings)
    )


def require_provider_credential(
    descriptor: ProviderDescriptor, credential: str
) -> None:
    """Raise a user-facing configuration error when a required key is missing."""
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    message = f"{descriptor.credential_env} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise ApplicationUnavailableError(message)


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings
) -> ProviderConfig:
    """Build shared provider configuration for one provider descriptor."""
    credential = provider_credential(descriptor, settings)
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
    # A single configured key is not a pool: leaving it empty keeps single-key
    # setups on the existing provider-wide admission path unchanged.
    pool = provider_credential_pool(descriptor, settings)
    rate_limit, rate_window = resolve_rate_policy(descriptor, settings)
    key_usage_limit, key_usage_window = resolve_key_usage_policy(descriptor, settings)
    return ProviderConfig(
        api_key=credential,
        base_url=resolved_base_url,
        api_keys=pool if len(pool) > 1 else (),
        key_usage_limit=key_usage_limit,
        key_usage_window_seconds=key_usage_window,
        rate_limit=rate_limit,
        rate_window=rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )

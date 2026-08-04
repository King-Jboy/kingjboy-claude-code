"""Provider configuration construction from neutral catalog metadata."""

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
    return ProviderConfig(
        api_key=credential,
        base_url=resolved_base_url,
        api_keys=pool if len(pool) > 1 else (),
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )

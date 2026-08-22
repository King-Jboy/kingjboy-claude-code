from dataclasses import FrozenInstanceError

import pytest

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
    ProviderDescriptor,
)


def test_provider_descriptors_are_immutable_values() -> None:
    descriptor = ProviderDescriptor(
        provider_id="local",
        display_name="Local",
        local=True,
    )

    assert descriptor.local is True
    assert not hasattr(descriptor, "__dict__")
    with pytest.raises(FrozenInstanceError):
        descriptor.__setattr__("local", False)


def test_catalog_has_no_transport_metadata() -> None:
    assert "transport_type" not in ProviderDescriptor.__slots__
    assert "capabilities" not in ProviderDescriptor.__slots__


def test_catalog_local_assignments_are_exact() -> None:
    assert {
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if descriptor.local
    } == {"lmstudio", "ollama"}


def test_every_provider_declares_its_configuration_boundary() -> None:
    assert all(
        descriptor.configuration_attrs()
        or descriptor.static_credential is not None
        or descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT
        for descriptor in PROVIDER_CATALOG.values()
    )


def test_openai_is_a_connected_account_without_api_key_configuration() -> None:
    descriptor = PROVIDER_CATALOG["openai"]

    assert descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT
    assert descriptor.credential_env is None
    assert descriptor.configuration_attrs() == ()
    assert descriptor.default_base_url == "https://chatgpt.com/backend-api/codex"

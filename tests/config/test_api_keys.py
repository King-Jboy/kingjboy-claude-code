"""Credential-pool parsing and the settings/catalog wiring that consumes it."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import (
    commit_prepared_admin_update,
    prepare_admin_update,
)
from free_claude_code.config.admin.sources import dotenv_values_from_file
from free_claude_code.config.admin.values import MASKED_SECRET
from free_claude_code.config.api_keys import parse_api_key_list
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.runtime.config import (
    build_provider_config,
    has_provider_configuration,
    provider_credential_pool,
    rate_with_margin,
    resolve_rate_policy,
)
from free_claude_code.providers.runtime.factory import (
    MAX_POOLED_CONCURRENCY,
    create_provider,
)

POOL_KEYS = '["pool-a", "pool-b", "pool-c"]'

NIM = PROVIDER_CATALOG["nvidia_nim"]
OPEN_ROUTER = PROVIDER_CATALOG["open_router"]


def _settings(**overrides: str) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def test_parses_a_json_list_into_an_ordered_tuple() -> None:
    assert parse_api_key_list('["a", "b", "c"]', env_name="KEYS") == ("a", "b", "c")


def test_blank_input_is_an_empty_pool() -> None:
    assert parse_api_key_list("", env_name="KEYS") == ()
    assert parse_api_key_list("   ", env_name="KEYS") == ()


def test_entries_are_stripped_deduplicated_and_order_preserving() -> None:
    raw = '[" a ", "b", "a", "", "   ", "c", "b"]'

    assert parse_api_key_list(raw, env_name="KEYS") == ("a", "b", "c")


def test_pool_size_is_not_capped() -> None:
    keys = [f"key-{index}" for index in range(50)]
    raw = "[" + ", ".join(f'"{key}"' for key in keys) + "]"

    assert parse_api_key_list(raw, env_name="KEYS") == tuple(keys)


def test_malformed_json_names_the_variable_and_shows_the_shape() -> None:
    with pytest.raises(ValueError) as error:
        parse_api_key_list("key-one, key-two", env_name="NVIDIA_NIM_API_KEYS")

    message = str(error.value)
    assert "NVIDIA_NIM_API_KEYS" in message
    assert '["key-one", "key-two"]' in message


def test_a_json_scalar_is_rejected_rather_than_treated_as_one_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_NIM_API_KEYS"):
        parse_api_key_list('"key-one"', env_name="NVIDIA_NIM_API_KEYS")


def test_non_string_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEYS"):
        parse_api_key_list('["key-one", 7]', env_name="OPENROUTER_API_KEYS")


def test_malformed_pool_fails_settings_construction() -> None:
    with pytest.raises(ValueError, match="NVIDIA_NIM_API_KEYS"):
        _settings(NVIDIA_NIM_API_KEYS="not json")


def test_valid_pool_survives_settings_construction_verbatim() -> None:
    settings = _settings(NVIDIA_NIM_API_KEYS='["a", "b"]')

    assert settings.nvidia_nim_api_keys == '["a", "b"]'
    assert provider_credential_pool(NIM, settings) == ("a", "b")


def test_a_pool_alone_counts_as_a_configured_provider() -> None:
    settings = _settings(nvidia_nim_api_key="", NVIDIA_NIM_API_KEYS='["a", "b"]')

    assert has_provider_configuration(NIM, settings)


def test_an_unconfigured_provider_stays_unconfigured() -> None:
    settings = _settings(OPENROUTER_API_KEY="", OPENROUTER_API_KEYS="")

    assert not has_provider_configuration(OPEN_ROUTER, settings)


def test_pool_supplies_the_single_credential_when_only_the_pool_is_set() -> None:
    settings = _settings(nvidia_nim_api_key="", NVIDIA_NIM_API_KEYS='["a", "b"]')

    config = build_provider_config(NIM, settings)

    assert config.api_key == "a"
    assert config.api_keys == ("a", "b")


def test_a_single_pooled_key_leaves_the_provider_unpooled() -> None:
    settings = _settings(nvidia_nim_api_key="", NVIDIA_NIM_API_KEYS='["only"]')

    config = build_provider_config(NIM, settings)

    assert config.api_key == "only"
    assert config.api_keys == ()


def test_the_singular_key_still_wins_for_the_shared_client() -> None:
    settings = _settings(
        nvidia_nim_api_key="singular", NVIDIA_NIM_API_KEYS='["a", "b"]'
    )

    config = build_provider_config(NIM, settings)

    assert config.api_key == "singular"
    assert config.api_keys == ("a", "b")


def _managed_env_with_pools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the managed env at a temp home that already holds pooled keys."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    managed = managed_env_path()
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_text(
        'NVIDIA_NIM_API_KEY="single"\n'
        f"NVIDIA_NIM_API_KEYS='{POOL_KEYS}'\n"
        'OPENROUTER_API_KEYS=\'["or-a", "or-b"]\'\n',
        encoding="utf-8",
    )
    return managed


def test_pool_variables_are_registered_admin_fields() -> None:
    # Unregistered variables are dropped by the save rewrite, so registration
    # is what actually keeps a pool alive across an Admin save.
    for key in ("NVIDIA_NIM_API_KEYS", "OPENROUTER_API_KEYS"):
        field = FIELD_BY_KEY[key]
        assert field.field_type == "textarea"
        assert field.secret is True


def test_saving_an_unrelated_field_preserves_the_key_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = _managed_env_with_pools(tmp_path, monkeypatch)

    # Editing any unrelated field rewrites the whole managed file.
    prepared = prepare_admin_update({"GROQ_API_KEY": "groq-key"})
    assert prepared.valid, prepared.errors
    commit_prepared_admin_update(prepared)

    saved = dotenv_values_from_file(managed)
    assert saved["NVIDIA_NIM_API_KEYS"] == POOL_KEYS
    assert saved["OPENROUTER_API_KEYS"] == '["or-a", "or-b"]'
    assert saved["NVIDIA_NIM_API_KEY"] == "single"
    assert saved["GROQ_API_KEY"] == "groq-key"


def test_resubmitting_the_masked_pool_value_keeps_the_stored_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = _managed_env_with_pools(tmp_path, monkeypatch)

    # The Admin UI renders a configured secret as the mask, so an untouched
    # field posts the mask back and must not overwrite the real value.
    prepared = prepare_admin_update({"NVIDIA_NIM_API_KEYS": MASKED_SECRET})
    assert prepared.valid, prepared.errors
    commit_prepared_admin_update(prepared)

    assert dotenv_values_from_file(managed)["NVIDIA_NIM_API_KEYS"] == POOL_KEYS


def test_editing_the_pool_replaces_the_stored_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = _managed_env_with_pools(tmp_path, monkeypatch)
    replacement = '["fresh-a", "fresh-b"]'

    prepared = prepare_admin_update({"NVIDIA_NIM_API_KEYS": replacement})
    assert prepared.valid, prepared.errors
    commit_prepared_admin_update(prepared)

    assert dotenv_values_from_file(managed)["NVIDIA_NIM_API_KEYS"] == replacement


def test_saving_a_malformed_pool_is_rejected_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = _managed_env_with_pools(tmp_path, monkeypatch)

    prepared = prepare_admin_update({"NVIDIA_NIM_API_KEYS": "key-one, key-two"})

    assert not prepared.valid
    assert any("NVIDIA_NIM_API_KEYS" in error for error in prepared.errors)
    assert dotenv_values_from_file(managed)["NVIDIA_NIM_API_KEYS"] == POOL_KEYS


def _admission_for_pool(size: int, **overrides: str) -> Any:
    """Return the admission controller built for a pool of the given size."""
    keys = "[" + ", ".join(f'"key-{index}"' for index in range(size)) + "]"
    settings = _settings(nvidia_nim_api_key="", NVIDIA_NIM_API_KEYS=keys, **overrides)
    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch(
            "free_claude_code.providers.runtime.factory.ProviderAdmissionController"
        ) as controller,
    ):
        create_provider("nvidia_nim", settings)
    return controller.call_args.kwargs


def test_each_provider_is_paced_at_its_own_published_quota() -> None:
    # One shared rate limit cannot serve two providers with different ceilings.
    # Pacing every provider at the highest one guarantees refusals from the
    # lowest, which is what a single global setting used to do.
    settings = _settings(NVIDIA_NIM_API_KEY="nim-key", OPENROUTER_API_KEY="or-key")

    nim, _ = resolve_rate_policy(PROVIDER_CATALOG["nvidia_nim"], settings)
    openrouter, _ = resolve_rate_policy(PROVIDER_CATALOG["open_router"], settings)

    assert (nim, openrouter) == (38, 19)


def test_an_explicitly_configured_quota_overrides_the_published_one() -> None:
    # The provider's own figure is a better default than a shared one, but it
    # must not overrule an operator who knows their account's real limit.
    settings = _settings(NVIDIA_NIM_API_KEY="nim-key", PROVIDER_RATE_LIMIT="100")

    limit, _ = resolve_rate_policy(PROVIDER_CATALOG["nvidia_nim"], settings)

    assert limit == 95


def test_the_safety_margin_can_be_turned_off() -> None:
    settings = _settings(NVIDIA_NIM_API_KEY="nim-key", PROVIDER_RATE_MARGIN="0")

    limit, _ = resolve_rate_policy(PROVIDER_CATALOG["nvidia_nim"], settings)

    assert limit == 40


def test_the_margin_holds_back_at_least_one_whole_request() -> None:
    # A percentage of a small quota rounds to nothing, and a cushion of zero
    # requests is not a cushion.
    assert rate_with_margin(4, 0.05) == 3
    assert rate_with_margin(1, 0.05) == 1


def test_pooled_quota_scales_with_every_key() -> None:
    # Each key carries its own upstream quota, so the provider-wide window has
    # to admit the pooled total or the gate would cap the pool at one key.
    single = _admission_for_pool(2)["rate_limit"]

    assert _admission_for_pool(8)["rate_limit"] == single * 4


def test_pooled_concurrency_stops_scaling_at_the_ceiling() -> None:
    # Concurrency is bounded by local sockets rather than by quota, so it scales
    # with the pool only up to a ceiling. The ceiling has to sit well above one
    # client's handful of parallel requests: a streaming response holds its slot
    # for tens of seconds, so too low a ceiling - not the rate limit - becomes
    # the real throughput bound and strands most of a large pool's quota.
    assert _admission_for_pool(2)["max_concurrency"] == 10
    assert _admission_for_pool(4)["max_concurrency"] == 20
    assert _admission_for_pool(40)["max_concurrency"] == MAX_POOLED_CONCURRENCY


def test_a_pooled_ceiling_below_one_keys_concurrency_still_binds() -> None:
    # The ceiling is a cap, not a floor: an operator who lowers it below one
    # key's concurrency means it, and the setting must not be silently ignored.
    admission = _admission_for_pool(
        2, PROVIDER_MAX_CONCURRENCY="40", PROVIDER_MAX_POOLED_CONCURRENCY="32"
    )

    assert admission["max_concurrency"] == 32


def test_the_pooled_ceiling_never_touches_a_single_key_provider() -> None:
    # The setting exists to bound pool scaling; a single-credential provider
    # keeps its own concurrency whatever the pooled ceiling says.
    admission = _admission_for_pool(
        1, PROVIDER_MAX_CONCURRENCY="40", PROVIDER_MAX_POOLED_CONCURRENCY="8"
    )

    assert admission["max_concurrency"] == 40


def test_providers_without_a_pool_attribute_never_report_one() -> None:
    settings = _settings(GROQ_API_KEY="groq-key")

    assert provider_credential_pool(PROVIDER_CATALOG["groq"], settings) == ()
    assert build_provider_config(PROVIDER_CATALOG["groq"], settings).api_keys == ()

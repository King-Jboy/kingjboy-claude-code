"""PINNED_MODELS parsing and the settings/admin wiring that consumes it."""

from pathlib import Path
from typing import Any

import pytest

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import (
    commit_prepared_admin_update,
    prepare_admin_update,
)
from free_claude_code.config.admin.sources import dotenv_values_from_file
from free_claude_code.config.model_refs import (
    parse_model_ref_list,
    pinned_model_refs,
)
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.settings import Settings

PINNED = '["nvidia_nim/deepseek-ai/deepseek-v4-flash", "open_router/z-ai/glm-5.2:free"]'


def _settings(**overrides: str) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def test_parses_a_json_list_into_an_ordered_tuple() -> None:
    assert parse_model_ref_list(PINNED, env_name="PINNED_MODELS") == (
        "nvidia_nim/deepseek-ai/deepseek-v4-flash",
        "open_router/z-ai/glm-5.2:free",
    )


def test_blank_input_is_an_empty_list() -> None:
    assert parse_model_ref_list("", env_name="PINNED_MODELS") == ()
    assert parse_model_ref_list("   ", env_name="PINNED_MODELS") == ()


def test_entries_are_stripped_deduplicated_and_order_preserving() -> None:
    raw = '[" a/one ", "b/two", "a/one", "", "   ", "c/three"]'

    assert parse_model_ref_list(raw, env_name="PINNED_MODELS") == (
        "a/one",
        "b/two",
        "c/three",
    )


def test_malformed_json_names_the_variable_and_shows_the_shape() -> None:
    with pytest.raises(ValueError) as error:
        parse_model_ref_list("nvidia_nim/a, open_router/b", env_name="PINNED_MODELS")

    message = str(error.value)
    assert "PINNED_MODELS" in message
    assert "nvidia_nim/deepseek-ai/deepseek-v4-flash" in message


def test_a_json_scalar_is_rejected_rather_than_treated_as_one_ref() -> None:
    with pytest.raises(ValueError, match="PINNED_MODELS"):
        parse_model_ref_list('"nvidia_nim/model"', env_name="PINNED_MODELS")


def test_non_string_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="PINNED_MODELS"):
        parse_model_ref_list('["nvidia_nim/model", 7]', env_name="PINNED_MODELS")


def test_an_unprefixed_model_is_rejected() -> None:
    # A bare model id silently routes nowhere, so it must not reach a picker.
    with pytest.raises(ValueError, match="provider/model"):
        parse_model_ref_list('["deepseek-v4-flash"]', env_name="PINNED_MODELS")


def test_an_empty_ref_segment_is_rejected() -> None:
    with pytest.raises(ValueError, match="provider/model"):
        parse_model_ref_list('["nvidia_nim/"]', env_name="PINNED_MODELS")


def test_malformed_pinned_list_fails_settings_construction() -> None:
    with pytest.raises(ValueError, match="PINNED_MODELS"):
        _settings(PINNED_MODELS="not json")


def test_valid_pinned_list_survives_settings_construction_verbatim() -> None:
    settings = _settings(PINNED_MODELS=PINNED)

    assert settings.pinned_models == PINNED
    assert pinned_model_refs(settings) == (
        "nvidia_nim/deepseek-ai/deepseek-v4-flash",
        "open_router/z-ai/glm-5.2:free",
    )


def test_pinned_models_is_a_registered_admin_textarea() -> None:
    # Unregistered variables are dropped by the save rewrite, so registration
    # is what keeps the list alive across an Admin save.
    field = FIELD_BY_KEY["PINNED_MODELS"]

    assert field.field_type == "textarea"
    assert field.section_id == "models"
    assert field.secret is False


def test_saving_the_pinned_list_round_trips_through_the_managed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    managed = managed_env_path()
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_text("MODEL=nvidia_nim/a/b\n", encoding="utf-8")

    prepared = prepare_admin_update({"PINNED_MODELS": PINNED})
    assert prepared.valid, prepared.errors
    commit_prepared_admin_update(prepared)

    assert dotenv_values_from_file(managed)["PINNED_MODELS"] == PINNED


def test_saving_a_malformed_pinned_list_is_rejected_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    managed = managed_env_path()
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_text(f"PINNED_MODELS='{PINNED}'\n", encoding="utf-8")

    prepared = prepare_admin_update({"PINNED_MODELS": "nvidia_nim/a, nvidia_nim/b"})

    assert not prepared.valid
    assert any("PINNED_MODELS" in error for error in prepared.errors)
    assert dotenv_values_from_file(managed)["PINNED_MODELS"] == PINNED

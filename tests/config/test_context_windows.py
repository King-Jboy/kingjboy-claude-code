"""Reading context.md back, and resolving the window to advertise from it."""

from pathlib import Path
from typing import Any

import pytest

from free_claude_code.config.constants import DEFAULT_CLIENT_CONTEXT_WINDOW
from free_claude_code.config.context_windows import (
    context_windows_path,
    load_context_windows,
    resolve_client_context_window,
)
from free_claude_code.config.settings import Settings

TABLE = """# Model context windows

## nvidia_nim (2 of 3 known)

| Model | Context | Source |
| --- | ---: | --- |
| `deepseek-ai/deepseek-v4-flash` | 1,048,576 | measured |
| `deepseek-ai/deepseek-v4-pro` | 262,144 | measured |
| `thinkingmachines/inkling` | unknown | Internal server error |

## open_router (1 of 1 known)

| Model | Context | Source |
| --- | ---: | --- |
| `z-ai/glm-5.2:free` | 131,072 | published |
"""


def _settings(**overrides: Any) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _write_table(tmp_path: Path, body: str = TABLE) -> Path:
    table = tmp_path / "context.md"
    table.write_text(body, encoding="utf-8")
    return table


def test_rows_are_keyed_by_provider_prefixed_ref(tmp_path: Path) -> None:
    # The heading supplies the provider, so a bare model id becomes a routable
    # ref that matches what MODEL is actually set to.
    windows = load_context_windows(_write_table(tmp_path))

    assert windows["nvidia_nim/deepseek-ai/deepseek-v4-flash"] == 1_048_576
    assert windows["nvidia_nim/deepseek-ai/deepseek-v4-pro"] == 262_144
    assert windows["open_router/z-ai/glm-5.2:free"] == 131_072


def test_unresolved_rows_are_omitted_rather_than_recorded_as_zero(
    tmp_path: Path,
) -> None:
    windows = load_context_windows(_write_table(tmp_path))

    assert "nvidia_nim/thinkingmachines/inkling" not in windows


def test_a_missing_table_is_an_empty_mapping_not_an_error(tmp_path: Path) -> None:
    # Losing the table must degrade to the default, never stop a CLI launching.
    assert load_context_windows(tmp_path / "absent.md") == {}


def test_rows_before_any_provider_heading_are_ignored(tmp_path: Path) -> None:
    body = "| `orphan/model` | 999 | measured |\n"

    assert load_context_windows(_write_table(tmp_path, body)) == {}


def test_the_default_path_is_the_managed_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert context_windows_path() == tmp_path / ".fcc" / "context.md"


def test_an_explicit_setting_wins_over_the_table() -> None:
    resolved = resolve_client_context_window(
        _settings(model="nvidia_nim/deepseek-ai/deepseek-v4-pro"),
        configured=300_000,
        windows={"nvidia_nim/deepseek-ai/deepseek-v4-pro": 262_144},
    )

    assert resolved.value == 300_000
    assert resolved.source == "CLIENT_CONTEXT_WINDOW"


def test_a_blank_setting_takes_the_routed_model_window() -> None:
    resolved = resolve_client_context_window(
        _settings(model="nvidia_nim/deepseek-ai/deepseek-v4-flash"),
        configured=None,
        windows={"nvidia_nim/deepseek-ai/deepseek-v4-flash": 1_048_576},
    )

    assert resolved.value == 1_048_576
    assert resolved.source == "context.md"
    assert resolved.model_ref == "nvidia_nim/deepseek-ai/deepseek-v4-flash"


def test_the_smallest_configured_route_bounds_the_window() -> None:
    # Claude Code compacts against one number for the whole session but can be
    # pointed at any routed tier, so a larger value would overflow the smallest.
    resolved = resolve_client_context_window(
        _settings(
            model="nvidia_nim/big/model",
            MODEL_HAIKU="nvidia_nim/small/model",
        ),
        configured=None,
        windows={
            "nvidia_nim/big/model": 1_000_000,
            "nvidia_nim/small/model": 128_000,
        },
    )

    assert resolved.value == 128_000
    assert resolved.model_ref == "nvidia_nim/small/model"


def test_an_unmeasured_route_does_not_drag_the_window_down() -> None:
    # Absence of a row is not evidence of a small window; only recorded
    # ceilings constrain the result.
    resolved = resolve_client_context_window(
        _settings(
            model="nvidia_nim/measured/model",
            MODEL_HAIKU="nvidia_nim/never/measured",
        ),
        configured=None,
        windows={"nvidia_nim/measured/model": 262_144},
    )

    assert resolved.value == 262_144


def test_nothing_recorded_falls_back_to_the_conservative_default() -> None:
    resolved = resolve_client_context_window(
        _settings(model="nvidia_nim/never/measured"),
        configured=None,
        windows={},
    )

    assert resolved.value == DEFAULT_CLIENT_CONTEXT_WINDOW
    assert resolved.source == "default"
    assert resolved.model_ref is None


def test_a_blank_env_value_means_resolve_rather_than_reject() -> None:
    # The Admin number field posts "" when cleared, which must not fail
    # validation the way a blank int normally would.
    assert _settings(CLIENT_CONTEXT_WINDOW="").client_context_window is None

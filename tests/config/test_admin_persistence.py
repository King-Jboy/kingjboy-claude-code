"""Managed env rendering, and survival of variables the manifest does not own."""

from pathlib import Path

import pytest

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import (
    commit_prepared_admin_update,
    prepare_admin_update,
    render_env_file,
    unmanaged_values_from_managed_file,
)
from free_claude_code.config.admin.sources import dotenv_values_from_file
from free_claude_code.config.admin.values import MASKED_SECRET
from free_claude_code.config.env_migrations import RETIRED_ENV_KEYS
from free_claude_code.config.paths import managed_env_path


def _managed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> Path:
    """Point the managed env at a temp home holding the given contents."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    managed = managed_env_path()
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_text(contents, encoding="utf-8")
    return managed


def test_unmanaged_variables_survive_an_admin_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rendered file is built from FIELDS alone, so anything hand-added had
    # been dropped by the full rewrite that every save performs.
    managed = _managed_env(
        tmp_path,
        monkeypatch,
        'HTTPS_PROXY="http://127.0.0.1:8080"\nNO_PROXY=localhost\n',
    )

    prepared = prepare_admin_update({"GROQ_API_KEY": "groq-key"})
    assert prepared.valid, prepared.errors
    commit_prepared_admin_update(prepared)

    saved = dotenv_values_from_file(managed)
    assert saved["HTTPS_PROXY"] == "http://127.0.0.1:8080"
    assert saved["NO_PROXY"] == "localhost"
    assert saved["GROQ_API_KEY"] == "groq-key"


def test_unmanaged_variables_survive_repeated_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The second save must read the preserved block back out of the file the
    # first save wrote, so round-tripping is what actually keeps them alive.
    managed = _managed_env(tmp_path, monkeypatch, "CUSTOM_TOKEN=keep-me\n")

    for value in ("first", "second", "third"):
        prepared = prepare_admin_update({"GROQ_API_KEY": value})
        assert prepared.valid, prepared.errors
        commit_prepared_admin_update(prepared)

    saved = dotenv_values_from_file(managed)
    assert saved["CUSTOM_TOKEN"] == "keep-me"
    assert saved["GROQ_API_KEY"] == "third"


def test_values_needing_quotes_round_trip_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = _managed_env(tmp_path, monkeypatch, 'CUSTOM_NOTE="a b # c"\n')

    prepared = prepare_admin_update({"GROQ_API_KEY": "groq-key"})
    commit_prepared_admin_update(prepared)

    assert dotenv_values_from_file(managed)["CUSTOM_NOTE"] == "a b # c"


def test_manifest_fields_are_never_treated_as_unmanaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A key registered in the manifest is rendered by its own section. Counting
    # it as unmanaged too would emit it twice and let the stale copy win.
    _managed_env(
        tmp_path,
        monkeypatch,
        "GROQ_API_KEY=managed\nCUSTOM_TOKEN=unmanaged\n",
    )

    unmanaged = unmanaged_values_from_managed_file()

    assert unmanaged == {"CUSTOM_TOKEN": "unmanaged"}
    assert "GROQ_API_KEY" in FIELD_BY_KEY


def test_retired_settings_are_still_cleared_by_a_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rewrite is what finally removes settings FCC no longer reads, so
    # preservation must not resurrect them alongside genuinely foreign keys.
    managed = _managed_env(
        tmp_path,
        monkeypatch,
        "ZAI_BASE_URL=https://custom.zai.invalid/v1\n"
        "CLAUDE_WORKSPACE=C:/custom/workspace\n"
        "CUSTOM_TOKEN=keep-me\n",
    )

    prepared = prepare_admin_update({"GROQ_API_KEY": "groq-key"})
    commit_prepared_admin_update(prepared)

    saved = dotenv_values_from_file(managed)
    assert "ZAI_BASE_URL" not in saved
    assert "CLAUDE_WORKSPACE" not in saved
    assert saved["CUSTOM_TOKEN"] == "keep-me"


def test_every_retired_key_is_a_key_the_manifest_no_longer_owns() -> None:
    # A retired key that got re-registered would be silently dropped from its
    # own section, so the two sets must stay disjoint.
    assert not RETIRED_ENV_KEYS & set(FIELD_BY_KEY)


def test_rendered_file_lists_each_key_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _managed_env(
        tmp_path,
        monkeypatch,
        "GROQ_API_KEY=managed\nCUSTOM_TOKEN=unmanaged\n",
    )

    prepared = prepare_admin_update({"GROQ_API_KEY": "groq-key"})
    rendered = render_env_file(
        prepared.target_values,
        unmanaged=prepared.unmanaged_values,
    )

    assignments = [
        line.split("=", 1)[0]
        for line in rendered.splitlines()
        if line and not line.startswith("#")
    ]
    assert len(assignments) == len(set(assignments))
    assert assignments.count("GROQ_API_KEY") == 1
    assert assignments.count("CUSTOM_TOKEN") == 1


def test_preview_masks_unmanaged_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The manifest carries no secret flag for these, so a preview cannot tell a
    # proxy host from a token; mask them rather than leak one.
    _managed_env(tmp_path, monkeypatch, "CUSTOM_TOKEN=super-secret\n")

    preview = prepare_admin_update({"GROQ_API_KEY": "groq-key"}).validation_response()[
        "env_preview"
    ]

    assert "super-secret" not in preview
    assert f"CUSTOM_TOKEN={MASKED_SECRET}" in preview


def test_render_without_unmanaged_values_adds_no_section() -> None:
    rendered = render_env_file({"GROQ_API_KEY": "groq-key"})

    assert "Not managed by the Admin UI" not in rendered

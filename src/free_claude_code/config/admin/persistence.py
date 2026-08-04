"""Managed env persistence, validation preview, and rendering."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from free_claude_code.config.env_migrations import RETIRED_ENV_KEYS
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.settings import Settings

from .manifest import FIELD_BY_KEY, FIELDS, SECTIONS, ConfigFieldSpec
from .sources import dotenv_values_from_file, is_locked_source, template_values
from .validation import settings_from_values
from .values import MASKED_SECRET, load_value_state, normalize_for_env


@dataclass(frozen=True, slots=True)
class PreparedAdminUpdate:
    """Validated Admin update ready for an atomic managed-file commit."""

    target_values: dict[str, str]
    settings: Settings | None
    errors: tuple[str, ...]
    pending_fields: tuple[str, ...]
    path: Path
    unmanaged_values: dict[str, str] = dataclass_field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.settings is not None

    def validation_response(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "env_preview": self._env_preview(),
        }

    def applied_response(self) -> dict[str, Any]:
        if not self.valid:
            return self.validation_response() | {
                "applied": False,
                "pending_fields": [],
            }
        return {
            "applied": True,
            "valid": True,
            "errors": [],
            "env_preview": self._env_preview(),
            "path": str(self.path),
            "pending_fields": list(self.pending_fields),
        }

    def _env_preview(self) -> str:
        return render_env_file(
            self.target_values,
            mask_secrets=True,
            unmanaged=self.unmanaged_values,
        )


def target_values_with_updates(updates: Mapping[str, Any]) -> dict[str, str]:
    """Return managed env values after applying admin updates."""

    state = load_value_state()
    values = template_values()

    # Preserve existing managed values when present. If no managed config exists,
    # seed the first write from effective repo values to migrate legacy setups.
    managed_values = dotenv_values_from_file(managed_env_path())
    if managed_values:
        values.update(
            {key: val for key, val in managed_values.items() if key in values}
        )
    else:
        for key, entry in state.items():
            if entry["source"] in {"repo_env", "template", "default"}:
                values[key] = str(entry["value"])

    for key, value in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None:
            continue
        if is_locked_source(state[key]["source"]):
            continue
        if field.secret and value == MASKED_SECRET:
            continue
        values[key] = normalize_for_env(value)

    for field in FIELDS:
        values.setdefault(field.key, field.default)
    return values


def unmanaged_values_from_managed_file() -> dict[str, str]:
    """Return managed-file entries the Admin manifest does not own.

    The rendered file is built from ``FIELDS`` alone, so anything hand-added to
    ``~/.fcc/.env`` — a proxy variable, or a key belonging to a newer manifest
    than this build — would be dropped on the next save. Carrying these through
    keeps a save from destroying settings the Admin UI never claimed.

    Retired FCC settings are the deliberate exception: the rewrite is what
    finally clears them, so they stay excluded.
    """

    return {
        key: value
        for key, value in dotenv_values_from_file(managed_env_path()).items()
        if key not in FIELD_BY_KEY and key not in RETIRED_ENV_KEYS
    }


def effective_values_for_validation(
    target_values: Mapping[str, str],
) -> dict[str, str]:
    """Return values validated after preserving locked external sources."""

    values = dict(target_values)
    for key, entry in load_value_state().items():
        if is_locked_source(entry["source"]):
            values[key] = str(entry["value"])
    return values


def validate_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Validate partial admin updates and return a masked generated env preview."""

    return prepare_admin_update(updates).validation_response()


def changed_pending_fields(
    updates: Mapping[str, Any],
    *,
    settings: Settings,
) -> list[str]:
    """Return changed fields that require manual runtime action."""

    state = load_value_state()
    pending: list[str] = []
    for key, value in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or is_locked_source(state[key]["source"]):
            continue
        if field.secret and value == MASKED_SECRET:
            continue
        requires_restart = field.restart_required or field.session_sensitive
        if not requires_restart:
            requires_restart = _active_voice_credential(settings) == key
        if not requires_restart:
            continue
        if normalize_for_env(value) == str(state[key]["value"]):
            continue
        pending.append(key)
    return pending


def _active_voice_credential(settings: Settings) -> str | None:
    if not settings.voice_note_enabled:
        return None
    if settings.whisper_device == "nvidia_nim":
        return "NVIDIA_NIM_API_KEY"
    return "HUGGINGFACE_API_KEY"


def prepare_admin_update(updates: Mapping[str, Any]) -> PreparedAdminUpdate:
    """Validate an update and construct its prospective Settings snapshot."""

    target_values = target_values_with_updates(updates)
    effective_values = effective_values_for_validation(target_values)
    settings, errors = settings_from_values(effective_values)
    pending_fields = (
        tuple(changed_pending_fields(updates, settings=settings))
        if settings is not None
        else ()
    )
    return PreparedAdminUpdate(
        target_values=target_values,
        settings=settings,
        errors=tuple(errors),
        pending_fields=pending_fields,
        path=managed_env_path(),
        unmanaged_values=unmanaged_values_from_managed_file(),
    )


def commit_prepared_admin_update(prepared: PreparedAdminUpdate) -> dict[str, Any]:
    """Atomically persist a previously validated Admin update."""

    if not prepared.valid:
        raise ValueError("Cannot commit an invalid Admin update")

    path = prepared.path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(
            render_env_file(
                prepared.target_values,
                unmanaged=prepared.unmanaged_values,
            ),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return prepared.applied_response()


# Characters that make dotenv read a bare value as syntax rather than data.
# The apostrophe matters as much as the double quote: dotenv opens a
# single-quoted string on it, so a bare "'" was an unparseable line that
# dropped the value, and a bare "'x'" silently came back as "x".
_ENV_CHARS_REQUIRING_QUOTES = ('"', "'", "#", "=", "$")


def quote_env_value(value: str) -> str:
    """Quote a value when dotenv syntax requires it."""

    if value == "":
        return ""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    if any(char.isspace() for char in value) or any(
        char in value for char in _ENV_CHARS_REQUIRING_QUOTES
    ):
        return f'"{escaped}"'
    return value


def render_env_file(
    values: Mapping[str, str],
    *,
    mask_secrets: bool = False,
    unmanaged: Mapping[str, str] | None = None,
) -> str:
    """Render a complete grouped env file.

    ``unmanaged`` entries are appended verbatim so hand-added variables survive
    a save. They are always masked in previews because the manifest carries no
    secret flag for them.
    """

    lines: list[str] = [
        "# Managed by Free Claude Code /admin.",
        "# Edit in the server UI when possible.",
        "",
    ]
    fields_by_section: dict[str, list[ConfigFieldSpec]] = {
        section.section_id: [] for section in SECTIONS
    }
    for field in FIELDS:
        fields_by_section.setdefault(field.section_id, []).append(field)

    for section in SECTIONS:
        lines.append(f"# {section.label}")
        for field in fields_by_section.get(section.section_id, []):
            value = values.get(field.key, field.default)
            if mask_secrets and field.secret and value:
                value = MASKED_SECRET
            lines.append(f"{field.key}={quote_env_value(value)}")
        lines.append("")

    if unmanaged:
        lines.append("# Not managed by the Admin UI; preserved as written.")
        for key in sorted(unmanaged):
            value = unmanaged[key]
            if mask_secrets and value:
                value = MASKED_SECRET
            lines.append(f"{key}={quote_env_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

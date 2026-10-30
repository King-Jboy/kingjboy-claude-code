"""Managed env persistence, validation preview, and rendering."""

import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any
from uuid import uuid4

from free_claude_code.config.context_windows import (
    context_windows_path,
    sync_pinned_context_windows,
)
from free_claude_code.config.env_migrations import RETIRED_ENV_KEYS
from free_claude_code.config.model_refs import pinned_model_refs
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.settings import Settings

from .manifest import FIELD_BY_KEY, FIELDS, SECTIONS, ConfigFieldSpec
from .sources import dotenv_values_from_file, is_locked_source
from .validation import settings_from_values
from .values import MASKED_SECRET, load_value_state, normalize_for_env

MIN_CONTEXT_WINDOW_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class PreparedAdminUpdate:
    """Validated Admin update ready for an atomic managed-file commit."""

    target_values: dict[str, str]
    settings: Settings | None
    errors: tuple[str, ...]
    pending_fields: tuple[str, ...]
    path: Path
    unmanaged_values: dict[str, str] = dataclass_field(default_factory=dict)
    context_window_updates: dict[str, int | None] = dataclass_field(
        default_factory=dict
    )
    sync_context_windows: bool = False

    @property
    def valid(self) -> bool:
        return self.settings is not None and not self.errors

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


def _validate_context_window_updates(
    raw_updates: Mapping[str, Any] | None,
) -> tuple[dict[str, int | None], list[str]]:
    if not raw_updates:
        return {}, []
    cleaned: dict[str, int | None] = {}
    errors: list[str] = []
    for raw_ref, raw_val in raw_updates.items():
        ref = str(raw_ref).strip()
        if not ref or "/" not in ref:
            errors.append(f"Context window entry {raw_ref!r}: invalid provider/model ref")
            continue
        if raw_val is None or (isinstance(raw_val, str) and not raw_val.strip()):
            cleaned[ref] = None
            continue
        if isinstance(raw_val, bool):
            errors.append(
                f"Context window for {ref}: must be an integer >= {MIN_CONTEXT_WINDOW_TOKENS}"
            )
            continue
        try:
            text_val = (
                raw_val.replace(",", "").strip()
                if isinstance(raw_val, str)
                else raw_val
            )
            parsed = int(text_val)
        except (TypeError, ValueError):
            errors.append(
                f"Context window for {ref}: must be an integer >= {MIN_CONTEXT_WINDOW_TOKENS}"
            )
            continue
        if parsed < MIN_CONTEXT_WINDOW_TOKENS:
            errors.append(
                f"Context window for {ref}: must be at least {MIN_CONTEXT_WINDOW_TOKENS} tokens"
            )
            continue
        cleaned[ref] = parsed
    return cleaned, errors


def target_values_with_updates(updates: Mapping[str, Any]) -> dict[str, str]:
    """Return managed env values after applying admin updates."""

    state = load_value_state()
    values = {field.key: field.default for field in FIELDS}

    # Seed from effective repo/template/default values first, then overlay any
    # existing managed values so partial or unmanaged-only ~/.fcc/.env files do
    # not drop repo_env settings or drift from manifest defaults.
    for key, entry in state.items():
        if entry["source"] in {"repo_env", "template", "default"}:
            values[key] = str(entry["value"])

    managed_values = dotenv_values_from_file(managed_env_path())
    if managed_values:
        values.update(
            {key: val for key, val in managed_values.items() if key in values}
        )

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


def validate_updates(
    updates: Mapping[str, Any],
    *,
    context_windows: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate partial admin updates and return a masked generated env preview."""

    return prepare_admin_update(
        updates, context_windows=context_windows
    ).validation_response()


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


def prepare_admin_update(
    updates: Mapping[str, Any],
    *,
    context_windows: Mapping[str, Any] | None = None,
) -> PreparedAdminUpdate:
    """Validate an update and construct its prospective Settings snapshot."""

    target_values = target_values_with_updates(updates)
    effective_values = effective_values_for_validation(target_values)
    unmanaged_values = unmanaged_values_from_managed_file()
    for key, value in unmanaged_values.items():
        effective_values.setdefault(key, value)
    settings, errors = settings_from_values(effective_values)
    cw_updates, cw_errors = _validate_context_window_updates(context_windows)
    all_errors = [*errors, *cw_errors]
    pending_fields = (
        tuple(changed_pending_fields(updates, settings=settings))
        if settings is not None and not all_errors
        else ()
    )
    return PreparedAdminUpdate(
        target_values=target_values,
        settings=settings if not all_errors else None,
        errors=tuple(all_errors),
        pending_fields=pending_fields,
        path=managed_env_path(),
        unmanaged_values=unmanaged_values,
        context_window_updates=cw_updates,
        sync_context_windows=bool(
            context_windows is not None or "PINNED_MODELS" in updates
        ),
    )


def commit_prepared_admin_update(prepared: PreparedAdminUpdate) -> dict[str, Any]:
    """Atomically persist a previously validated Admin update."""

    if not prepared.valid:
        raise ValueError("Cannot commit an invalid Admin update")

    path = prepared.path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp.{uuid4().hex}")
    try:
        # Create the secrets file owner-only; a chmod after writing would leave
        # it briefly readable under the default umask.
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write(
                render_env_file(
                    prepared.target_values,
                    unmanaged=prepared.unmanaged_values,
                )
            )
        if os.name != "nt":
            with contextlib.suppress(OSError):
                path.parent.chmod(0o700)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)

    if prepared.settings is not None and (
        prepared.sync_context_windows or context_windows_path().exists()
    ):
        sync_pinned_context_windows(
            pinned_model_refs(prepared.settings),
            updates=prepared.context_window_updates,
        )

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
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "")
        .replace("\n", "\\n")
    )
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

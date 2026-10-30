"""Admin config value state and API response assembly."""

import os
from typing import Any

from free_claude_code.config.context_windows import (
    context_windows_path,
    pinned_context_windows_state,
)
from free_claude_code.config.env_files import ANTHROPIC_AUTH_TOKEN_ENV
from free_claude_code.config.model_refs import parse_model_ref_list
from free_claude_code.config.paths import managed_env_path

from .manifest import (
    FIELD_BY_KEY,
    FIELDS,
    SECTIONS,
    ConfigFieldSpec,
    ConfigOptionSpec,
)
from .sources import (
    configured_env_files,
    dotenv_values_from_file,
    explicit_env_path,
    is_locked_source,
    repo_env_path,
    template_values,
)
from .status import provider_config_status

MASKED_SECRET = "********"
ValueState = dict[str, dict[str, Any]]


def normalize_for_env(value: Any) -> str:
    """Normalize a submitted admin value for dotenv persistence."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def display_value(field: ConfigFieldSpec, value: str) -> str:
    """Return the Admin UI display value for a raw config value."""

    if field.secret and value:
        return MASKED_SECRET
    return value


def load_value_state() -> ValueState:
    """Load effective admin field values and their sources."""

    values = template_values()
    sources = {key: "template" if key in values else "default" for key in FIELD_BY_KEY}

    for source, path in configured_env_files():
        file_values = dotenv_values_from_file(path)
        for key, value in file_values.items():
            if key in FIELD_BY_KEY:
                values[key] = value
                sources[key] = source

    for key in FIELD_BY_KEY:
        if key in os.environ:
            # Mirror runtime precedence: a dotenv auth token beats the shell's.
            if key == ANTHROPIC_AUTH_TOKEN_ENV and sources[key] not in (
                "template",
                "default",
            ):
                continue
            values[key] = os.environ[key]
            sources[key] = "process"

    return {
        key: {
            "value": values.get(key, ""),
            "source": sources.get(key, "default"),
        }
        for key in FIELD_BY_KEY
    }


def _safe_pinned_refs(raw_pinned: str) -> tuple[str, ...]:
    try:
        return parse_model_ref_list(raw_pinned, env_name="PINNED_MODELS")
    except ValueError:
        return ()


def load_config_response() -> dict[str, Any]:
    """Return manifest and current config values for the admin UI."""

    state = load_value_state()
    fields: list[dict[str, Any]] = []
    for field in FIELDS:
        entry = state[field.key]
        source = entry["source"]
        raw_value = entry["value"]
        fields.append(
            {
                "key": field.key,
                "label": field.label,
                "section": field.section_id,
                "type": field.field_type,
                "value": display_value(field, raw_value),
                "configured": bool(str(raw_value).strip()),
                "source": source,
                "locked": is_locked_source(source),
                "secret": field.secret,
                "advanced": field.advanced,
                "restart_required": field.restart_required,
                "session_sensitive": field.session_sensitive,
                "options": [
                    (
                        {"value": option.value, "label": option.label}
                        if isinstance(option, ConfigOptionSpec)
                        else {"value": option, "label": option}
                    )
                    for option in field.options
                ],
                "description": field.description,
            }
        )

    pinned_raw = str(state.get("PINNED_MODELS", {}).get("value", ""))
    pinned_refs = _safe_pinned_refs(pinned_raw)

    return {
        "sections": [
            {
                "id": section.section_id,
                "label": section.label,
                "description": section.description,
                "advanced": section.advanced,
            }
            for section in SECTIONS
        ],
        "fields": fields,
        "paths": {
            "managed": str(managed_env_path()),
            "repo": str(repo_env_path()),
            "explicit": str(explicit_env_path()) if explicit_env_path() else None,
            "context_windows": str(context_windows_path()),
        },
        "provider_status": provider_config_status(state),
        "context_windows": pinned_context_windows_state(pinned_refs),
    }

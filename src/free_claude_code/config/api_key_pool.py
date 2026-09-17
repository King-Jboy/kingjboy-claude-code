"""Parsing for the two provider credential pools supported by FCC."""

import json
import re
from collections.abc import Iterable


def parse_api_key_pool(value: str) -> tuple[str, ...]:
    """Parse a JSON list or comma/whitespace-separated API-key list.

    Environment variables are strings, and both formats were supported by the
    committed reference implementation. Empty and duplicate items are ignored
    so one credential always has one quota and one failure state.
    """
    raw = value.strip()
    if not raw:
        return ()
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(
            isinstance(key, str) for key in parsed
        ):
            raise ValueError("API key pools must be a JSON array of strings.")
        return _unique_keys(key.strip() for key in parsed)
    return _unique_keys(key for key in re.split(r"[\s,]+", raw) if key)


def _unique_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Keep non-empty credential strings in their first configured order."""
    return tuple(dict.fromkeys(key for key in keys if key))

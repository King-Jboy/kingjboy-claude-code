"""Parsing for the two provider credential pools supported by FCC."""

import json
import re


def parse_api_key_pool(value: str) -> tuple[str, ...]:
    """Parse a JSON list or comma/whitespace-separated API-key list.

    Environment variables are strings, and both formats were supported by the
    committed reference implementation. Empty items are ignored; duplicate
    keys are deliberately retained so configuration order remains literal.
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
        return tuple(key.strip() for key in parsed if key.strip())
    return tuple(key for key in re.split(r"[\s,]+", raw) if key)

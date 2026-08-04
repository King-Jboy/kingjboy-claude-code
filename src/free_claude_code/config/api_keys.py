"""Parsing for provider credentials configured as a pool of equivalent keys."""

import json

KEY_POOL_EXAMPLE = '["key-one", "key-two"]'


def parse_api_key_list(raw: str, *, env_name: str) -> tuple[str, ...]:
    """Parse a JSON array of API keys into an ordered, de-duplicated tuple.

    Blank input yields an empty pool. Malformed input raises instead of falling
    back silently: a pool that quietly shrinks to nothing looks like ordinary
    slowness rather than a configuration error.
    """
    text = raw.strip()
    if not text:
        return ()
    try:
        decoded = json.loads(text)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} is not valid JSON: expected a list of key strings, "
            f"for example {KEY_POOL_EXAMPLE}"
        ) from exc
    if not isinstance(decoded, list):
        raise ValueError(
            f"{env_name} must be a JSON list of key strings, for example "
            f"{KEY_POOL_EXAMPLE}"
        )

    keys: list[str] = []
    seen: set[str] = set()
    for entry in decoded:
        if not isinstance(entry, str):
            raise ValueError(
                f"{env_name} must contain only key strings, for example "
                f"{KEY_POOL_EXAMPLE}"
            )
        key = entry.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return tuple(keys)

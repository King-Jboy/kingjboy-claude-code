"""JSON Schema helpers for text-emitted Anthropic tool input."""

import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import jsonschema_rs
import orjson
import xxhash

_VALIDATOR_CACHE_MAX_ENTRIES = 512
_validator_cache: OrderedDict[bytes, Any] = OrderedDict()
_validator_cache_lock = threading.Lock()


def schema_type(schema: Mapping[str, Any]) -> str | None:
    """Resolve one useful non-null JSON type from a simple or union schema."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        non_null = [
            item for item in declared if isinstance(item, str) and item != "null"
        ]
        if len(non_null) == 1:
            return non_null[0]

    for keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(keyword)
        if not isinstance(alternatives, list):
            continue
        types = {
            nested_type
            for alternative in alternatives
            if isinstance(alternative, Mapping)
            if (nested_type := schema_type(alternative)) not in (None, "null")
        }
        if len(types) == 1:
            return types.pop()
    return None


def coerce_text_argument(value: str, schema: Mapping[str, Any]) -> Any:
    """Decode a textual tool argument according to its declared JSON type."""
    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        for enum_value in enum_values:
            if str(enum_value) == value:
                return enum_value
    if "const" in schema and str(schema["const"]) == value:
        return schema["const"]

    value_type = schema_type(schema)
    stripped = value.strip()
    try:
        if value_type == "integer":
            return int(stripped)
        if value_type == "number":
            number = json.loads(stripped)
            if isinstance(number, int | float) and not isinstance(number, bool):
                return number
            raise ValueError
        if value_type == "boolean":
            if stripped.lower() == "true":
                return True
            if stripped.lower() == "false":
                return False
            raise ValueError
        if value_type == "null":
            if stripped.lower() == "null":
                return None
            raise ValueError
        if value_type == "array":
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            raise ValueError
        if value_type == "object":
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError
    except json.JSONDecodeError as error:
        raise ValueError from error
    return value


def compile_json_schema_validator(schema: Mapping[str, Any]) -> Any:
    """Compile and cache one Rust-backed ``jsonschema_rs`` validator."""
    plain_schema = schema if isinstance(schema, dict) else dict(schema)
    cache_key: bytes | None = None
    try:
        cache_key = xxhash.xxh3_128_digest(
            orjson.dumps(plain_schema, option=orjson.OPT_SORT_KEYS)
        )
    except (TypeError, ValueError):
        cache_key = None

    if cache_key is not None:
        with _validator_cache_lock:
            cached = _validator_cache.get(cache_key)
            if cached is not None:
                _validator_cache.move_to_end(cache_key)
                return cached

    validator = jsonschema_rs.validator_for(plain_schema)

    if cache_key is not None:
        with _validator_cache_lock:
            _validator_cache[cache_key] = validator
            _validator_cache.move_to_end(cache_key)
            while len(_validator_cache) > _VALIDATOR_CACHE_MAX_ENTRIES:
                _validator_cache.popitem(last=False)

    return validator


def arguments_match_schema(
    arguments: dict[str, Any], schema: Mapping[str, Any]
) -> bool:
    """Return whether arguments satisfy a valid JSON Schema."""
    try:
        validator = compile_json_schema_validator(schema)
    except ValueError, TypeError, jsonschema_rs.ValidationError:
        return True
    return bool(validator.is_valid(arguments))

"""High-performance JSON serialization and self-healing parsing utilities."""

import json
from typing import Any

import json_repair
import orjson


def fast_json_dumps(
    obj: Any, *, ensure_ascii: bool = False, indent: int | None = None
) -> str:
    """Serialize an object to a JSON string with Rust-level performance."""
    option = 0
    if indent == 2:
        option |= orjson.OPT_INDENT_2
    try:
        return orjson.dumps(obj, option=option).decode("utf-8")
    except TypeError, ValueError:
        if indent is not None:
            return json.dumps(
                obj, ensure_ascii=ensure_ascii, indent=indent, default=str
            )
        return json.dumps(obj, ensure_ascii=ensure_ascii, default=str)


def fast_json_loads(data: str | bytes) -> Any:
    """Deserialize a JSON string or bytes with Rust-level performance."""
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError, json.JSONDecodeError, TypeError, ValueError:
        return json.loads(data)


def repair_json_loads(data: str | bytes) -> Any:
    """Deserialize a JSON string, self-healing malformed JSON tool inputs/outputs."""
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError, json.JSONDecodeError, TypeError, ValueError:
        try:
            text = data.decode("utf-8") if isinstance(data, bytes) else data
            return json_repair.loads(text)
        except Exception:
            return json.loads(data)


def repair_json_string(data: str) -> str:
    """Self-heal a malformed JSON string into valid JSON."""
    if not data or not data.strip():
        return "{}"
    try:
        orjson.loads(data)
        return data
    except orjson.JSONDecodeError, json.JSONDecodeError, TypeError, ValueError:
        try:
            repaired = json_repair.repair_json(data, return_objects=False)
            if isinstance(repaired, str):
                return repaired
            return fast_json_dumps(repaired)
        except Exception:
            return data


def safe_parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Parse tool arguments safely from str, bytes, dict, or malformed JSON."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)):
        try:
            parsed = repair_json_loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}

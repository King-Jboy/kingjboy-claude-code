"""Tests for Rust-accelerated and self-healing json_utils."""

from free_claude_code.core.json_utils import (
    fast_json_dumps,
    fast_json_loads,
    repair_json_loads,
    repair_json_string,
    safe_parse_tool_arguments,
)


def test_fast_json_dumps_and_loads_roundtrip() -> None:
    payload = {"message": "hello", "count": 42, "items": [1, 2, 3], "flag": True}
    encoded = fast_json_dumps(payload)
    decoded = fast_json_loads(encoded)
    assert decoded == payload


def test_fast_json_dumps_with_indent() -> None:
    payload = {"a": 1}
    encoded = fast_json_dumps(payload, indent=2)
    assert "\n" in encoded
    assert fast_json_loads(encoded) == payload


def test_repair_json_loads_fixes_trailing_commas() -> None:
    malformed = '{"file": "app.py", "line": 10,}'
    parsed = repair_json_loads(malformed)
    assert parsed == {"file": "app.py", "line": 10}


def test_repair_json_loads_fixes_unescaped_quotes_and_newlines() -> None:
    malformed = '{"code": "print("hello world")"}'
    parsed = repair_json_loads(malformed)
    assert isinstance(parsed, dict)
    assert "code" in parsed


def test_repair_json_string_self_heals() -> None:
    malformed = '{"action": "edit", "path": "/src/index.js",}'
    healed = repair_json_string(malformed)
    decoded = fast_json_loads(healed)
    assert decoded == {"action": "edit", "path": "/src/index.js"}


def test_safe_parse_tool_arguments_handles_various_types() -> None:
    assert safe_parse_tool_arguments(None) == {}
    assert safe_parse_tool_arguments("") == {}
    assert safe_parse_tool_arguments({"file": "test.txt"}) == {"file": "test.txt"}
    assert safe_parse_tool_arguments('{"query": "search term",}') == {
        "query": "search term"
    }
    assert safe_parse_tool_arguments(b'{"bytes": true}') == {"bytes": True}
    assert safe_parse_tool_arguments("not-valid-json-at-all") == {}

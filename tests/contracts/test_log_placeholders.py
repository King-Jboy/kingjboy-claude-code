"""Loguru formats messages with ``{}``; printf placeholders are never filled."""

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "free_claude_code"
_LOG_METHODS = {
    "trace",
    "debug",
    "info",
    "success",
    "warning",
    "error",
    "exception",
    "critical",
}
_PRINTF = re.compile(r"%[-+ #0-9.]*[sdrfixX]")


def _printf_log_calls() -> list[str]:
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LOG_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            if _PRINTF.search(node.args[0].value):
                found.append(f"{path.relative_to(_SRC)}:{node.lineno}")
    return found


def test_loguru_calls_use_brace_placeholders() -> None:
    # A "%s" message logs the literal text and silently drops the values.
    assert _printf_log_calls() == []

"""Tests for streaming recovery body generation."""

from free_claude_code.core.anthropic.streaming.recovery import make_text_recovery_body


def test_make_text_recovery_body_removes_parallel_tool_calls() -> None:
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "test"}}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    recovery = make_text_recovery_body(body, "partial text")
    assert "tools" not in recovery
    assert "tool_choice" not in recovery
    assert "parallel_tool_calls" not in recovery

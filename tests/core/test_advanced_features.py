"""Unit tests for semantic tool selection."""

from free_claude_code.core.tools.semantic_selector import (
    select_relevant_tools,
)


def test_semantic_selector_preserves_all_tools_when_under_threshold() -> None:
    tools = [
        {"function": {"name": f"tool_{i}", "description": "desc"}} for i in range(10)
    ]
    selected = select_relevant_tools(tools, "read this file")
    assert len(selected) == 10


def test_semantic_selector_always_preserves_core_and_explicit_tools() -> None:
    # 25 total tools (> 20)
    tools = [
        {"function": {"name": "Read", "description": "Read file"}},
        {"function": {"name": "Write", "description": "Write file"}},
        {"function": {"name": "Grep", "description": "Search code"}},
        {
            "function": {
                "name": "mcp_github_issues",
                "description": "Read GitHub issues",
            }
        },
        {"function": {"name": "mcp_sqlite_query", "description": "Query database"}},
    ]
    # Add dummy tools to exceed 20
    tools.extend(
        {"function": {"name": f"unrelated_tool_{i}", "description": "Unrelated"}}
        for i in range(25)
    )

    # Prompt explicitly asks for sqlite
    selected = select_relevant_tools(
        tools,
        "Please query the database using mcp_sqlite_query",
        explicit_tool_names=frozenset({"mcp_sqlite_query"}),
    )

    names = {
        t.get("function", {}).get("name") if isinstance(t, dict) else t.name
        for t in selected
    }

    # Core tools MUST be preserved
    assert "Read" in names
    assert "Write" in names
    assert "Grep" in names

    # Explicitly requested / matched tool MUST be preserved
    assert "mcp_sqlite_query" in names

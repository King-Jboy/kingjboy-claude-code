"""Smart semantic tool filtering with 100% core and pinned tool preservation."""

from collections.abc import Sequence
from typing import Any

from loguru import logger

_CORE_PINNED_TOOLS = frozenset(
    {
        "read",
        "write",
        "edit",
        "grep",
        "glob",
        "bash",
        "task",
        "notebook",
        "terminal",
        "view",
        "replace",
        "search",
        "patch",
        "str_replace",
    }
)

MAX_UNFILTERED_TOOLS = 20


def select_relevant_tools(
    tools: Sequence[Any],
    prompt_text: str,
    *,
    explicit_tool_names: frozenset[str] = frozenset(),
    top_k: int = 15,
) -> list[Any]:
    """Select the most relevant tools for a prompt while preserving core and explicitly named tools.

    Guarantees:
    1. If total tools <= MAX_UNFILTERED_TOOLS (20), all tools are returned untouched (0 filtering).
    2. Core agent tools (Read, Write, Edit, Grep, Bash, Task, etc.) are always preserved 100%.
    3. Any tool mentioned by name in prompt_text or explicit_tool_names is always preserved 100%.
    """
    if len(tools) <= MAX_UNFILTERED_TOOLS:
        return list(tools)

    prompt_lower = prompt_text.lower()
    pinned: list[Any] = []
    candidates: list[Any] = []

    for tool in tools:
        name = getattr(tool, "name", "") or (
            tool.get("function", {}).get("name", "") if isinstance(tool, dict) else ""
        )
        name_lower = name.lower()

        # Check if core tool, explicitly pinned, or mentioned in user prompt
        if (
            name_lower in _CORE_PINNED_TOOLS
            or name_lower in explicit_tool_names
            or name in explicit_tool_names
            or (name_lower and name_lower in prompt_lower)
        ):
            pinned.append(tool)
        else:
            candidates.append(tool)

    # If all tools were pinned, return all
    if not candidates:
        return list(tools)

    # Score remaining candidates against prompt keywords
    scored_candidates: list[tuple[float, Any]] = []
    prompt_words = set(prompt_lower.split())

    for tool in candidates:
        name = getattr(tool, "name", "") or (
            tool.get("function", {}).get("name", "") if isinstance(tool, dict) else ""
        )
        description = getattr(tool, "description", "") or (
            tool.get("function", {}).get("description", "")
            if isinstance(tool, dict)
            else ""
        )
        tool_text = f"{name} {description}".lower()

        # Calculate keyword match score
        score = sum(1.0 for word in prompt_words if len(word) > 2 and word in tool_text)
        scored_candidates.append((score, tool))

    # Sort descending by relevance score and pick top_k
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    selected_candidates = [item[1] for item in scored_candidates[:top_k]]

    logger.debug(
        "Semantic tool selector: kept {}/{} tools ({} pinned, {} selected)",
        len(pinned) + len(selected_candidates),
        len(tools),
        len(pinned),
        len(selected_candidates),
    )
    return pinned + selected_candidates

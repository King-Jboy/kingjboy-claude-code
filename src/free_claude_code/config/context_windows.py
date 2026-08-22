"""Per-model context windows recorded by the ``fcc-context`` command.

Providers almost never publish a context length on the wire, so the window a
client should compact against cannot be discovered at runtime. The probe script
records it once into ``~/.fcc/context.md``; this module reads that table back so
switching ``MODEL`` also switches the advertised window, with no second setting
to keep in sync by hand.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .constants import DEFAULT_CLIENT_CONTEXT_WINDOW
from .model_refs import ChatModelConfig, configured_chat_model_refs
from .paths import config_dir_path

CONTEXT_WINDOWS_FILENAME = "context.md"

_PROVIDER_HEADING = re.compile(r"^##\s+(?P<provider>\S+)")
_TABLE_ROW = re.compile(
    r"^\|\s*`(?P<model>[^`]+)`\s*\|\s*(?P<context>[^|]+?)\s*\|\s*(?P<source>[^|]+?)\s*\|$"
)


@dataclass(frozen=True, slots=True)
class ResolvedContextWindow:
    """The window to advertise, and where the number came from."""

    value: int
    source: str
    # The routed ref whose recorded window was used, when one was.
    model_ref: str | None = None


def context_windows_path() -> Path:
    """Return the managed context-window table path."""

    return config_dir_path() / CONTEXT_WINDOWS_FILENAME


def load_context_windows(path: Path | None = None) -> dict[str, int]:
    """Parse ``context.md`` into ``{provider/model: tokens}``.

    A missing or malformed file yields an empty mapping rather than raising:
    the table is an optimisation, and losing it should degrade to the
    conservative default rather than stop a CLI from launching.
    """
    target = path or context_windows_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}

    windows: dict[str, int] = {}
    provider = ""
    for line in text.splitlines():
        if heading := _PROVIDER_HEADING.match(line):
            provider = heading.group("provider")
            continue
        row = _TABLE_ROW.match(line.strip())
        if not row or not provider:
            continue
        raw = row.group("context").replace(",", "").strip()
        if raw.isdigit():
            windows[f"{provider}/{row.group('model')}"] = int(raw)
    return windows


def recorded_route_windows(
    settings: ChatModelConfig,
    *,
    windows: dict[str, int] | None = None,
) -> list[tuple[str, int]]:
    """Return each configured route's recorded window, in configuration order.

    This is the display half of the table: the resolver picks one number for
    the CLI, but the operator deciding whether that number is right wants to
    see every route it was chosen from.
    """
    recorded = load_context_windows() if windows is None else windows
    return [
        (ref.model_ref, recorded[ref.model_ref])
        for ref in configured_chat_model_refs(settings)
        if ref.model_ref in recorded
    ]


def resolve_client_context_window(
    settings: ChatModelConfig,
    *,
    configured: int | None,
    windows: dict[str, int] | None = None,
) -> ResolvedContextWindow:
    """Return the window to advertise to a launched client CLI.

    An explicit ``CLIENT_CONTEXT_WINDOW`` always wins. Otherwise the smallest
    recorded window across every configured route is used: Claude Code compacts
    against one number for the whole session but can be pointed at any routed
    tier, so the largest safe value is the smallest route's ceiling.
    """
    if configured is not None:
        return ResolvedContextWindow(configured, "CLIENT_CONTEXT_WINDOW")

    recorded = load_context_windows() if windows is None else windows
    known = [
        (recorded[ref.model_ref], ref.model_ref)
        for ref in configured_chat_model_refs(settings)
        if ref.model_ref in recorded
    ]
    if not known:
        return ResolvedContextWindow(DEFAULT_CLIENT_CONTEXT_WINDOW, "default")
    value, model_ref = min(known)
    return ResolvedContextWindow(value, CONTEXT_WINDOWS_FILENAME, model_ref)

"""Lightweight entry points for installed Free Claude Code commands."""

import sys
from collections.abc import Sequence

from free_claude_code.core.version import package_version


def serve(argv: Sequence[str] | None = None) -> None:
    """Start the FastAPI server (registered as ``fcc-server``)."""
    if _print_version_if_requested(argv):
        return

    # Keep the server composition root off metadata-only command paths.
    from free_claude_code.cli.commands import serve as run_server

    run_server()


def doctor(argv: Sequence[str] | None = None) -> None:
    """Diagnose the install (registered as ``fcc-doctor``)."""
    if _print_version_if_requested(argv):
        return

    # Imported lazily so --version stays a metadata-only path.
    from free_claude_code.cli.doctor import run

    raise SystemExit(run(argv))


def context(argv: Sequence[str] | None = None) -> None:
    """Record model context windows (registered as ``fcc-context``)."""
    if _print_version_if_requested(argv):
        return

    # Imported lazily so --version stays a metadata-only path.
    from free_claude_code.cli.context_scan import run

    raise SystemExit(run(argv))


def extension(argv: Sequence[str] | None = None) -> None:
    """Locate the Chrome side panel (registered as ``fcc-extension``)."""
    if _print_version_if_requested(argv):
        return

    # Imported lazily so --version stays a metadata-only path.
    from free_claude_code.cli.extension import run

    raise SystemExit(run(argv))


def _print_version_if_requested(argv: Sequence[str] | None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    if "--version" not in args:
        return False
    print(f"free-claude-code {package_version()}")
    return True

"""Installed `fcc-extension` command: locate and configure the Chrome side panel."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings

EXTENSION_ASSETS_DIRNAME = "extension_assets"
MANIFEST_FILENAME = "manifest.json"

# Every file manifest.json points at. Packaging drops non-Python assets more
# easily than it drops modules, and Chrome's failure for a missing file is a
# generic "Could not load manifest" with no name in it.
REQUIRED_ASSETS = (
    MANIFEST_FILENAME,
    "console_probe.js",
    "page_tools.js",
    "service_worker.js",
    "sidepanel.css",
    "sidepanel.html",
    "sidepanel.js",
)

_USAGE = (
    "Usage: fcc-extension [--path] [--show-token] [--json]\n\n"
    "  --path        Print only the extension directory, for scripting.\n"
    "  --show-token  Print the proxy token instead of masking it.\n"
    "  --json        Emit the connection details as JSON.\n"
)


def extension_dir() -> Path:
    """Return the absolute installed path to the unpacked Chrome extension."""

    return (Path(__file__).parent / EXTENSION_ASSETS_DIRNAME).resolve()


def missing_assets(directory: Path | None = None) -> tuple[str, ...]:
    """Return the manifest-referenced files absent from the extension directory."""

    root = extension_dir() if directory is None else directory
    return tuple(name for name in REQUIRED_ASSETS if not (root / name).is_file())


def connection_details(settings: Settings) -> dict[str, str | bool]:
    """Return what the side panel's settings pane needs to reach this proxy."""

    return {
        "extension_dir": str(extension_dir()),
        "proxy_url": local_proxy_root_url(settings),
        "auth_token": settings.anthropic_auth_token.strip(),
        "auth_required": bool(settings.anthropic_auth_token.strip()),
    }


def _token_line(token: str, *, reveal: bool) -> str:
    if not token:
        return "  Auth token   (none - the proxy accepts unauthenticated requests)"
    if reveal:
        return f"  Auth token   {token}"
    # Masked by default: this prints in terminals that get shared and recorded,
    # and the panel stores the token anyway once it is pasted in once.
    return "  Auth token   (set - re-run with --show-token to print it)"


def _report(details: dict[str, str | bool], *, reveal: bool) -> str:
    token = str(details["auth_token"])
    return "\n".join(
        (
            "Free Claude Code - Chrome side panel",
            "",
            "Load the extension (once):",
            "  1. Open chrome://extensions",
            "  2. Turn on Developer mode",
            "  3. Load unpacked, and choose:",
            f"     {details['extension_dir']}",
            "",
            "Then open the side panel from the toolbar and paste:",
            f"  Proxy URL    {details['proxy_url']}",
            _token_line(token, reveal=reveal),
            "",
            "The panel needs fcc-server running to answer anything.",
        )
    )


def run(argv: Sequence[str] | None = None) -> int:
    """Report how to load the Chrome extension and return a shell exit code."""

    args = sys.argv[1:] if argv is None else list(argv)
    if "--help" in args or "-h" in args:
        print(_USAGE)
        return 0

    if absent := missing_assets():
        print(
            "The bundled Chrome extension is incomplete; reinstall FCC. Missing: "
            + ", ".join(absent),
            file=sys.stderr,
        )
        return 1

    if "--path" in args:
        print(extension_dir())
        return 0

    try:
        settings = Settings()
    except ValidationError:
        # Same reasoning as fcc-doctor: the URL and token come from config, so
        # unreadable config means we cannot answer, but a traceback is not an
        # answer either. The directory is still useful on its own.
        print(
            "Configuration could not be loaded, so the proxy URL is unknown.\n"
            "Run fcc-doctor to see why. The extension directory is:\n"
            f"  {extension_dir()}",
            file=sys.stderr,
        )
        return 1

    details = connection_details(settings)
    if "--json" in args:
        if "--show-token" not in args:
            details["auth_token"] = ""
        print(json.dumps(details, indent=2))
        return 0

    print(_report(details, reveal="--show-token" in args))
    return 0

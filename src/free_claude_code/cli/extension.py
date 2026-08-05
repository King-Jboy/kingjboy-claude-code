"""Installed `fcc-extension` command: locate and configure the Chrome side panel."""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from free_claude_code.cli.bridge import shell_root
from free_claude_code.cli.native_host import RegistrationError, stored_manifest_path
from free_claude_code.cli.native_host import install as install_bridge
from free_claude_code.cli.native_host import uninstall as uninstall_bridge
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
    "shell_tool.js",
    "sidepanel.css",
    "sidepanel.html",
    "sidepanel.js",
)

_USAGE = (
    "Usage: fcc-extension [--path] [--show-token] [--json]\n"
    "       fcc-extension install --extension-id ID\n"
    "       fcc-extension uninstall\n\n"
    "  --path            Print only the extension directory, for scripting.\n"
    "  --show-token      Print the proxy token instead of masking it.\n"
    "  --json            Emit the connection details as JSON.\n\n"
    "  install           Let the side panel run shell commands, by registering\n"
    "                    the fcc-bridge host for this one extension ID.\n"
    "  uninstall         Remove that registration.\n"
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


def _shell_lines(settings: Settings) -> tuple[str, ...]:
    """Describe the command bridge: whether it is registered, and whether it is on."""

    registered = stored_manifest_path().is_file()
    if not registered:
        return (
            "",
            "To let the panel run shell commands (optional):",
            "  fcc-extension install --extension-id ID   (the ID is on the card)",
        )
    if not settings.browser_shell_enabled:
        # Registered but disabled is the deliberate default, not a half-install.
        return (
            "",
            "Command bridge  registered, but disabled.",
            "  Set BROWSER_SHELL_ENABLED=true in ~/.fcc/.env to allow commands.",
        )
    return (
        "",
        "Command bridge  enabled.",
        f"  Commands run under {shell_root(settings)} and need your approval each time.",
    )


def _report(details: dict[str, str | bool], settings: Settings, *, reveal: bool) -> str:
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
            *_shell_lines(settings),
            "",
            "The panel needs fcc-server running to answer anything.",
        )
    )


def _run_install(args: Sequence[str]) -> int:
    if "--extension-id" not in args:
        print(
            "install needs the extension ID: fcc-extension install --extension-id ID\n"
            "Load the extension at chrome://extensions first; the ID is on its card.",
            file=sys.stderr,
        )
        return 2

    position = list(args).index("--extension-id") + 1
    if position >= len(args):
        print("--extension-id needs a value.", file=sys.stderr)
        return 2

    try:
        registered = install_bridge(args[position])
    except RegistrationError as error:
        print(str(error), file=sys.stderr)
        return 1

    if not registered:
        print(
            "No Chromium-family browser profile was found to register with.",
            file=sys.stderr,
        )
        return 1

    print(
        "\n".join(
            (
                f"Registered the command bridge with: {', '.join(registered)}",
                "",
                "Two things still have to be true before any command runs:",
                "  1. BROWSER_SHELL_ENABLED=true in ~/.fcc/.env",
                "  2. You approve each command in the side panel",
                "",
                "Restart the browser so it picks up the new host.",
            )
        )
    )
    return 0


def _run_uninstall() -> int:
    removed = uninstall_bridge()
    if removed:
        print(f"Removed the command bridge from: {', '.join(removed)}")
    else:
        print("The command bridge was not registered.")
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    """Report how to load the Chrome extension and return a shell exit code."""

    args = sys.argv[1:] if argv is None else list(argv)
    if "--help" in args or "-h" in args:
        print(_USAGE)
        return 0

    if args and args[0] == "uninstall":
        return _run_uninstall()
    if args and args[0] == "install":
        return _run_install(args)

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

    print(_report(details, settings, reveal="--show-token" in args))
    return 0

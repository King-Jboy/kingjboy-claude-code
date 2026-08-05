"""Registration of the `fcc-bridge` native messaging host with the browser.

A host is reachable only if the browser can find a manifest naming it, and the
manifest's ``allowed_origins`` decides which extension may talk to it. That list
is the security boundary this whole feature rests on, so it holds exactly one
extension ID -- the one the user passed -- and never a wildcard.

Where the manifest has to live differs by platform. Windows keeps a registry
value pointing at a file anywhere on disk; macOS and Linux require the file
itself to sit in a per-browser directory.
"""

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from free_claude_code.cli.bridge import HOST_NAME
from free_claude_code.config.paths import config_dir_path

BRIDGE_EXECUTABLE = "fcc-bridge"
NATIVE_HOST_DIRNAME = "native-host"

# Chrome derives an unpacked extension's ID from its directory, so it is stable
# per machine but differs across machines. 32 letters, a through p.
EXTENSION_ID = re.compile(r"^[a-p]{32}$")


@dataclass(frozen=True, slots=True)
class BrowserTarget:
    """One Chromium-family browser's native messaging host location."""

    name: str
    # Windows: the registry subkey under HKCU. Otherwise: the manifest directory.
    location: str


_WINDOWS_TARGETS = (
    BrowserTarget("Chrome", r"Software\Google\Chrome\NativeMessagingHosts"),
    BrowserTarget("Chromium", r"Software\Chromium\NativeMessagingHosts"),
    BrowserTarget("Edge", r"Software\Microsoft\Edge\NativeMessagingHosts"),
)

_MACOS_TARGETS = (
    BrowserTarget("Chrome", "Library/Application Support/Google/Chrome"),
    BrowserTarget("Chromium", "Library/Application Support/Chromium"),
    BrowserTarget("Edge", "Library/Application Support/Microsoft Edge"),
)

_LINUX_TARGETS = (
    BrowserTarget("Chrome", ".config/google-chrome"),
    BrowserTarget("Chromium", ".config/chromium"),
    BrowserTarget("Edge", ".config/microsoft-edge"),
)


class RegistrationError(Exception):
    """Registration could not proceed, with a reason worth printing."""


def browser_targets(platform: str | None = None) -> tuple[BrowserTarget, ...]:
    """Return the Chromium-family browsers this platform can register with."""

    match platform or sys.platform:
        case "win32":
            return _WINDOWS_TARGETS
        case "darwin":
            return _MACOS_TARGETS
        case _:
            return _LINUX_TARGETS


def bridge_executable() -> str:
    """Return the absolute path to the installed fcc-bridge executable."""

    resolved = shutil.which(BRIDGE_EXECUTABLE)
    if resolved is None:
        raise RegistrationError(
            f"Could not find {BRIDGE_EXECUTABLE} on PATH. Reinstall FCC so the "
            "entry point is created, then run this again."
        )
    return str(Path(resolved).resolve())


def host_manifest(extension_id: str, executable: str) -> dict[str, object]:
    """Return the native messaging host manifest for one extension."""

    return {
        "name": HOST_NAME,
        "description": "Free Claude Code command bridge",
        "path": executable,
        "type": "stdio",
        # Exactly one origin. A second entry here would hand the same shell to
        # another extension, which is the failure this file exists to prevent.
        "allowed_origins": [f"chrome-extension://{extension_id}/"],
    }


def stored_manifest_path() -> Path:
    """Return where FCC keeps its own copy of the host manifest."""

    return config_dir_path() / NATIVE_HOST_DIRNAME / f"{HOST_NAME}.json"


def validate_extension_id(extension_id: str) -> str:
    """Return the extension ID, or explain why it cannot be one."""

    candidate = extension_id.strip()
    if not EXTENSION_ID.match(candidate):
        raise RegistrationError(
            f"{candidate!r} is not a Chrome extension ID. Load the extension at "
            "chrome://extensions and copy the 32-letter ID shown on its card."
        )
    return candidate


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _install_windows(manifest: dict[str, object]) -> tuple[str, ...]:
    # Windows-only stdlib module; importing it at module scope would break
    # every other platform's import of this file.
    import winreg

    path = stored_manifest_path()
    _write_manifest(path, manifest)

    registered: list[str] = []
    for target in browser_targets("win32"):
        try:
            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER, f"{target.location}\\{HOST_NAME}"
            ) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(path))
        except OSError:
            # A browser that is not installed still registers cleanly under
            # HKCU, so a failure here is unusual and not worth aborting for.
            continue
        registered.append(target.name)
    return tuple(registered)


def _install_posix(manifest: dict[str, object], platform: str) -> tuple[str, ...]:
    _write_manifest(stored_manifest_path(), manifest)

    registered: list[str] = []
    for target in browser_targets(platform):
        directory = Path.home() / target.location
        # Only register with browsers that are actually installed: creating the
        # profile directory for an absent browser litters the home directory.
        if not directory.is_dir():
            continue
        _write_manifest(
            directory / "NativeMessagingHosts" / f"{HOST_NAME}.json", manifest
        )
        registered.append(target.name)
    return tuple(registered)


def install(extension_id: str) -> tuple[str, ...]:
    """Register the bridge for one extension and return the browsers reached."""

    manifest = host_manifest(validate_extension_id(extension_id), bridge_executable())
    if sys.platform == "win32":
        return _install_windows(manifest)
    return _install_posix(manifest, sys.platform)


def _uninstall_windows() -> tuple[str, ...]:
    import winreg

    removed: list[str] = []
    for target in browser_targets("win32"):
        try:
            winreg.DeleteKey(
                winreg.HKEY_CURRENT_USER, f"{target.location}\\{HOST_NAME}"
            )
        except OSError:
            continue
        removed.append(target.name)
    return tuple(removed)


def _uninstall_posix(platform: str) -> tuple[str, ...]:
    removed: list[str] = []
    for target in browser_targets(platform):
        path = (
            Path.home() / target.location / "NativeMessagingHosts" / f"{HOST_NAME}.json"
        )
        if not path.is_file():
            continue
        path.unlink()
        removed.append(target.name)
    return tuple(removed)


def uninstall() -> tuple[str, ...]:
    """Remove the bridge registration and return the browsers it was taken from."""

    removed = (
        _uninstall_windows()
        if sys.platform == "win32"
        else _uninstall_posix(sys.platform)
    )
    stored = stored_manifest_path()
    if stored.is_file():
        stored.unlink()
    return removed

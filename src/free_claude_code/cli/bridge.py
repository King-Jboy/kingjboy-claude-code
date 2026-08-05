"""Chrome Native Messaging host: `fcc-bridge`.

Chrome launches this as a subprocess and speaks a length-prefixed JSON protocol
over stdin/stdout. It exists because Manifest V3 cannot spawn a process, and the
alternative -- an exec endpoint on fcc-server -- would be remote code execution:
that server binds 0.0.0.0 by default and skips auth entirely when
ANTHROPIC_AUTH_TOKEN is blank. Native messaging moves the boundary from the
network to the OS, where only Chrome can reach it and only for the one extension
ID named in the host manifest's allowed_origins.

Three further gates sit in front of execution, because "only Chrome can reach
it" is not on its own a reason to run arbitrary commands:

  * BROWSER_SHELL_ENABLED is false unless the user sets it. Installing the
    extension and registering this host is not consent to run commands.
  * Every command runs inside BROWSER_SHELL_ROOT (default: home) and nowhere
    else, so a traversal in the requested cwd is refused rather than followed.
  * The side panel asks the user to approve each command before sending it.
    That approval is the real control; everything here is defence in depth.

Every accepted command is appended to ~/.fcc/logs/bridge.log.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from free_claude_code.config.paths import bridge_log_path
from free_claude_code.config.settings import Settings

HOST_NAME = "com.free_claude_code.bridge"

# Chrome permits 4GB inbound, which is not a size any command line needs.
MAX_REQUEST_BYTES = 1_048_576
# Chrome caps host-to-extension messages at 1MB, and the output becomes prompt
# text besides. Truncating here keeps both limits honest.
MAX_OUTPUT_CHARS = 32_000
DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600

_LENGTH_PREFIX = struct.Struct("<I")


class BridgeError(Exception):
    """A request that must be refused with a reason rather than executed."""


@dataclass(frozen=True, slots=True)
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool


def default_shell() -> tuple[str, ...]:
    """Return the argv prefix that runs a command string on this platform."""

    if sys.platform == "win32":
        # cmd.exe is what shell=True would pick and it is the wrong answer on a
        # machine where the user's shell is PowerShell: half the commands they
        # would type by hand fail in it.
        for candidate in ("pwsh", "powershell"):
            if resolved := shutil.which(candidate):
                return (resolved, "-NoProfile", "-NonInteractive", "-Command")
        return (os.environ.get("COMSPEC", "cmd.exe"), "/c")
    return (os.environ.get("SHELL") or "/bin/sh", "-c")


def shell_root(settings: Settings) -> Path:
    """Return the directory tree commands are confined to."""

    configured = settings.browser_shell_root.strip()
    return Path(configured).expanduser().resolve() if configured else Path.home()


def resolve_cwd(requested: str | None, root: Path) -> Path:
    """Return a working directory inside ``root``, or refuse."""

    if not requested:
        return root

    candidate = Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    # resolve() collapses .. before the comparison, so "root/../etc" is caught
    # rather than passed through as a path that merely starts with root.
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise BridgeError(
            f"{resolved} is outside BROWSER_SHELL_ROOT ({root}). "
            "Widen BROWSER_SHELL_ROOT or choose a directory inside it."
        )
    if not resolved.is_dir():
        raise BridgeError(f"{resolved} is not a directory.")
    return resolved


def _clamp_timeout(requested: object) -> int:
    if not isinstance(requested, int) or requested <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return min(requested, MAX_TIMEOUT_SECONDS)


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return f"{text[:MAX_OUTPUT_CHARS]}\n[truncated]", True


def run_shell_command(command: str, *, cwd: Path, timeout: int) -> ShellResult:
    """Run one command string in ``cwd`` and return its captured result."""

    argv = [*default_shell(), command]
    try:
        completed = subprocess.run(
            argv,
            check=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            # No stdin: a command that prompts would otherwise hang until the
            # timeout with nobody able to answer it.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise BridgeError(f"Command timed out after {timeout}s.") from None
    except OSError as error:
        raise BridgeError(f"Could not start a shell: {error}") from None

    stdout, stdout_cut = _truncate(completed.stdout or "")
    stderr, stderr_cut = _truncate(completed.stderr or "")
    return ShellResult(
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        truncated=stdout_cut or stderr_cut,
    )


def audit(command: str, cwd: Path, outcome: str) -> None:
    """Append one command to the audit log, best effort."""

    try:
        path = bridge_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with path.open("a", encoding="utf-8") as log:
            log.write(f"{stamp}\t{outcome}\t{cwd}\t{command}\n")
    except OSError:
        # An unwritable log must not stop a command the user approved, and
        # there is nowhere to report it: stdout is the protocol channel.
        pass


def handle(request: dict[str, object], settings: Settings) -> dict[str, object]:
    """Answer one decoded request message."""

    kind = request.get("type")
    root = shell_root(settings)

    if kind == "ping":
        return {
            "ok": True,
            "type": "pong",
            "enabled": settings.browser_shell_enabled,
            "root": str(root),
            "shell": default_shell()[0],
        }

    if kind != "run":
        return {"ok": False, "error": f"Unknown request type {kind!r}."}

    if not settings.browser_shell_enabled:
        return {
            "ok": False,
            "error": (
                "Command execution is disabled. Set BROWSER_SHELL_ENABLED=true in "
                "~/.fcc/.env to allow the side panel to run commands."
            ),
        }

    command = request.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"ok": False, "error": "No command supplied."}

    requested_cwd = request.get("cwd")
    try:
        cwd = resolve_cwd(
            requested_cwd if isinstance(requested_cwd, str) else None, root
        )
        result = run_shell_command(
            command, cwd=cwd, timeout=_clamp_timeout(request.get("timeout"))
        )
    except BridgeError as error:
        audit(command, root, "refused")
        return {"ok": False, "error": str(error)}

    audit(command, cwd, f"exit={result.exit_code}")
    return {
        "ok": True,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "truncated": result.truncated,
        "cwd": str(cwd),
    }


def read_message(stream: BinaryIO) -> dict[str, object] | None:
    """Read one length-prefixed message, or None once Chrome closes the pipe."""

    header = stream.read(_LENGTH_PREFIX.size)
    if len(header) < _LENGTH_PREFIX.size:
        return None

    (length,) = _LENGTH_PREFIX.unpack(header)
    if length > MAX_REQUEST_BYTES:
        raise BridgeError(
            f"Request of {length} bytes exceeds the {MAX_REQUEST_BYTES} limit."
        )

    payload = stream.read(length)
    if len(payload) < length:
        return None

    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise BridgeError("Request was not a JSON object.")
    return decoded


def write_message(stream: BinaryIO, message: dict[str, object]) -> None:
    """Write one length-prefixed message and flush it."""

    payload = json.dumps(message).encode("utf-8")
    stream.write(_LENGTH_PREFIX.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def run(argv: Sequence[str] | None = None) -> int:
    """Serve Chrome's native messaging protocol until the pipe closes."""

    args = sys.argv[1:] if argv is None else list(argv)
    if "--help" in args or "-h" in args:
        print(
            "Usage: fcc-bridge\n\n"
            "  Chrome Native Messaging host for the Free Claude Code side panel.\n"
            "  Chrome launches it; there is nothing to run by hand.\n"
            "  Register it with: fcc-extension install --extension-id ID\n"
        )
        return 0

    # stdout is the protocol channel. A stray print anywhere in the import
    # graph would corrupt a frame and leave Chrome reporting a generic native
    # host failure, so the real handle is taken away and stdout aliased to
    # stderr for the rest of the process.
    channel = sys.stdout.buffer
    sys.stdout = sys.stderr

    settings = Settings()
    while True:
        try:
            request = read_message(sys.stdin.buffer)
        except (BridgeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            write_message(channel, {"ok": False, "error": str(error)})
            continue

        if request is None:
            return 0
        write_message(channel, handle(request, settings))

"""Track and clean up spawned CLI subprocesses.

This is a safety net for cases where the server is interrupted (Ctrl+C) and the
FastAPI lifespan cleanup doesn't run to completion. We only track processes we
spawn so we don't accidentally kill unrelated system processes.
"""

import atexit
import os
import signal
import subprocess
import threading

from loguru import logger

_lock = threading.Lock()
_pids: set[int] = set()
_atexit_registered = False


def ensure_atexit_registered() -> None:
    global _atexit_registered
    with _lock:
        if _atexit_registered:
            return
        atexit.register(kill_all_best_effort)
        _atexit_registered = True


def register_pid(pid: int) -> None:
    if not pid:
        return
    ensure_atexit_registered()
    with _lock:
        _pids.add(int(pid))


def unregister_pid(pid: int) -> None:
    if not pid:
        return
    with _lock:
        _pids.discard(int(pid))


def kill_pid_tree_best_effort(pid: int) -> None:
    """Kill a tracked process and its children where the platform supports it."""

    _signal_pid_tree_best_effort(pid, signal.SIGTERM)


def force_kill_pid_tree_best_effort(pid: int) -> None:
    """Force-kill a tracked process tree after graceful termination timed out."""

    # Windows has no SIGKILL constant, but its taskkill branch already uses /F.
    _signal_pid_tree_best_effort(pid, getattr(signal, "SIGKILL", signal.SIGTERM))


def _signal_pid_tree_best_effort(pid: int, sig: int) -> None:
    """Send ``sig`` to one tracked process tree without raising cleanup errors."""
    if not pid:
        return
    if os.name == "nt":
        try:
            # /T kills child processes, /F forces termination.
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as e:
            logger.debug("process_registry: taskkill failed pid=%s: %s", pid, e)
        return

    # All tracked POSIX children start a fresh session, so their process group
    # contains the command and its descendants without touching this process.
    try:
        os.killpg(os.getpgid(pid), sig)
    except Exception as e:
        logger.debug("process_registry: signal failed pid=%s: %s", pid, e)


def kill_all_best_effort() -> None:
    """Kill any still-running registered pids (best-effort)."""
    with _lock:
        pids = list(_pids)
        _pids.clear()

    for pid in pids:
        kill_pid_tree_best_effort(pid)

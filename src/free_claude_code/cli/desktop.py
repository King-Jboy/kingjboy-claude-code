"""Platform-neutral lifecycle for the FCC desktop shell."""

import threading
from collections.abc import Callable
from typing import Protocol

from free_claude_code.cli.commands import (
    ServerStatus,
    ServerSupervisor,
    load_server_settings,
    open_admin_when_ready,
    schedule_open_admin_browser,
)
from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.config.paths import config_dir_path
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import get_settings
from free_claude_code.core.interprocess_lock import InterprocessFileLock


class DesktopTray(Protocol):
    """UI loop owned by the platform tray adapter."""

    def run(self) -> None: ...

    def stop(self) -> None: ...


class DesktopTrayFactory(Protocol):
    """Construct a tray adapter around a desktop controller."""

    def __call__(self, controller: DesktopController) -> DesktopTray: ...


class ServerOwner(Protocol):
    """Server lifecycle used by the desktop controller."""

    @property
    def status(self) -> ServerStatus: ...

    def schedule_run(self) -> bool: ...

    def run(self, *, open_admin_browser: bool | None = None) -> None: ...

    def request_restart(self) -> bool: ...

    def request_stop(self) -> None: ...


class DesktopController:
    """Coordinate one tray loop with one in-process FCC server owner."""

    def __init__(
        self,
        supervisor_factory: Callable[[], ServerOwner],
        tray_factory: DesktopTrayFactory,
        open_admin: Callable[[], None],
    ) -> None:
        self._supervisor_factory = supervisor_factory
        self._supervisor = supervisor_factory()
        self._open_admin = open_admin
        self._thread_lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._tray = tray_factory(self)

    @property
    def status(self) -> ServerStatus:
        return self._supervisor.status

    def run(self) -> None:
        """Run the tray on this thread and the FCC server on its owned worker."""

        self._start_server()
        try:
            self._tray.run()
        finally:
            self._supervisor.request_stop()
            self._tray.stop()
            with self._thread_lock:
                thread = self._server_thread
            if thread is not None:
                thread.join()

    def open_admin(self) -> None:
        self._open_admin()

    def restart_server(self) -> None:
        """Restart an active server or relaunch one that exited unexpectedly."""

        with self._thread_lock:
            thread = self._server_thread
        if thread is not None and thread.is_alive():
            self._supervisor.request_restart()
            return
        self._start_server()

    def quit(self) -> None:
        """Close the server gracefully and end the platform tray loop."""

        self._supervisor.request_stop()
        self._tray.stop()

    def _start_server(self) -> None:
        with self._thread_lock:
            if self._server_thread is not None and self._server_thread.is_alive():
                return
            if not self._supervisor.schedule_run():
                # A supervisor that has been asked to stop stays stopped:
                # schedule_run and request_restart both refuse from then on.
                # Replacing it is what lets the tray bring the server back after
                # Admin stops it, instead of leaving a live tray over a server
                # that can never start again.
                self._supervisor = self._supervisor_factory()
                if not self._supervisor.schedule_run():
                    return
            supervisor = self._supervisor
            self._server_thread = threading.Thread(
                target=self._run_server,
                args=(supervisor,),
                name="fcc-desktop-server",
            )
            self._server_thread.start()

    def _run_server(self, supervisor: ServerOwner) -> None:
        supervisor.run(open_admin_browser=False)


def launch_desktop(tray_factory: DesktopTrayFactory) -> None:
    """Start the singleton desktop host or focus the already running FCC UI."""

    settings = load_server_settings()
    instance_lock = InterprocessFileLock(config_dir_path() / "desktop.lock")
    if not instance_lock.acquire():
        open_admin_when_ready(settings)
        return

    try:
        if preflight_proxy(local_proxy_root_url(settings)) is None:
            open_admin_when_ready(settings)
            return

        def open_current_admin() -> None:
            schedule_open_admin_browser(get_settings())

        DesktopController(
            lambda: ServerSupervisor(console_logging=False),
            tray_factory,
            open_current_admin,
        ).run()
    finally:
        instance_lock.release()

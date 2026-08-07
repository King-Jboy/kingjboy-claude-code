"""pywebview shell hosting the Admin UI in its own window.

The desktop app used to be a tray icon that opened a browser tab, so the
product was two things wearing one name. This window is the app: the Admin UI
fills it, and the tray keeps running behind it so closing the window leaves the
proxy serving rather than killing it.

The window owns the main thread because both backends here demand it, so the
tray runs detached alongside. Where a platform refuses to detach a tray, the
window runs on its own and closing it quits, which is worse than having both
but better than not starting.
"""

from contextlib import suppress
from pathlib import Path

import webview

from free_claude_code.cli.desktop import DesktopController, launch_desktop
from free_claude_code.cli.desktop_tray import PystrayDesktopTray
from free_claude_code.config.server_urls import local_admin_url
from free_claude_code.config.settings import get_settings

SHELL_PATH = Path(__file__).resolve().parent / "desktop_shell.html"

WINDOW_TITLE = "Free Claude Code"
WINDOW_SIZE = (1180, 820)
WINDOW_MIN_SIZE = (760, 560)


class ShellApi:
    """The calls the shell page is allowed to make.

    Deliberately not HTTP. The shell exists to be useful when the server is
    down, which is precisely when the Admin API cannot answer, so every method
    here reaches the supervisor in this process instead.
    """

    def __init__(self, controller: DesktopController) -> None:
        self._controller = controller

    def state(self) -> dict[str, str]:
        return {
            "state": str(self._controller.status),
            "admin_url": local_admin_url(get_settings()),
        }

    def start_server(self) -> None:
        self._controller.restart_server()


class WebviewDesktopShell:
    """A desktop UI loop backed by a native window, with a tray behind it."""

    def __init__(self, controller: DesktopController) -> None:
        self._controller = controller
        self._tray = PystrayDesktopTray(controller, open_admin=self.show)
        self._tray_detached = False
        self._quitting = False
        window = webview.create_window(
            WINDOW_TITLE,
            str(SHELL_PATH),
            js_api=ShellApi(controller),
            width=WINDOW_SIZE[0],
            height=WINDOW_SIZE[1],
            min_size=WINDOW_MIN_SIZE,
        )
        if window is None:
            raise RuntimeError("pywebview could not create the desktop window.")
        self._window = window
        self._window.events.closing += self._on_closing

    def run(self) -> None:
        """Run the window on this thread, with the tray detached beside it."""

        self._tray_detached = self._tray.run_detached()
        webview.start()

    def stop(self) -> None:
        self._quitting = True
        if self._tray_detached:
            self._tray.stop()
        # Already gone is the ordinary case: stop() arrives once when the user
        # closes the window and again from the controller's shutdown path.
        with suppress(Exception):
            self._window.destroy()

    def show(self) -> None:
        """Bring the window back, which is what the tray's default action means."""

        self._window.show()

    def _on_closing(self) -> bool:
        """Hide instead of quitting while a tray is there to restore the window.

        Returning False cancels the close. Without a tray there would be no way
        back to a hidden window, so there the close is allowed through and the
        controller shuts the server down after it.
        """

        if self._quitting or not self._tray_detached:
            return True
        self._window.hide()
        return False


def launch() -> None:
    """Launch the windowed desktop shell."""

    launch_desktop(WebviewDesktopShell)

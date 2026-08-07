"""pystray adapter for the Windows tray and macOS menu bar."""

from collections.abc import Callable
from io import BytesIO

from PIL import Image
from pystray import Icon, Menu, MenuItem

from free_claude_code.cli.desktop import DesktopController
from free_claude_code.cli.desktop_assets import app_icon_bytes


class PystrayDesktopTray:
    """Render desktop lifecycle actions through the native status area."""

    def __init__(
        self,
        controller: DesktopController,
        *,
        open_admin: Callable[[], None] | None = None,
    ) -> None:
        self._controller = controller
        # The windowed shell restores its window here instead of launching a
        # browser, so the tray points at whichever surface is actually the app.
        self._open_admin_action = open_admin or controller.open_admin
        self._icon = Icon(
            "free-claude-code",
            _create_icon(),
            "Free Claude Code",
            Menu(
                MenuItem("Open Admin", self._open_admin, default=True),
                MenuItem("Check Server Status", self._check_status),
                MenuItem("Restart Server", self._restart_server),
                Menu.SEPARATOR,
                MenuItem("Quit", self._quit),
            ),
        )

    def run(self) -> None:
        self._icon.run()

    def run_detached(self) -> bool:
        """Run the tray off the main thread, reporting whether that was allowed.

        The windowed shell needs the main thread for its own loop, so the tray
        has to detach. Backends that cannot report it rather than raising into a
        startup that would otherwise have worked without a tray at all.
        """

        try:
            self._icon.run_detached()
        except NotImplementedError:
            return False
        return True

    def stop(self) -> None:
        self._icon.stop()

    def _open_admin(self, _icon: Icon, _item: MenuItem) -> None:
        self._open_admin_action()

    def _check_status(self, _icon: Icon, _item: MenuItem) -> None:
        self._icon.notify(
            f"Server is {self._controller.status}.",
            "Free Claude Code",
        )

    def _restart_server(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.restart_server()

    def _quit(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.quit()


def _create_icon() -> Image.Image:
    """Load the same branded artwork used by native desktop launchers."""

    with Image.open(BytesIO(app_icon_bytes(".png"))) as image:
        return image.convert("RGBA")

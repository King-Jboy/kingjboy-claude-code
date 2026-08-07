"""Windowed desktop shell: what it shows, and what closing it means."""

from unittest.mock import MagicMock

import pytest

# The window backends only install on the desktop platforms the app supports.
pytest.importorskip("webview")
pytest.importorskip("pystray")

from free_claude_code.cli import desktop_window

ADMIN_URL = "http://127.0.0.1:8082/admin"


def _shell(monkeypatch, *, detached: bool = True):
    """Build the shell with both native surfaces replaced.

    Nothing here may open a real window: these assertions are about the wiring,
    and a test that pops up a window cannot run on a build machine.
    """

    window = MagicMock()
    tray = MagicMock()
    tray.run_detached.return_value = detached
    controller = MagicMock()

    monkeypatch.setattr(
        desktop_window.webview, "create_window", MagicMock(return_value=window)
    )
    monkeypatch.setattr(
        desktop_window, "PystrayDesktopTray", MagicMock(return_value=tray)
    )
    monkeypatch.setattr(desktop_window, "get_settings", MagicMock())
    monkeypatch.setattr(
        desktop_window, "local_admin_url", MagicMock(return_value=ADMIN_URL)
    )

    return desktop_window.WebviewDesktopShell(controller), window, tray, controller


def test_shell_api_reports_status_without_asking_the_server(monkeypatch):
    # The shell has to describe a server that is not answering, so its state
    # comes from the supervisor in this process rather than the Admin API.
    monkeypatch.setattr(desktop_window, "get_settings", MagicMock())
    monkeypatch.setattr(
        desktop_window, "local_admin_url", MagicMock(return_value=ADMIN_URL)
    )
    controller = MagicMock()
    controller.status = "Stopped"

    api = desktop_window.ShellApi(controller)

    assert api.state() == {"state": "Stopped", "admin_url": ADMIN_URL}


def test_shell_api_start_goes_through_the_controller(monkeypatch):
    monkeypatch.setattr(desktop_window, "get_settings", MagicMock())
    controller = MagicMock()

    desktop_window.ShellApi(controller).start_server()

    controller.restart_server.assert_called_once_with()


def test_running_detaches_the_tray_before_the_window_takes_the_thread(monkeypatch):
    shell, _window, tray, _controller = _shell(monkeypatch)
    start = MagicMock()
    monkeypatch.setattr(desktop_window.webview, "start", start)

    shell.run()

    tray.run_detached.assert_called_once_with()
    start.assert_called_once_with()


def test_closing_hides_the_window_while_a_tray_can_bring_it_back(monkeypatch):
    # Closing the window must not stop the proxy: the tray is still there to
    # restore it, and the server is what the user actually left running.
    shell, window, _tray, _controller = _shell(monkeypatch, detached=True)
    monkeypatch.setattr(desktop_window.webview, "start", MagicMock())
    shell.run()

    assert shell._on_closing() is False
    window.hide.assert_called_once_with()


def test_closing_quits_when_no_tray_survived_to_reopen_the_window(monkeypatch):
    # Without a tray a hidden window is unreachable, so the close goes through
    # and the controller shuts the server down behind it.
    shell, window, _tray, _controller = _shell(monkeypatch, detached=False)
    monkeypatch.setattr(desktop_window.webview, "start", MagicMock())
    shell.run()

    assert shell._on_closing() is True
    window.hide.assert_not_called()


def test_stopping_tears_down_both_surfaces(monkeypatch):
    shell, window, tray, _controller = _shell(monkeypatch, detached=True)
    monkeypatch.setattr(desktop_window.webview, "start", MagicMock())
    shell.run()

    shell.stop()

    tray.stop.assert_called_once_with()
    window.destroy.assert_called_once_with()
    # A second teardown arrives from the controller's shutdown path.
    assert shell._on_closing() is True


def test_a_window_that_cannot_be_created_fails_loudly(monkeypatch):
    monkeypatch.setattr(
        desktop_window.webview, "create_window", MagicMock(return_value=None)
    )
    monkeypatch.setattr(
        desktop_window, "PystrayDesktopTray", MagicMock(return_value=MagicMock())
    )

    with pytest.raises(RuntimeError, match="could not create"):
        desktop_window.WebviewDesktopShell(MagicMock())


def test_the_tray_opens_whatever_surface_it_was_given(monkeypatch):
    from free_claude_code.cli import desktop_tray

    monkeypatch.setattr(desktop_tray, "Icon", MagicMock())
    monkeypatch.setattr(desktop_tray, "_create_icon", MagicMock())
    controller = MagicMock()
    opened: list[str] = []

    tray = desktop_tray.PystrayDesktopTray(
        controller, open_admin=lambda: opened.append("window")
    )
    tray._open_admin(MagicMock(), MagicMock())

    assert opened == ["window"]
    controller.open_admin.assert_not_called()


def test_a_tray_that_cannot_detach_reports_it_rather_than_raising(monkeypatch):
    from free_claude_code.cli import desktop_tray

    icon = MagicMock()
    icon.run_detached.side_effect = NotImplementedError
    monkeypatch.setattr(desktop_tray, "Icon", MagicMock(return_value=icon))
    monkeypatch.setattr(desktop_tray, "_create_icon", MagicMock())

    tray = desktop_tray.PystrayDesktopTray(MagicMock())

    assert tray.run_detached() is False

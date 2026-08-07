"""Desktop shell lifecycle and singleton contracts."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from free_claude_code.cli.commands import ServerStatus, ServerSupervisor
from free_claude_code.cli.desktop import DesktopController
from free_claude_code.config.settings import Settings
from free_claude_code.core.interprocess_lock import InterprocessFileLock


def _settings() -> Settings:
    return Settings.model_construct(host="0.0.0.0", port=8082)


def test_desktop_instance_lock_is_exclusive_and_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "desktop.lock"
    first = InterprocessFileLock(lock_path)
    second = InterprocessFileLock(lock_path)

    assert first.acquire() is True
    assert first.acquire() is True
    assert second.acquire() is False

    first.release()
    first.release()
    assert second.acquire() is True
    second.release()


def test_supervisor_accepts_restart_during_scheduled_startup() -> None:
    supervisor = ServerSupervisor(console_logging=False)
    settings = _settings()

    with (
        patch(
            "free_claude_code.cli.commands.load_server_settings",
            return_value=settings,
        ),
        patch.object(supervisor, "_run_once", return_value=False) as run_once,
        patch("free_claude_code.cli.commands.kill_all_best_effort"),
    ):
        assert supervisor.schedule_run() is True
        assert supervisor.status is ServerStatus.STARTING
        assert supervisor.request_restart() is True
        supervisor.run(open_admin_browser=False)

    run_once.assert_called_once_with(
        settings,
        open_admin_browser=False,
        restart_generation=1,
    )
    assert supervisor.status is ServerStatus.STOPPED


def test_desktop_controller_owns_server_thread_and_graceful_quit() -> None:
    opened = threading.Event()

    class FakeSupervisor:
        def __init__(self) -> None:
            self.status = ServerStatus.STARTING
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.run_arguments: list[bool | None] = []
            self.schedule_count = 0
            self.restart_count = 0
            self.stop_count = 0

        def schedule_run(self) -> bool:
            self.schedule_count += 1
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            self.run_arguments.append(open_admin_browser)
            self.status = ServerStatus.RUNNING
            self.started.set()
            assert self.stopped.wait(2)
            self.status = ServerStatus.STOPPED

        def request_restart(self) -> bool:
            self.restart_count += 1
            return True

        def request_stop(self) -> None:
            self.stop_count += 1
            self.status = ServerStatus.STOPPING
            self.stopped.set()

    class FakeTray:
        def __init__(self, controller: DesktopController) -> None:
            self.controller = controller
            self.run_thread_id: int | None = None
            self.stop_count = 0

        def run(self) -> None:
            self.run_thread_id = threading.get_ident()
            assert supervisor.started.wait(2)
            self.controller.open_admin()
            self.controller.restart_server()
            self.controller.quit()

        def stop(self) -> None:
            self.stop_count += 1

    supervisor = FakeSupervisor()
    tray: FakeTray | None = None

    def make_tray(controller: DesktopController) -> FakeTray:
        nonlocal tray
        tray = FakeTray(controller)
        return tray

    main_thread_id = threading.get_ident()
    controller = DesktopController(lambda: supervisor, make_tray, opened.set)
    controller.run()

    assert tray is not None
    assert tray.run_thread_id == main_thread_id
    assert supervisor.run_arguments == [False]
    assert supervisor.schedule_count == 1
    assert supervisor.restart_count == 1
    assert supervisor.stop_count >= 1
    assert tray.stop_count >= 1
    assert opened.is_set()


def test_restart_during_server_startup_is_accepted_without_waiting() -> None:
    class StartupSupervisor:
        def __init__(self) -> None:
            self.status = ServerStatus.STARTING
            self.run_called = threading.Event()
            self.allow_run = threading.Event()
            self.worker_started = threading.Event()
            self.release_worker = threading.Event()
            self.run_scheduled = False
            self.restart_count = 0
            self.accepted_restart_count = 0

        def schedule_run(self) -> bool:
            self.run_scheduled = True
            return True

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            assert open_admin_browser is False
            self.run_called.set()
            assert self.allow_run.wait(2)
            self.run_scheduled = False
            self.worker_started.set()
            assert self.release_worker.wait(2)
            self.status = ServerStatus.STOPPED

        def request_restart(self) -> bool:
            self.restart_count += 1
            if self.run_scheduled:
                self.accepted_restart_count += 1
                return True
            return False

        def request_stop(self) -> None:
            self.release_worker.set()

    class WaitingTray:
        def __init__(self, _controller: DesktopController) -> None:
            self.started = threading.Event()
            self.stopped = threading.Event()

        def run(self) -> None:
            self.started.set()
            assert self.stopped.wait(2)

        def stop(self) -> None:
            self.stopped.set()

    supervisor = StartupSupervisor()
    tray: WaitingTray | None = None

    def make_tray(controller: DesktopController) -> WaitingTray:
        nonlocal tray
        tray = WaitingTray(controller)
        return tray

    controller = DesktopController(lambda: supervisor, make_tray, MagicMock())
    controller_thread = threading.Thread(target=controller.run)
    controller_thread.start()
    assert tray is not None
    assert tray.started.wait(2)
    assert supervisor.run_called.wait(2)

    restart_thread = threading.Thread(target=controller.restart_server)
    restart_thread.start()
    restart_thread.join(0.5)
    restart_blocked = restart_thread.is_alive()

    supervisor.allow_run.set()
    assert supervisor.worker_started.wait(2)
    controller.quit()
    supervisor.release_worker.set()
    restart_thread.join(2)
    controller_thread.join(2)

    assert restart_blocked is False
    assert supervisor.restart_count == 1
    assert supervisor.accepted_restart_count == 1
    assert not restart_thread.is_alive()
    assert not controller_thread.is_alive()


def test_second_desktop_launch_opens_existing_admin_without_new_server() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = False

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "InterprocessFileLock", return_value=instance_lock),
        patch.object(desktop, "open_admin_when_ready", return_value=True) as open_admin,
        patch.object(desktop, "ServerSupervisor") as supervisor,
    ):
        desktop.launch_desktop(MagicMock())

    open_admin.assert_called_once_with(settings)
    supervisor.assert_not_called()
    instance_lock.release.assert_not_called()


def test_desktop_attaches_to_terminal_server_instead_of_binding_twice() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = True

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "InterprocessFileLock", return_value=instance_lock),
        patch.object(desktop, "preflight_proxy", return_value=None),
        patch.object(desktop, "open_admin_when_ready", return_value=True) as open_admin,
        patch.object(desktop, "ServerSupervisor") as supervisor,
    ):
        desktop.launch_desktop(MagicMock())

    open_admin.assert_called_once_with(settings)
    supervisor.assert_not_called()
    instance_lock.release.assert_called_once_with()


def test_fresh_desktop_launch_disables_console_and_automatic_browser() -> None:
    from free_claude_code.cli import desktop

    settings = _settings()
    instance_lock = MagicMock()
    instance_lock.acquire.return_value = True
    supervisor = MagicMock()
    controller = MagicMock()

    with (
        patch.object(desktop, "load_server_settings", return_value=settings),
        patch.object(desktop, "InterprocessFileLock", return_value=instance_lock),
        patch.object(desktop, "preflight_proxy", return_value="connection refused"),
        patch.object(desktop, "ServerSupervisor", return_value=supervisor) as owner,
        patch.object(desktop, "DesktopController", return_value=controller) as shell,
    ):
        tray_factory = MagicMock()
        desktop.launch_desktop(tray_factory)

        # The controller owns supervisor construction now, so the launch hands
        # it a factory rather than an instance. That is what lets it build a
        # replacement after Admin stops the server, and it means nothing is
        # constructed until the controller asks. The factory has to be called
        # inside the patch, since it resolves ServerSupervisor when invoked.
        supervisor_factory, passed_tray_factory = shell.call_args.args[:2]
        owner.assert_not_called()
        assert supervisor_factory() is supervisor
        owner.assert_called_once_with(console_logging=False)

    assert passed_tray_factory is tray_factory
    controller.run.assert_called_once_with()
    instance_lock.release.assert_called_once_with()


def test_restart_after_an_admin_stop_builds_a_replacement_supervisor() -> None:
    """A stopped supervisor stays stopped, so the tray needs a fresh one.

    request_stop latches _stop_requested permanently: schedule_run and
    request_restart both refuse from then on. Without a replacement, Admin's
    Stop would leave a live tray sitting over a server that can never start
    again, and the tray's Restart would silently do nothing.
    """

    class OneShotSupervisor:
        def __init__(self) -> None:
            self.status = ServerStatus.STOPPED
            self.stopped = False
            self.started = threading.Event()
            self.release = threading.Event()
            self.run_count = 0

        def schedule_run(self) -> bool:
            return not self.stopped

        def run(self, *, open_admin_browser: bool | None = None) -> None:
            self.run_count += 1
            self.started.set()
            assert self.release.wait(2)

        def request_restart(self) -> bool:
            return not self.stopped

        def request_stop(self) -> None:
            self.stopped = True
            self.release.set()

    built: list[OneShotSupervisor] = []

    def build_supervisor() -> OneShotSupervisor:
        supervisor = OneShotSupervisor()
        built.append(supervisor)
        return supervisor

    def server_thread_is_alive() -> bool:
        return any(
            thread.name == "fcc-desktop-server" and thread.is_alive()
            for thread in threading.enumerate()
        )

    controller = DesktopController(build_supervisor, MagicMock(), MagicMock())
    controller.restart_server()
    assert built[0].started.wait(2)

    # Admin stops the server the way the endpoint does, then the worker has to
    # actually finish: restart_server reuses a live thread, so asserting before
    # it exits would test the wrong branch.
    built[0].request_stop()
    for _ in range(200):
        if not server_thread_is_alive():
            break
        threading.Event().wait(0.02)
    assert not server_thread_is_alive()

    controller.restart_server()

    assert len(built) == 2, "a stopped supervisor must be replaced, not reused"
    assert built[1].started.wait(2)
    assert built[1].run_count == 1
    built[1].request_stop()

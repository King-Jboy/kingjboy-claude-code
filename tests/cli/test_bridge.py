"""`fcc-bridge`: the protocol it speaks, and what it refuses to run."""

import io
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from free_claude_code.cli import bridge
from free_claude_code.config.settings import Settings


def _settings(**overrides: Any) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _framed(message: dict[str, object]) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    return struct.pack("<I", len(payload)) + payload


# ---------- the gate ----------


def test_commands_are_refused_until_the_setting_is_turned_on(tmp_path: Path) -> None:
    # Registering the host must not be sufficient by itself. This is the
    # difference between "the browser can reach a shell" and "it may use one".
    response = bridge.handle(
        {"type": "run", "command": "echo hello"},
        _settings(BROWSER_SHELL_ENABLED=False, BROWSER_SHELL_ROOT=str(tmp_path)),
    )

    assert response["ok"] is False
    assert "BROWSER_SHELL_ENABLED" in str(response["error"])


def test_ping_answers_without_the_setting_so_the_panel_can_explain_itself(
    tmp_path: Path,
) -> None:
    response = bridge.handle(
        {"type": "ping"},
        _settings(BROWSER_SHELL_ENABLED=False, BROWSER_SHELL_ROOT=str(tmp_path)),
    )

    assert response["ok"] is True
    assert response["enabled"] is False
    assert response["root"] == str(tmp_path.resolve())


def test_an_unknown_request_type_is_named_in_the_refusal() -> None:
    response = bridge.handle({"type": "sudo"}, _settings())

    assert response["ok"] is False
    assert "sudo" in str(response["error"])


# ---------- directory confinement ----------


def test_a_relative_cwd_resolves_inside_the_root(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()

    assert bridge.resolve_cwd("project", tmp_path) == (tmp_path / "project").resolve()


def test_a_traversal_out_of_the_root_is_refused(tmp_path: Path) -> None:
    # resolve() has to collapse ".." before the containment check, or a path
    # that merely starts with the root string would be accepted.
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside").mkdir()

    with pytest.raises(bridge.BridgeError, match="outside BROWSER_SHELL_ROOT"):
        bridge.resolve_cwd("../outside", root)


def test_an_absolute_cwd_elsewhere_on_disk_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(bridge.BridgeError, match="outside BROWSER_SHELL_ROOT"):
        bridge.resolve_cwd(str(elsewhere), root)


def test_a_cwd_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    with pytest.raises(bridge.BridgeError, match="not a directory"):
        bridge.resolve_cwd("file.txt", tmp_path)


def test_no_cwd_means_the_root_itself(tmp_path: Path) -> None:
    assert bridge.resolve_cwd(None, tmp_path) == tmp_path


def test_a_blank_root_setting_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert bridge.shell_root(_settings(BROWSER_SHELL_ROOT="")) == tmp_path


# ---------- execution ----------


def test_an_approved_command_runs_and_reports_its_exit_code(tmp_path: Path) -> None:
    response = bridge.handle(
        {"type": "run", "command": "echo bridge-works"},
        _settings(BROWSER_SHELL_ENABLED=True, BROWSER_SHELL_ROOT=str(tmp_path)),
    )

    assert response["ok"] is True
    assert response["exit_code"] == 0
    assert "bridge-works" in str(response["stdout"])


def test_a_failing_command_reports_the_failure_rather_than_erroring(
    tmp_path: Path,
) -> None:
    # A non-zero exit is a result the model should see and reason about, not a
    # transport failure.
    response = bridge.handle(
        {"type": "run", "command": "exit 3"},
        _settings(BROWSER_SHELL_ENABLED=True, BROWSER_SHELL_ROOT=str(tmp_path)),
    )

    assert response["ok"] is True
    assert response["exit_code"] == 3


def test_a_blank_command_is_refused(tmp_path: Path) -> None:
    response = bridge.handle(
        {"type": "run", "command": "   "},
        _settings(BROWSER_SHELL_ENABLED=True, BROWSER_SHELL_ROOT=str(tmp_path)),
    )

    assert response["ok"] is False


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, 120), (0, 120), (-5, 120), (30, 30), (10_000, 600)],
)
def test_timeouts_are_clamped_to_a_sane_range(requested: object, expected: int) -> None:
    assert bridge._clamp_timeout(requested) == expected


def test_output_larger_than_the_cap_is_truncated_and_says_so() -> None:
    # Chrome refuses host messages over 1MB, and the output becomes prompt text
    # besides, so an unbounded `cat` would break both.
    text, was_cut = bridge._truncate("x" * (bridge.MAX_OUTPUT_CHARS + 500))

    assert was_cut is True
    assert text.endswith("[truncated]")
    assert len(text) < bridge.MAX_OUTPUT_CHARS + 100


def test_a_command_that_runs_is_written_to_the_audit_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    bridge.handle(
        {"type": "run", "command": "echo audited"},
        _settings(BROWSER_SHELL_ENABLED=True, BROWSER_SHELL_ROOT=str(tmp_path)),
    )

    log = (tmp_path / ".fcc" / "logs" / "bridge.log").read_text(encoding="utf-8")
    assert "echo audited" in log
    assert "exit=0" in log


def test_a_refused_command_is_also_audited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "root"
    root.mkdir()

    bridge.handle(
        {"type": "run", "command": "echo nope", "cwd": str(tmp_path)},
        _settings(BROWSER_SHELL_ENABLED=True, BROWSER_SHELL_ROOT=str(root)),
    )

    log = (tmp_path / ".fcc" / "logs" / "bridge.log").read_text(encoding="utf-8")
    assert "refused" in log


def test_the_windows_shell_is_powershell_rather_than_cmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # shell=True would pick cmd.exe, where half of what a PowerShell user would
    # type by hand fails.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        bridge.shutil, "which", lambda name: r"C:\pwsh.exe" if name == "pwsh" else None
    )

    assert bridge.default_shell()[0] == r"C:\pwsh.exe"
    assert "-NoProfile" in bridge.default_shell()


# ---------- wire protocol ----------


def test_a_message_round_trips_through_the_frame_codec() -> None:
    buffer = io.BytesIO()
    bridge.write_message(buffer, {"ok": True, "value": 7})
    buffer.seek(0)

    assert bridge.read_message(buffer) == {"ok": True, "value": 7}


def test_a_closed_pipe_reads_as_end_of_stream() -> None:
    # Chrome closes stdin when the extension goes away; that is a clean exit,
    # not an error to report into a pipe nobody is reading.
    assert bridge.read_message(io.BytesIO(b"")) is None


def test_a_truncated_payload_reads_as_end_of_stream() -> None:
    assert bridge.read_message(io.BytesIO(struct.pack("<I", 100) + b"{}")) is None


def test_an_oversized_frame_is_refused_and_taken_off_the_pipe() -> None:
    length = bridge.MAX_REQUEST_BYTES + 1
    frame = struct.pack("<I", length) + b"x" * length
    stream = io.BytesIO(frame)

    with pytest.raises(bridge.BridgeError, match="exceeds"):
        bridge.read_message(stream)

    # Refusing the message is only half of it: the bytes have to leave the pipe
    # too, or the next read starts mid-payload.
    assert stream.tell() == len(frame)


def test_a_refused_frame_does_not_desynchronize_the_ones_after_it() -> None:
    # The payload of a refused frame used to stay in the pipe, so the next read
    # took four payload bytes for a length prefix. Every later frame was
    # garbage and the host answered nonsense until Chrome restarted it.
    length = bridge.MAX_REQUEST_BYTES + 1
    oversized = struct.pack("<I", length) + b"x" * length
    stream = io.BytesIO(oversized + _framed({"type": "ping"}))

    with pytest.raises(bridge.BridgeError, match="exceeds"):
        bridge.read_message(stream)

    assert bridge.read_message(stream) == {"type": "ping"}


def test_an_oversized_frame_cut_short_reads_as_end_of_stream() -> None:
    # A frame that never arrives in full is a closed pipe, the same as any
    # other truncated payload -- there is no frame boundary left to refuse to.
    header = struct.pack("<I", bridge.MAX_REQUEST_BYTES + 1)

    assert bridge.read_message(io.BytesIO(header)) is None


def test_the_host_keeps_serving_after_it_refuses_an_oversized_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    length = bridge.MAX_REQUEST_BYTES + 1
    stdin = io.BytesIO(
        struct.pack("<I", length) + b"x" * length + _framed({"type": "ping"})
    )
    channel = io.BytesIO()
    monkeypatch.setattr(bridge.sys, "stdin", type("S", (), {"buffer": stdin})())
    monkeypatch.setattr(
        bridge.sys, "stdout", type("O", (), {"buffer": channel})(), raising=False
    )

    assert bridge.run([]) == 0

    channel.seek(0)
    refusal = bridge.read_message(channel)
    assert refusal is not None
    assert refusal["ok"] is False
    assert "exceeds" in str(refusal["error"])

    answered = bridge.read_message(channel)
    assert answered is not None
    assert answered["type"] == "pong"
    assert bridge.read_message(channel) is None


def test_malformed_json_is_answered_rather_than_crashing_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A crash here surfaces to the user as a generic "native host has exited",
    # which says nothing about what went wrong.
    payload = b"{not json"
    stdin = io.BytesIO(struct.pack("<I", len(payload)) + payload)
    channel = io.BytesIO()
    monkeypatch.setattr(bridge.sys, "stdin", type("S", (), {"buffer": stdin})())
    monkeypatch.setattr(
        bridge.sys, "stdout", type("O", (), {"buffer": channel})(), raising=False
    )

    assert bridge.run([]) == 0
    channel.seek(0)
    answer = bridge.read_message(channel)
    assert answer is not None
    assert answer["ok"] is False


def test_help_does_not_start_the_protocol_loop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert bridge.run(["--help"]) == 0
    assert "fcc-extension install" in capsys.readouterr().out

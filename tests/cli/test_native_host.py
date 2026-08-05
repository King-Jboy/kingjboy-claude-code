"""Native messaging host registration: the manifest, and where it is written."""

import json
import sys
from pathlib import Path

import pytest

from free_claude_code.cli import native_host
from free_claude_code.cli.bridge import HOST_NAME

VALID_ID = "abcdefghijklmnopabcdefghijklmnop"


def test_the_manifest_admits_exactly_one_extension() -> None:
    # allowed_origins is the boundary the whole bridge rests on. A second entry
    # here would hand the same shell to another extension.
    manifest = native_host.host_manifest(VALID_ID, "/usr/local/bin/fcc-bridge")

    assert manifest["allowed_origins"] == [f"chrome-extension://{VALID_ID}/"]
    assert manifest["name"] == HOST_NAME
    assert manifest["type"] == "stdio"
    assert manifest["path"] == "/usr/local/bin/fcc-bridge"


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "too-short",
        "ABCDEFGHIJKLMNOPABCDEFGHIJKLMNOP",  # IDs are lowercase
        "abcdefghijklmnopabcdefghijklmnoz",  # z is outside a-p
        "abcdefghijklmnopabcdefghijklmno",  # 31 characters
        "*",
    ],
)
def test_anything_that_is_not_an_extension_id_is_rejected(candidate: str) -> None:
    with pytest.raises(
        native_host.RegistrationError, match="not a Chrome extension ID"
    ):
        native_host.validate_extension_id(candidate)


def test_a_valid_id_is_returned_stripped() -> None:
    assert native_host.validate_extension_id(f"  {VALID_ID}  ") == VALID_ID


def test_a_missing_bridge_executable_is_explained_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_host.shutil, "which", lambda _name: None)

    with pytest.raises(native_host.RegistrationError, match="Reinstall FCC"):
        native_host.bridge_executable()


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("win32", r"Software\Google\Chrome\NativeMessagingHosts"),
        ("darwin", "Library/Application Support/Google/Chrome"),
        ("linux", ".config/google-chrome"),
    ],
)
def test_each_platform_registers_where_its_browsers_look(
    platform: str, expected: str
) -> None:
    targets = native_host.browser_targets(platform)

    assert targets[0].name == "Chrome"
    assert targets[0].location == expected
    # Chromium and Edge are Chromium-family too; the request named them.
    assert {target.name for target in targets} == {"Chrome", "Chromium", "Edge"}


def test_posix_install_writes_only_into_profiles_that_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Creating the profile directory for a browser that is not installed would
    # litter the home directory with folders the user never asked for.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".config" / "google-chrome").mkdir(parents=True)

    registered = native_host._install_posix(
        native_host.host_manifest(VALID_ID, "/bin/fcc-bridge"), "linux"
    )

    assert registered == ("Chrome",)
    written = (
        tmp_path
        / ".config"
        / "google-chrome"
        / "NativeMessagingHosts"
        / f"{HOST_NAME}.json"
    )
    assert json.loads(written.read_text(encoding="utf-8"))["allowed_origins"] == [
        f"chrome-extension://{VALID_ID}/"
    ]
    assert not (tmp_path / ".config" / "chromium").exists()


def test_posix_uninstall_removes_the_manifest_it_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    (tmp_path / ".config" / "google-chrome").mkdir(parents=True)
    native_host._install_posix(
        native_host.host_manifest(VALID_ID, "/bin/fcc-bridge"), "linux"
    )

    removed = native_host.uninstall()

    assert removed == ("Chrome",)
    assert not native_host.stored_manifest_path().exists()
    assert not (
        tmp_path
        / ".config"
        / "google-chrome"
        / "NativeMessagingHosts"
        / f"{HOST_NAME}.json"
    ).exists()


def test_uninstalling_when_nothing_is_registered_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")

    assert native_host.uninstall() == ()


def test_install_refuses_a_bad_id_before_touching_the_filesystem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(native_host.RegistrationError):
        native_host.install("not-an-id")

    assert not (tmp_path / ".fcc").exists()

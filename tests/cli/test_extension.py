"""`fcc-extension`: what it reports, and that the bundled assets stay coherent."""

import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from free_claude_code.cli import extension
from free_claude_code.config.settings import Settings


def _settings(**overrides: Any) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _manifest() -> dict[str, Any]:
    path = extension.extension_dir() / extension.MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    root = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return tomllib.loads(root.read_text(encoding="utf-8"))["project"]["version"]


# ---------- bundled assets ----------


def test_every_file_the_manifest_points_at_is_packaged() -> None:
    # Chrome's failure for a missing entry is "Could not load manifest", which
    # names nothing. Catching it here beats debugging it in chrome://extensions.
    manifest = _manifest()
    referenced = {
        manifest["background"]["service_worker"],
        manifest["side_panel"]["default_path"],
        *(js for script in manifest["content_scripts"] for js in script["js"]),
    }

    absent = [
        name for name in referenced if not (extension.extension_dir() / name).is_file()
    ]

    assert absent == []


def test_the_required_asset_list_matches_what_is_on_disk() -> None:
    # REQUIRED_ASSETS drives the packaging check, so a new asset that nobody
    # added to it would ship broken and the check would still pass.
    on_disk = {
        path.name
        for path in extension.extension_dir().iterdir()
        if path.is_file() and not path.name.startswith(".")
    }

    assert on_disk == set(extension.REQUIRED_ASSETS)


def test_the_manifest_version_tracks_the_package_version() -> None:
    # Chrome requires a literal version string, so it cannot be computed at
    # load time. This assertion is the only thing keeping the two in step.
    assert _manifest()["version"] == _pyproject_version()


def test_the_manifest_reaches_the_proxy_on_any_local_port() -> None:
    # Match patterns carry no port, so the localhost pair covers every PORT
    # setting; narrowing them to a literal port would break every non-default
    # install. `<all_urls>` is separate and deliberate: scripting and
    # captureVisibleTab need host access to the target page, which the proxy
    # patterns do not give. It adds no install warning the content_scripts
    # declaration does not already trigger.
    assert set(_manifest()["host_permissions"]) == {
        "http://127.0.0.1/*",
        "http://localhost/*",
        "<all_urls>",
    }


def test_the_console_probe_runs_early_in_the_page_world() -> None:
    # It patches the page's own console object, so the isolated world would
    # see nothing, and document_idle would miss everything logged during load.
    (probe,) = _manifest()["content_scripts"]

    assert probe["world"] == "MAIN"
    assert probe["run_at"] == "document_start"


def test_the_bundled_javascript_parses(tmp_path: Path) -> None:
    # Nothing else in CI executes this code: a syntax error would ship and only
    # surface as a blank side panel. Skipped where node is unavailable, so it
    # is a local guard rather than a merge gate.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    for source in sorted(extension.extension_dir().glob("*.js")):
        # node treats a bare .js file as CommonJS, and these use ESM imports.
        module = tmp_path / f"{source.stem}.mjs"
        module.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        checked = subprocess.run(
            [node, "--check", str(module)],
            capture_output=True,
            check=False,
            text=True,
        )
        assert checked.returncode == 0, f"{source.name}: {checked.stderr}"


def test_missing_assets_are_reported_by_name(tmp_path: Path) -> None:
    (tmp_path / extension.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")

    assert extension.missing_assets(tmp_path) == tuple(
        name
        for name in extension.REQUIRED_ASSETS
        if name != extension.MANIFEST_FILENAME
    )


def test_an_incomplete_install_fails_rather_than_printing_a_dead_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(extension, "missing_assets", lambda: ("sidepanel.js",))

    assert extension.run([]) == 1
    assert "sidepanel.js" in capsys.readouterr().err


# ---------- reporting ----------


def test_the_report_names_the_directory_and_the_proxy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        extension, "Settings", lambda: _settings(port=9191, ANTHROPIC_AUTH_TOKEN="")
    )

    assert extension.run([]) == 0
    out = capsys.readouterr().out
    assert str(extension.extension_dir()) in out
    assert "http://127.0.0.1:9191" in out
    assert "chrome://extensions" in out


def test_a_configured_token_is_masked_until_it_is_asked_for(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # This prints in terminals that get screen-shared and recorded.
    monkeypatch.setattr(
        extension, "Settings", lambda: _settings(ANTHROPIC_AUTH_TOKEN="secret-token")
    )

    extension.run([])
    masked = capsys.readouterr().out
    extension.run(["--show-token"])
    revealed = capsys.readouterr().out

    assert "secret-token" not in masked
    assert "--show-token" in masked
    assert "secret-token" in revealed


def test_no_token_is_reported_as_a_working_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A blank ANTHROPIC_AUTH_TOKEN disables the check; the panel must be told
    # to leave its own field blank rather than invent a value.
    monkeypatch.setattr(
        extension, "Settings", lambda: _settings(ANTHROPIC_AUTH_TOKEN="")
    )

    extension.run([])

    assert "none" in capsys.readouterr().out


def test_path_prints_only_the_directory(capsys: pytest.CaptureFixture[str]) -> None:
    assert extension.run(["--path"]) == 0
    assert capsys.readouterr().out.strip() == str(extension.extension_dir())


def test_json_output_withholds_the_token_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        extension,
        "Settings",
        lambda: _settings(port=9191, ANTHROPIC_AUTH_TOKEN="secret-token"),
    )

    extension.run(["--json"])
    withheld = json.loads(capsys.readouterr().out)
    extension.run(["--json", "--show-token"])
    revealed = json.loads(capsys.readouterr().out)

    assert withheld["auth_token"] == ""
    # auth_required stays true so a consumer can tell "withheld" from "unset".
    assert withheld["auth_required"] is True
    assert withheld["proxy_url"] == "http://127.0.0.1:9191"
    assert revealed["auth_token"] == "secret-token"


def test_unloadable_config_reports_the_directory_and_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The URL comes from config, so bad config means we cannot answer in full --
    # but the directory is static, and it is half of what the user came for.
    monkeypatch.setattr(
        extension, "Settings", lambda: _settings(NVIDIA_NIM_API_KEYS="a, b")
    )

    assert extension.run([]) == 1
    err = capsys.readouterr().err
    assert str(extension.extension_dir()) in err
    assert "fcc-doctor" in err


def test_help_exits_zero_without_reading_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        extension, "Settings", lambda: pytest.fail("--help must not load config")
    )

    assert extension.run(["--help"]) == 0
    assert "--show-token" in capsys.readouterr().out


def _sidepanel_css() -> str:
    return (extension.extension_dir() / "sidepanel.css").read_text(encoding="utf-8")


def _css_custom_property_blocks(css: str) -> list[str]:
    """Return the body of each ``:root`` block, light first then dark."""
    blocks: list[str] = []
    for match in re.finditer(r":root\s*\{", css):
        depth, index = 1, match.end()
        while depth and index < len(css):
            depth += (css[index] == "{") - (css[index] == "}")
            index += 1
        blocks.append(css[match.end() : index - 1])
    return blocks


def test_every_css_variable_the_panel_uses_is_defined() -> None:
    # An undefined custom property fails silently: the declaration is dropped
    # and the element keeps whatever it inherited. That is how --warn-soft went
    # unnoticed while the approval card - the one place a shell command gets
    # authorised - rendered with no background and read as ordinary text.
    # Checked against the base :root alone, not every block pooled together: a
    # token defined only under the dark media query is still undefined in light
    # mode, so pooling them would report a half-broken palette as healthy.
    css = _sidepanel_css()
    used = set(re.findall(r"var\((--[a-z-]+)\)", css))
    base = set(re.findall(r"(--[a-z-]+)\s*:", _css_custom_property_blocks(css)[0]))

    assert not used - base, f"used but never defined: {sorted(used - base)}"


def test_the_dark_palette_only_overrides_tokens_light_already_defines() -> None:
    # A token defined only under the dark media query is undefined in light
    # mode, which is the same silent-drop failure in one theme only.
    light, dark = _css_custom_property_blocks(_sidepanel_css())[:2]
    light_names = set(re.findall(r"(--[a-z-]+)\s*:", light))
    dark_names = set(re.findall(r"(--[a-z-]+)\s*:", dark))

    assert not dark_names - light_names, (
        f"dark-only tokens are undefined in light mode: {sorted(dark_names - light_names)}"
    )


def test_the_approval_card_is_visually_distinct_from_an_ordinary_turn() -> None:
    # The security property is visual: approving a shell command must never
    # look like reading a reply.
    css = _sidepanel_css()
    approval = re.search(r"\n\.approval\s*\{(.*?)\}", css, re.S)

    assert approval is not None, ".approval must be styled"
    body = approval.group(1)
    assert "var(--warn" in body, "the approval card must carry the warning colour"
    assert "box-shadow" in body, "the approval card must be raised off the transcript"


def test_the_side_panel_defaults_to_the_port_the_server_listens_on() -> None:
    # These two numbers are declared in different languages in different files,
    # so nothing but a test keeps them equal. When they drifted, a fresh install
    # reported "Could not reach http://127.0.0.1:8081. Is fcc-server running?"
    # while fcc-server was running, and the message sent people to look at the
    # server rather than at the port the panel had asked for.
    default_port = Settings.model_fields["port"].default
    expected = f"http://127.0.0.1:{default_port}"
    root = extension.extension_dir()

    script = (root / "sidepanel.js").read_text(encoding="utf-8")
    markup = (root / "sidepanel.html").read_text(encoding="utf-8")

    assert f'const DEFAULT_BASE_URL = "{expected}";' in script
    assert f'placeholder="{expected}"' in markup


def test_a_hidden_settings_sheet_stays_hidden() -> None:
    # .sheet declares display:flex, which outranks the user agent's [hidden]
    # rule. Without an explicit override the sheet covers the transcript and the
    # composer for the whole session, and the panel looks like it lost the
    # conversation rather than like a stylesheet lost a line.
    assert re.search(r"\.sheet\[hidden\]\s*\{[^}]*display:\s*none", _sidepanel_css()), (
        "a hidden .sheet must be display:none"
    )


def test_every_element_the_panel_script_looks_up_exists_in_the_markup() -> None:
    # getElementById returns null rather than raising, so an id that drifts out
    # of the markup surfaces as a control that silently does nothing.
    root = extension.extension_dir()
    html = (root / "sidepanel.html").read_text(encoding="utf-8")
    script = (root / "sidepanel.js").read_text(encoding="utf-8")

    looked_up = set(re.findall(r'getElementById\("([\w-]+)"\)', script))
    missing = sorted(name for name in looked_up if f'id="{name}"' not in html)

    assert not missing, f"looked up by the script but never rendered: {missing}"

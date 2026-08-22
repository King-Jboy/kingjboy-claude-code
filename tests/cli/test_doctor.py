"""Doctor checks: what they report, and that offline stays offline."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from free_claude_code.cli import doctor
from free_claude_code.cli.doctor import Finding, Level
from free_claude_code.config.constants import DEFAULT_CLIENT_CONTEXT_WINDOW
from free_claude_code.config.settings import Settings


def _settings(**overrides: Any) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _home_with_context_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    """Point the context-window lookup at a temp home holding this table."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    table = tmp_path / ".fcc" / "context.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(body, encoding="utf-8")


def test_a_recorded_window_is_reported_with_the_model_it_came_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _home_with_context_table(
        monkeypatch,
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `deepseek-ai/deepseek-v4-flash` | 1,048,576 | measured |\n",
    )
    settings = _settings(model="nvidia_nim/deepseek-ai/deepseek-v4-flash")

    (finding,) = list(doctor.check_context_window(settings))

    assert finding.level is Level.OK
    assert "1,048,576 tokens" in finding.detail
    assert "context.md" in finding.detail
    assert "nvidia_nim/deepseek-ai/deepseek-v4-flash" in finding.detail


def test_every_recorded_route_is_named_when_the_smallest_one_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The session advertises one window - the smallest recorded route - so an
    # operator chasing a too-small number needs to see which route set it.
    _home_with_context_table(
        monkeypatch,
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `small/model` | 131,072 | measured |\n\n"
        "## open_router\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `big/model` | 1,048,576 | published |\n",
    )
    settings = _settings(
        model="nvidia_nim/small/model", MODEL_SONNET="open_router/big/model"
    )

    (finding,) = list(doctor.check_context_window(settings))

    assert "131,072 tokens" in finding.detail
    assert "smallest of 2 routes" in finding.detail
    assert "nvidia_nim/small/model=131,072" in finding.detail
    assert "open_router/big/model=1,048,576" in finding.detail


def test_an_unrecorded_model_falls_back_and_says_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fallback is silent capacity loss unless doctor names the remedy.
    _home_with_context_table(monkeypatch, tmp_path, "## nvidia_nim\n")
    settings = _settings(model="nvidia_nim/never/measured")

    (finding,) = list(doctor.check_context_window(settings))

    assert f"{DEFAULT_CLIENT_CONTEXT_WINDOW:,} tokens" in finding.detail
    assert "fcc-context" in finding.detail


def test_an_explicit_window_overrides_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _home_with_context_table(
        monkeypatch,
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `a/b` | 1,000,000 | measured |\n",
    )
    # Deliberately not the fallback value, so a regression that ignored the
    # setting could not pass by coincidence.
    settings = _settings(model="nvidia_nim/a/b", CLIENT_CONTEXT_WINDOW=300000)

    (finding,) = list(doctor.check_context_window(settings))

    assert finding.detail == "300,000 tokens (set by CLIENT_CONTEXT_WINDOW)"


def test_a_configured_pool_reports_its_parsed_size() -> None:
    settings = _settings(NVIDIA_NIM_API_KEYS='["a", "b", "c"]')

    findings = list(doctor.check_key_pools(settings))

    assert [(f.level, f.check, f.detail) for f in findings] == [
        (Level.OK, "NVIDIA_NIM_API_KEYS", "3 keys pooled")
    ]


def test_a_single_key_warns_that_no_pool_is_built() -> None:
    # Below two keys KeyPool is never constructed, which is easy to miss.
    settings = _settings(NVIDIA_NIM_API_KEYS='["only"]')

    (finding,) = list(doctor.check_key_pools(settings))

    assert finding.level is Level.WARN
    assert "no pool" in finding.detail


def test_config_that_will_not_load_is_reported_not_raised(monkeypatch, capsys) -> None:
    # Settings rejects a malformed pool at construction, so the failure a user
    # actually hits is fcc-doctor itself refusing to start. It must explain.
    monkeypatch.setattr(
        doctor, "Settings", lambda: _settings(NVIDIA_NIM_API_KEYS="a, b")
    )

    assert doctor.run(["--offline"]) == 1
    out = capsys.readouterr().out
    assert "was rejected" in out
    assert "NVIDIA_NIM_API_KEYS" in out


def test_pool_env_names_come_from_the_manifest_not_from_upper_casing() -> None:
    # open_router_api_keys is exposed as OPENROUTER_API_KEYS, so upper-casing
    # the attribute would print a variable that does not exist.
    settings = _settings(OPENROUTER_API_KEYS='["a", "b"]')

    (finding,) = list(doctor.check_key_pools(settings))

    assert finding.check == "OPENROUTER_API_KEYS"


def test_a_model_routed_to_an_unconfigured_provider_fails(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "load_value_state", lambda: {})
    monkeypatch.setattr(
        doctor,
        "provider_config_status",
        lambda _state: [
            {
                "provider_id": "groq",
                "display_name": "Groq",
                "status": "missing_key",
                "label": "Missing key",
            }
        ],
    )
    settings = _settings(model="groq/some-model")

    findings = [f for f in doctor.check_providers(settings) if f.check == "MODEL"]

    assert findings[0].level is Level.FAIL
    assert "Groq" in findings[0].remedy


@respx.mock
def test_a_retired_model_is_reported_as_gone() -> None:
    # The failure this exists for: OpenRouter moved kimi-k2.6:free off the free
    # tier and nothing surfaced it until a request 404'd mid-session.
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "still/here:free"}]})
    )
    settings = _settings(model="open_router/gone/model:free")

    findings = list(doctor.check_models_still_exist(settings))

    assert findings[0].level is Level.FAIL
    assert "no longer advertises" in findings[0].detail


@respx.mock
def test_a_present_model_passes_the_catalog_check() -> None:
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "here/model:free"}]})
    )
    settings = _settings(model="open_router/here/model:free")

    findings = list(doctor.check_models_still_exist(settings))

    assert findings[0].level is Level.OK


@respx.mock
def test_an_unreachable_provider_warns_rather_than_fails() -> None:
    # A flaky network must not read as a broken install.
    respx.get("https://openrouter.ai/api/v1/models").mock(
        side_effect=httpx.ConnectError("no route")
    )
    settings = _settings(model="open_router/some/model")

    findings = list(doctor.check_models_still_exist(settings))

    assert findings[0].level is Level.WARN


def test_offline_makes_no_network_calls(monkeypatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--offline must not contact providers")

    monkeypatch.setattr(doctor.httpx, "get", explode)

    doctor.collect_findings(_settings(model="open_router/x/y"), offline=True)


def test_a_failing_finding_sets_a_nonzero_exit_code(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_findings",
        lambda _settings, offline: [Finding(Level.FAIL, "check", "broken", "fix it")],
    )

    exit_code = doctor.run([])

    assert exit_code == 1
    captured = capsys.readouterr().out
    assert "fix it" in captured


def test_a_clean_run_exits_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_findings",
        lambda _settings, offline: [Finding(Level.OK, "check", "fine")],
    )

    assert doctor.run([]) == 0
    assert "Everything checks out." in capsys.readouterr().out


def test_json_output_is_machine_readable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_findings",
        lambda _settings, offline: [Finding(Level.WARN, "check", "detail", "remedy")],
    )

    doctor.run(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "level": "warn",
            "check": "check",
            "detail": "detail",
            "remedy": "remedy",
        }
    ]


def test_help_exits_zero_without_running_checks(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "collect_findings",
        lambda *_a, **_k: pytest.fail("--help must not run checks"),
    )

    assert doctor.run(["--help"]) == 0
    assert "--offline" in capsys.readouterr().out


def test_a_missing_managed_env_warns(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(doctor, "managed_env_path", lambda: tmp_path / "absent.env")

    (finding,) = list(doctor.check_managed_env())

    assert finding.level is Level.WARN
    assert "Admin UI" in finding.remedy

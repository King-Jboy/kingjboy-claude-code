"""`fcc-context`: which models it covers, and how it merges into the table."""

import argparse
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from free_claude_code.cli import context_scan
from free_claude_code.cli.context_scan import ModelContext
from free_claude_code.config.settings import Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
NIM_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _settings(**overrides: Any) -> Settings:
    # Mirrors the Admin validation path: dotenv discovery stays off in tests.
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def test_routable_refs_are_the_pinned_list_plus_the_configured_routes() -> None:
    settings = _settings(
        model="nvidia_nim/routed/model",
        MODEL_HAIKU="nvidia_nim/haiku/model",
        PINNED_MODELS='["open_router/pinned/model", "nvidia_nim/routed/model"]',
    )

    refs = context_scan.routable_model_refs(settings)

    # Order is pinned-first, and the ref shared with a route appears once.
    assert refs == (
        "open_router/pinned/model",
        "nvidia_nim/routed/model",
        "nvidia_nim/haiku/model",
    )


def test_a_pool_supplies_the_probe_credentials_before_the_single_key() -> None:
    settings = _settings(
        nvidia_nim_api_key="single", NVIDIA_NIM_API_KEYS='["pool-a", "pool-b"]'
    )

    assert context_scan.nim_keys(settings) == ("pool-a", "pool-b")


def test_the_single_key_is_used_when_no_pool_is_configured() -> None:
    settings = _settings(nvidia_nim_api_key="single", NVIDIA_NIM_API_KEYS="")

    assert context_scan.nim_keys(settings) == ("single",)


def test_non_chat_models_are_never_probed() -> None:
    # Embedding and safety endpoints reject a chat prompt, so probing them
    # only burns rate limit that pooled keys need for real models.
    assert not context_scan.is_chat_model("nvidia/nv-embedqa-e5-v5")
    assert not context_scan.is_chat_model("meta/llama-guard-4-12b")
    assert context_scan.is_chat_model("deepseek-ai/deepseek-v4-pro")


def test_a_derived_limit_snaps_to_the_power_of_two_a_provider_configures() -> None:
    # The overflow arithmetic is exact but our token estimate for the probe
    # body is not, so 131,006 is really the usual 131,072.
    assert context_scan.nearest_power_of_two(131_006) == 131_072
    assert context_scan.nearest_power_of_two(262_157) == 262_144
    # A genuinely odd ceiling is left alone rather than forced onto a power.
    assert context_scan.nearest_power_of_two(1_000_000) == 1_000_000


def test_a_stated_ceiling_is_read_straight_out_of_the_rejection() -> None:
    with respx.mock:
        respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(
                400,
                json={
                    "message": "This model's maximum context length is 262144 tokens.",
                    "code": 400,
                },
            )
        )

        row = context_scan.probe_nim_model("a/model", "key", timeout=5.0)

    assert row.context == 262_144
    assert row.source == "measured"


def test_a_negative_token_budget_still_pins_the_ceiling() -> None:
    # Some backends subtract the prompt from the window and complain about the
    # remainder; 1,100,000 - 968,994 is the window.
    with respx.mock:
        respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(
                400,
                json={
                    "error": {
                        "message": "max_tokens must be at least 1, got -968994.",
                    }
                },
            )
        )

        row = context_scan.probe_nim_model("a/model", "key", timeout=5.0)

    assert row.context == 131_072
    assert row.source == "derived"


def test_a_model_missing_from_the_account_stops_after_one_probe() -> None:
    # Walking the whole ladder for a model that is not served wastes minutes
    # across a catalog where most models are unavailable.
    with respx.mock:
        route = respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(
                404, json={"detail": "Function 'x': Not found for account 'y'"}
            )
        )

        row = context_scan.probe_nim_model("a/model", "key", timeout=5.0)

    assert route.call_count == 1
    assert row.context is None
    assert "Not found" in row.source


def _table(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "context.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_model_no_longer_routable_leaves_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The table is a reading aid for the models you use. A catalogue that only
    # ever grows is the thing it exists to avoid.
    output = _table(
        tmp_path,
        "## open_router\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `unpinned/model` | 262,144 | published |\n",
    )
    monkeypatch.setattr(
        context_scan, "Settings", lambda: _settings(model="open_router/new/model")
    )
    with respx.mock:
        respx.get(OPENROUTER_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "new/model", "context_length": 131072},
                        {"id": "unpinned/model", "context_length": 262144},
                    ]
                },
            )
        )

        exit_code = context_scan.run(
            ["--provider", "open_router", "--output", str(output)]
        )

    assert exit_code == 0
    written = output.read_text(encoding="utf-8")
    assert "`new/model`" in written
    assert "`unpinned/model`" not in written


def test_narrowing_to_one_provider_keeps_the_others_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --provider limits what is re-measured, not what the table contains, so a
    # still-pinned NVIDIA NIM model must survive an OpenRouter-only run.
    output = _table(
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `kept/model` | 262,144 | measured |\n",
    )
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(
            model="open_router/new/model",
            PINNED_MODELS='["nvidia_nim/kept/model"]',
        ),
    )
    with respx.mock:
        respx.get(OPENROUTER_URL).mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "new/model", "context_length": 131072}]}
            )
        )

        context_scan.run(["--provider", "open_router", "--output", str(output)])

    written = output.read_text(encoding="utf-8")
    assert "`kept/model`" in written
    assert "`new/model`" in written


def test_an_already_measured_model_is_not_probed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _table(
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `known/model` | 262,144 | measured |\n",
    )
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(
            model="nvidia_nim/known/model", NVIDIA_NIM_API_KEYS='["a", "b"]'
        ),
    )
    with respx.mock:
        chat = respx.post(NIM_CHAT_URL).mock(return_value=httpx.Response(400, json={}))

        exit_code = context_scan.run(
            ["--provider", "nvidia_nim", "--output", str(output)]
        )

    assert exit_code == 0
    assert chat.call_count == 0


def test_a_hand_written_window_is_kept_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Some models cannot be measured at all -- a backend that 500s on every
    # request, or a gateway that drops the oversized body. Filling the number
    # in by hand is the only option, so a normal run must not undo it.
    output = _table(
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `unmeasurable/model` | 270,000 | manual |\n",
    )
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(
            model="nvidia_nim/unmeasurable/model", NVIDIA_NIM_API_KEYS='["a", "b"]'
        ),
    )
    with respx.mock:
        chat = respx.post(NIM_CHAT_URL).mock(return_value=httpx.Response(500, json={}))

        context_scan.run(["--provider", "nvidia_nim", "--output", str(output)])

    assert chat.call_count == 0
    written = output.read_text(encoding="utf-8")
    assert "| `unmeasurable/model` | 270,000 | manual |" in written


def test_refresh_re_probes_a_recorded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _table(
        tmp_path,
        "## nvidia_nim\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `known/model` | 262,144 | measured |\n",
    )
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(
            model="nvidia_nim/known/model", NVIDIA_NIM_API_KEYS='["a", "b"]'
        ),
    )
    with respx.mock:
        respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(
                400,
                json={"message": "This model's maximum context length is 999 tokens."},
            )
        )

        context_scan.run(
            ["--provider", "nvidia_nim", "--refresh", "--output", str(output)]
        )

    assert "| 999 |" in output.read_text(encoding="utf-8")


def test_routing_only_to_unmeasurable_providers_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Someone on a provider with no published, curated, or probeable window gets
    # a run that covers nothing. Reporting zero models would read as a bug
    # rather than as a routing choice.
    monkeypatch.setattr(
        context_scan, "Settings", lambda: _settings(model="huggingface/some/model")
    )

    exit_code = context_scan.run(["--output", str(tmp_path / "context.md")])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "--all" in stderr
    assert "nvidia_nim" in stderr


def test_the_rendered_table_round_trips_through_the_reader(tmp_path: Path) -> None:
    # render() and read_existing() are the two halves of the merge, so a shape
    # only one of them understands would silently drop rows on the next run.
    rows = [
        ModelContext("nvidia_nim", "a/one", 1_048_576, "measured"),
        ModelContext("nvidia_nim", "b/two", None, "Internal server error"),
        ModelContext("open_router", "c/three:free", 131_072, "published"),
    ]
    path = _table(tmp_path, context_scan.render(rows))

    parsed = context_scan.read_existing(path)

    assert parsed[("nvidia_nim", "a/one")].context == 1_048_576
    assert parsed[("nvidia_nim", "b/two")].context is None
    assert parsed[("open_router", "c/three:free")].context == 131_072


# ---------- layered resolution ----------


GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def test_a_curated_provider_resolves_without_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DeepSeek documents its window but publishes nothing; the curated table
    # fills it in with zero requests spent.
    output = tmp_path / "context.md"
    monkeypatch.setattr(
        context_scan, "Settings", lambda: _settings(model="deepseek/deepseek-chat")
    )

    exit_code = context_scan.run(["--output", str(output)])

    assert exit_code == 0
    written = output.read_text(encoding="utf-8")
    assert "| `deepseek-chat` | 128,000 | curated |" in written


def test_groq_publishes_its_windows_when_a_key_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "context.md"
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(
            model="groq/llama-3.3-70b-versatile", GROQ_API_KEY="groq-key"
        ),
    )
    with respx.mock:
        respx.get(GROQ_MODELS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "llama-3.3-70b-versatile", "context_window": 131072},
                        {"id": "future/model", "context_window": 65536},
                    ]
                },
            )
        )

        exit_code = context_scan.run(["--output", str(output)])

    assert exit_code == 0
    written = output.read_text(encoding="utf-8")
    assert "| `llama-3.3-70b-versatile` | 131,072 | published |" in written


def test_groq_without_a_key_falls_back_to_the_curated_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "context.md"
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(model="groq/llama-3.3-70b-versatile"),
    )

    exit_code = context_scan.run(["--output", str(output)])

    assert exit_code == 0
    written = output.read_text(encoding="utf-8")
    assert "| `llama-3.3-70b-versatile` | 131,072 | curated |" in written
    assert "no GROQ_API_KEY; curated values only" in capsys.readouterr().err


def test_a_recorded_number_beats_a_conflicting_published_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The operator corrected a row by hand; the next run must not drag it back
    # to whatever the catalog claims.
    output = _table(
        tmp_path,
        "## open_router\n\n| Model | Context | Source |\n| --- | ---: | --- |\n"
        "| `model` | 270,000 | manual |\n",
    )
    monkeypatch.setattr(
        context_scan, "Settings", lambda: _settings(model="open_router/model")
    )
    with respx.mock:
        respx.get(OPENROUTER_URL).mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "model", "context_length": 131072}]}
            )
        )

        context_scan.run(["--provider", "open_router", "--output", str(output)])

    written = output.read_text(encoding="utf-8")
    assert "| `model` | 270,000 | manual |" in written


def test_no_probe_leaves_new_models_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "context.md"
    monkeypatch.setattr(
        context_scan,
        "Settings",
        lambda: _settings(model="nvidia_nim/new/model", NVIDIA_NIM_API_KEYS='["a"]'),
    )
    with respx.mock:
        chat = respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(400, json={"message": "no"})
        )

        exit_code = context_scan.run(
            ["--provider", "nvidia_nim", "--no-probe", "--output", str(output)]
        )

    assert exit_code == 0
    assert chat.call_count == 0
    written = output.read_text(encoding="utf-8")
    assert "| `new/model` | unknown | not probed (--no-probe) |" in written


def test_a_rate_limited_probe_waits_and_retries_on_another_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 429 is the scan's own pacing, not the model's ceiling: recording it as
    # an unknown turned a throttle into a permanent wrong answer.
    monkeypatch.setattr(context_scan.time, "sleep", lambda _s: None)
    with respx.mock:
        route = respx.post(NIM_CHAT_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "1"}),
                httpx.Response(
                    400,
                    json={
                        "message": "This model's maximum context length is 262144 tokens."
                    },
                ),
            ]
        )

        row = context_scan.nim_rows(
            _nim_args(),
            _settings(NVIDIA_NIM_API_KEYS='["a", "b"]'),
            {},
            frozenset({"model"}),
        )[0]

    assert route.call_count == 2
    assert row.context == 262_144
    assert row.source == "measured"


def test_a_model_that_swallows_the_largest_probe_is_not_escalated() -> None:
    # Escalating past an accepted probe buys real prefill on giant-window
    # models for a number nothing needs; the rung is recorded as a floor note.
    with respx.mock:
        route = respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"index": 0}]})
        )

        row = context_scan.probe_nim_model("a/model", "key", timeout=5.0)

    assert route.call_count == 1
    assert row.context is None
    assert "accepted" in row.source


def test_an_unrecognized_rejection_carries_the_providers_own_words() -> None:
    # When NVIDIA rewords the rejection, the regexes miss and the note is the
    # only thing that makes the failure fixable.
    with respx.mock:
        respx.post(NIM_CHAT_URL).mock(
            return_value=httpx.Response(
                400, text="Your prompt exceeds the allowed enormity quota."
            )
        )

        row = context_scan.probe_nim_model("a/model", "key", timeout=5.0)

    assert row.context is None
    assert "enormity quota" in row.source


def _nim_args() -> argparse.Namespace:
    return argparse.Namespace(
        refresh=False,
        probe=True,
        workers=2,
        timeout=5.0,
    )

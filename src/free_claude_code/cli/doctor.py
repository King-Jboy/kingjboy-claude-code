"""Diagnose a Free Claude Code install before it fails mid-session.

The checks here are the ones that otherwise only surface as confusing runtime
behaviour: a model that quietly left a provider's free tier, a credential that
expired, a pool that parsed to nothing. Each check reports one line so the
output stays readable when everything is fine.

This module stays inside the ``cli -> config, core`` import boundary. It never
constructs a provider, so an offline run spends no credentials and cannot be
slowed down by a wedged upstream.
"""

import json
import shutil
import socket
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

import httpx
from pydantic import ValidationError

from free_claude_code.config.admin.manifest import FIELDS
from free_claude_code.config.admin.status import provider_config_status
from free_claude_code.config.admin.values import load_value_state
from free_claude_code.config.api_keys import parse_api_key_list
from free_claude_code.config.context_windows import (
    CONTEXT_WINDOWS_FILENAME,
    recorded_route_windows,
    resolve_client_context_window,
)
from free_claude_code.config.model_refs import parse_model_name, parse_provider_type
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings

MODEL_ROUTE_ATTRS = (
    ("MODEL", "model"),
    ("MODEL_FABLE", "model_fable"),
    ("MODEL_OPUS", "model_opus"),
    ("MODEL_SONNET", "model_sonnet"),
    ("MODEL_HAIKU", "model_haiku"),
)

NETWORK_TIMEOUT_SECONDS = 10.0


class Level(StrEnum):
    """How much attention a finding deserves."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


_MARK = {Level.OK: "  ok  ", Level.WARN: " warn ", Level.FAIL: " FAIL "}


@dataclass(frozen=True, slots=True)
class Finding:
    """One diagnosed condition and, when it is not ok, what to do about it."""

    level: Level
    check: str
    detail: str
    remedy: str = ""

    def render(self) -> str:
        line = f"[{_MARK[self.level]}] {self.check}: {self.detail}"
        if self.remedy and self.level is not Level.OK:
            line += f"\n{' ' * 9}-> {self.remedy}"
        return line


def check_managed_env() -> Iterator[Finding]:
    """Report whether the Admin-managed env file exists."""
    path = managed_env_path()
    if path.is_file():
        yield Finding(Level.OK, "managed env", str(path))
    else:
        yield Finding(
            Level.WARN,
            "managed env",
            f"no file at {path}",
            "Open the Admin UI and save once to create it.",
        )


def check_port(settings: Settings) -> Iterator[Finding]:
    """Report whether the configured port is already taken."""
    port = settings.port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        in_use = probe.connect_ex(("127.0.0.1", port)) == 0
    if in_use:
        yield Finding(
            Level.OK,
            "server port",
            f"{port} is serving (an fcc-server is already running)",
        )
    else:
        yield Finding(Level.OK, "server port", f"{port} is free")


def check_claude_cli() -> Iterator[Finding]:
    """Report whether the Claude Code CLI is reachable on PATH."""
    resolved = shutil.which("claude")
    if resolved:
        yield Finding(Level.OK, "claude cli", resolved)
    else:
        yield Finding(
            Level.WARN,
            "claude cli",
            "not on PATH",
            "Install Claude Code, or ignore this if you drive FCC from another client.",
        )


def check_providers(settings: Settings) -> Iterator[Finding]:
    """Report configuration state for every provider a model routes to."""
    statuses = {
        status["provider_id"]: status
        for status in provider_config_status(load_value_state())
    }
    for env_name, attr in MODEL_ROUTE_ATTRS:
        model_ref = getattr(settings, attr, None)
        if not model_ref:
            continue
        provider_id = parse_provider_type(model_ref)
        status = statuses.get(provider_id)
        if status is None:
            yield Finding(
                Level.FAIL,
                env_name,
                f"{model_ref} names unknown provider {provider_id!r}",
                "Fix the provider prefix in the Admin UI.",
            )
            continue
        if status["status"] == "configured":
            yield Finding(Level.OK, env_name, model_ref)
        else:
            yield Finding(
                Level.FAIL,
                env_name,
                f"{model_ref} routes to {provider_id}, which is {status['label']!r}",
                f"Set credentials for {status['display_name']} in the Admin UI.",
            )


def check_context_window(settings: Settings) -> Iterator[Finding]:
    """Report the context window advertised to launched client CLIs.

    Saying where the number came from is the point: the same value means very
    different things when it was measured for your model versus when it is the
    conservative fallback nothing matched. With several routes recorded, the
    smallest one wins for the whole session, so each is named - an operator
    chasing a too-small window needs to see which route set it.
    """
    resolved = resolve_client_context_window(
        settings, configured=settings.client_context_window
    )
    detail = f"{resolved.value:,} tokens"
    if resolved.source == CONTEXT_WINDOWS_FILENAME:
        routes = recorded_route_windows(settings)
        if len(routes) > 1:
            listing = " · ".join(f"{ref}={value:,}" for ref, value in routes)
            yield Finding(
                Level.OK,
                "context window",
                f"{detail} (from {CONTEXT_WINDOWS_FILENAME}: smallest of "
                f"{len(routes)} routes - {listing})",
            )
            return
        yield Finding(
            Level.OK,
            "context window",
            f"{detail} (from {CONTEXT_WINDOWS_FILENAME} for {resolved.model_ref})",
        )
        return
    if resolved.source == "default":
        yield Finding(
            Level.OK,
            "context window",
            f"{detail} (default; no {CONTEXT_WINDOWS_FILENAME} entry for your "
            "models -- run fcc-context)",
        )
        return
    yield Finding(
        Level.OK, "context window", f"{detail} (set by CLIENT_CONTEXT_WINDOW)"
    )


def check_key_pools(settings: Settings) -> Iterator[Finding]:
    """Report the parsed size of every configured credential pool."""
    env_names = {field.settings_attr: field.key for field in FIELDS}
    for descriptor in PROVIDER_CATALOG.values():
        attr = descriptor.credential_pool_attr
        if attr is None:
            continue
        raw = getattr(settings, attr, "") or ""
        # Upper-casing the attribute is not the env name; OPENROUTER_API_KEYS
        # is backed by open_router_api_keys. The manifest owns the mapping.
        env_name = env_names.get(attr, attr.upper())
        if not raw.strip():
            continue
        # Settings already rejected malformed pools at construction, so this
        # parse cannot fail here; run() reports that case instead.
        keys = parse_api_key_list(raw, env_name=env_name)
        if len(keys) == 1:
            yield Finding(
                Level.WARN,
                env_name,
                "1 key, so no pool is built",
                "Pools need two or more keys; the single key is used directly.",
            )
        else:
            yield Finding(Level.OK, env_name, f"{len(keys)} keys pooled")


def check_models_still_exist(settings: Settings) -> Iterator[Finding]:
    """Confirm each routed model is still advertised by its provider.

    Providers retire models and move them off free tiers without notice, and
    the resulting 404 only appears once a real request is already in flight.
    """
    seen: set[str] = set()
    for env_name, attr in MODEL_ROUTE_ATTRS:
        model_ref = getattr(settings, attr, None)
        if not model_ref or model_ref in seen:
            continue
        seen.add(model_ref)
        provider_id = parse_provider_type(model_ref)
        descriptor = PROVIDER_CATALOG.get(provider_id)
        if descriptor is None or not descriptor.default_base_url:
            continue
        model_name = parse_model_name(model_ref)
        try:
            advertised = _advertised_model_ids(descriptor.default_base_url)
        except Exception as error:
            yield Finding(
                Level.WARN,
                f"{env_name} catalog",
                f"could not reach {provider_id}: {type(error).__name__}",
                "Re-run with --offline to skip network checks.",
            )
            continue
        if not advertised:
            continue
        if model_name in advertised:
            yield Finding(Level.OK, f"{env_name} catalog", f"{model_name} is available")
        else:
            yield Finding(
                Level.FAIL,
                f"{env_name} catalog",
                f"{provider_id} no longer advertises {model_name}",
                "Pick a current model in the Admin UI; providers retire these silently.",
            )


def _advertised_model_ids(base_url: str) -> frozenset[str]:
    """Return model ids from an OpenAI-compatible ``/models`` endpoint."""
    response = httpx.get(
        f"{base_url.rstrip('/')}/models", timeout=NETWORK_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return frozenset()
    return frozenset(
        entry["id"]
        for entry in payload.get("data", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def collect_findings(settings: Settings, *, offline: bool) -> list[Finding]:
    """Run every check and return the findings in report order."""
    findings = [
        *check_managed_env(),
        *check_port(settings),
        *check_claude_cli(),
        *check_providers(settings),
        *check_context_window(settings),
        *check_key_pools(settings),
    ]
    if not offline:
        findings.extend(check_models_still_exist(settings))
    return findings


def _report(findings: Sequence[Finding], *, as_json: bool) -> str:
    if as_json:
        return json.dumps(
            [
                {
                    "level": finding.level.value,
                    "check": finding.check,
                    "detail": finding.detail,
                    "remedy": finding.remedy,
                }
                for finding in findings
            ],
            indent=2,
        )
    lines = [finding.render() for finding in findings]
    failures = sum(1 for finding in findings if finding.level is Level.FAIL)
    warnings = sum(1 for finding in findings if finding.level is Level.WARN)
    if failures:
        lines.append(f"\n{failures} failing, {warnings} warning.")
    elif warnings:
        lines.append(f"\nNo failures, {warnings} warning.")
    else:
        lines.append("\nEverything checks out.")
    return "\n".join(lines)


def run(argv: Sequence[str] | None = None) -> int:
    """Diagnose the install and return a shell exit code."""
    args = sys.argv[1:] if argv is None else list(argv)
    if "--help" in args or "-h" in args:
        print(
            "Usage: fcc-doctor [--offline] [--json]\n\n"
            "  --offline  Skip checks that contact providers.\n"
            "  --json     Emit findings as JSON.\n"
        )
        return 0

    try:
        settings = Settings()
    except ValidationError as error:
        # The likeliest reason to run doctor at all: config is bad enough that
        # the server will not boot. A pydantic traceback is not an answer.
        print(_config_rejected(error))
        return 1

    findings = collect_findings(settings, offline="--offline" in args)
    print(_report(findings, as_json="--json" in args))
    return 1 if any(f.level is Level.FAIL for f in findings) else 0


def _config_rejected(error: ValidationError) -> str:
    """Render a config that would not load as findings rather than a traceback."""
    lines = [f"[{_MARK[Level.FAIL]}] config: {managed_env_path()} was rejected"]
    for problem in error.errors():
        field = ".".join(str(part) for part in problem["loc"]) or "settings"
        lines.append(f"{' ' * 9}-> {field}: {problem['msg']}")
    lines.append("\nFix these in the Admin UI, then re-run fcc-doctor.")
    return "\n".join(lines)

"""Record each routable model's real context window in ``~/.fcc/context.md``.

Registered as ``fcc-context``. FCC reads that table to decide what context
window to advertise to a launched CLI, so leaving ``CLIENT_CONTEXT_WINDOW``
blank makes the window follow whichever model ``MODEL`` points at. This command
is what fills the table in.

Most OpenAI-compatible providers never put a context length on the wire --
NVIDIA NIM's ``/v1/models`` returns only ``id``, ``object``, ``created``,
``owned_by`` -- so the number is read where published and measured where not:

* **OpenRouter** publishes ``context_length`` for every model. Instant, and no
  key is required.
* **NVIDIA NIM** publishes nothing, so each model gets one deliberately
  oversized request. The rejection states the ceiling ("This model's maximum
  context length is 262144 tokens"), which costs no inference.

The table holds exactly the models you can route to -- everything in
``PINNED_MODELS`` plus the ``MODEL`` / ``MODEL_*`` routes -- so it stays a
readable list of models you actually use rather than a growing catalogue. Add a
model to your pinned list, re-run, and its window joins the table; unpin one and
it leaves. Already-recorded models are not probed again, so adding a model costs
one probe rather than a full sweep.

This module stays inside the ``cli -> config, core`` import boundary. It talks
to provider HTTP endpoints directly rather than constructing a provider, the
same way ``doctor`` checks a model still exists.
"""

import argparse
import re
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from free_claude_code.config.api_keys import parse_api_key_list
from free_claude_code.config.context_windows import context_windows_path
from free_claude_code.config.model_refs import (
    configured_chat_model_refs,
    pinned_model_refs,
)
from free_claude_code.config.settings import Settings

NIM_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

SUPPORTED_PROVIDERS = ("nvidia_nim", "open_router")

# Probe sizes in tokens, largest first, because a rejection states the ceiling
# no matter how far over it the probe was. Overshooting therefore costs one
# fast request; undershooting makes the model actually prefill the prompt,
# which is the slow path. The smaller rungs exist only as a fallback for
# gateways that drop the connection on a multi-megabyte body.
PROBE_LADDER = (1_100_000, 400_000, 200_000)

# Only a model whose window exceeds the first rung needs escalation.
ESCALATION_LADDER = (2_200_000, 4_400_000)

# Models that cannot take a chat prompt at all. Probing them yields nothing but
# noise and burns rate limit.
NON_CHAT_MARKERS = (
    "embed",
    "rerank",
    "retriev",
    "guard",
    "safety",
    "reward",
    "-parse",
    "nemoretriever",
    "nvclip",
    "translate",
    "deplot",
    "kosmos",
    "diffusion",
    "video-detector",
    "calibration",
)

# Backends word the same rejection differently; both spellings are in the wild.
STATED_LIMIT = re.compile(
    r"(?:maximum context length is|context length is only)\s*(\d+)", re.I
)
# Some backends subtract the prompt from the window and complain about the
# negative remainder, which still pins the ceiling exactly.
NEGATIVE_BUDGET = re.compile(r"max_tokens must be at least 1, got -(\d+)", re.I)

TABLE_ROW = re.compile(
    r"^\|\s*`(?P<model>[^`]+)`\s*\|\s*(?P<context>[^|]+?)\s*\|\s*(?P<source>[^|]+?)\s*\|$"
)


@dataclass(frozen=True, slots=True)
class ModelContext:
    """One model's measured or published context window."""

    provider: str
    model: str
    context: int | None
    source: str

    @property
    def context_cell(self) -> str:
        return f"{self.context:,}" if self.context else "unknown"


def routable_model_refs(settings: Settings) -> tuple[str, ...]:
    """Return every ``provider/model`` ref this install can route to.

    That is the pinned list plus the configured tier routes -- exactly the set
    whose windows FCC will look up, so covering anything else is wasted effort.
    """
    refs = [
        *pinned_model_refs(settings),
        *(ref.model_ref for ref in configured_chat_model_refs(settings)),
    ]
    return tuple(dict.fromkeys(refs))


def nim_keys(settings: Settings) -> tuple[str, ...]:
    """Return every configured NVIDIA NIM credential, pool first."""
    pool = parse_api_key_list(
        settings.nvidia_nim_api_keys, env_name="NVIDIA_NIM_API_KEYS"
    )
    if pool:
        return pool
    single = settings.nvidia_nim_api_key.strip()
    return (single,) if single else ()


def is_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(marker in lowered for marker in NON_CHAT_MARKERS)


def openrouter_contexts() -> list[ModelContext]:
    """Read published context lengths; OpenRouter states them for every model."""
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=60.0)
    response.raise_for_status()
    rows: list[ModelContext] = []
    for item in response.json().get("data", []):
        model_id = item.get("id")
        context = item.get("context_length")
        if not isinstance(model_id, str) or not model_id:
            continue
        rows.append(
            ModelContext(
                provider="open_router",
                model=model_id,
                context=context if isinstance(context, int) and context > 0 else None,
                source="published",
            )
        )
    return rows


def nim_model_ids(key: str) -> list[str]:
    response = httpx.get(
        NIM_MODELS_URL, headers={"authorization": f"Bearer {key}"}, timeout=60.0
    )
    response.raise_for_status()
    return sorted(
        item["id"]
        for item in response.json().get("data", [])
        if isinstance(item.get("id"), str)
    )


def _probe_once(model: str, key: str, size: int, timeout: float) -> ModelContext | str:
    """Send one oversized request. Returns a resolved limit, or why it failed."""
    try:
        response = httpx.post(
            NIM_CHAT_URL,
            headers={"authorization": f"Bearer {key}"},
            json={
                "model": model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "word " * size}],
            },
            timeout=timeout,
        )
    except Exception as exc:
        return type(exc).__name__

    text = response.text
    if match := STATED_LIMIT.search(text):
        return ModelContext("nvidia_nim", model, int(match.group(1)), "measured")
    if match := NEGATIVE_BUDGET.search(text):
        # Round to the nearest power of two: the arithmetic is exact but our
        # token estimate for the probe body is not.
        derived = size - int(match.group(1))
        return ModelContext(
            "nvidia_nim", model, nearest_power_of_two(derived), "derived"
        )

    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"

    if isinstance(payload, dict) and payload.get("choices"):
        return f"accepted >={size:,}"
    return _error_note(payload) or f"HTTP {response.status_code}"


def probe_nim_model(model: str, key: str, *, timeout: float) -> ModelContext:
    """Find one model's ceiling, preferring a single overshooting request.

    The first rung is already past every window a provider is likely to serve,
    so the usual outcome is one rejection carrying the exact number. Smaller
    rungs are tried only when the transport rejects the body outright, and
    larger ones only when the model genuinely swallowed the first probe.
    """
    note = "no response"
    for size in PROBE_LADDER:
        outcome = _probe_once(model, key, size, timeout)
        if isinstance(outcome, ModelContext):
            return outcome
        note = outcome
        if outcome.startswith("accepted"):
            break
        # A model that is missing or broken will not answer any smaller probe.
        if "Not found" in outcome or "Internal server error" in outcome:
            return ModelContext("nvidia_nim", model, None, note)

    if note.startswith("accepted"):
        for size in ESCALATION_LADDER:
            outcome = _probe_once(model, key, size, timeout)
            if isinstance(outcome, ModelContext):
                return outcome
            note = outcome
            if not outcome.startswith("accepted"):
                break
    return ModelContext("nvidia_nim", model, None, note)


def nearest_power_of_two(value: int) -> int:
    """Snap a derived limit to the power of two providers actually configure."""
    if value <= 0:
        return value
    candidates = [1 << bit for bit in range(10, 25)]
    closest = min(candidates, key=lambda candidate: abs(candidate - value))
    return closest if abs(closest - value) <= max(64, value // 1000) else value


def _error_note(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("message", "detail", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value[:80]
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message[:80]
    return ""


def read_existing(path: Path) -> dict[tuple[str, str], ModelContext]:
    """Parse a previous run so re-runs only probe what is genuinely new."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    known: dict[tuple[str, str], ModelContext] = {}
    provider = ""
    for line in text.splitlines():
        if line.startswith("## "):
            provider = line[3:].strip().split(" ")[0]
            continue
        match = TABLE_ROW.match(line.strip())
        if not match or not provider:
            continue
        raw = match.group("context").replace(",", "").strip()
        context = int(raw) if raw.isdigit() else None
        model = match.group("model")
        known[(provider, model)] = ModelContext(
            provider, model, context, match.group("source")
        )
    return known


def render(rows: Sequence[ModelContext]) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        "# Model context windows",
        "",
        f"Generated by `fcc-context` on {stamp}.",
        "",
        "FCC reads this table to decide what context window to advertise to a",
        "launched CLI. Leave `CLIENT_CONTEXT_WINDOW` blank and the window follows",
        "whichever model `MODEL` routes to; set it to override this table.",
        "",
        "This lists exactly the models you can route to -- `PINNED_MODELS` plus",
        "your `MODEL` / `MODEL_*` routes. Add a model and re-run `fcc-context` to",
        "measure it; remove one and it leaves on the next run. Models already",
        "measured are not probed again.",
        "",
        "`published` came from the provider's own metadata. `measured` came from the",
        "ceiling the provider stated when refusing an oversized request. `derived`",
        "was computed from a reported token overflow and snapped to the nearest",
        "power of two. Anything else is the reason no number could be obtained --",
        "most often that the model is listed but not served to your account.",
        "",
        "You can fill a number in by hand for a model that cannot be measured;",
        "mark it `manual` so it is not mistaken for a reading. Any row with a",
        "number is kept as-is on later runs -- only `fcc-context --refresh`",
        "re-measures it.",
        "",
    ]
    by_provider: dict[str, list[ModelContext]] = {}
    for row in rows:
        by_provider.setdefault(row.provider, []).append(row)

    for provider in sorted(by_provider):
        entries = by_provider[provider]
        known = [entry for entry in entries if entry.context]
        unknown = [entry for entry in entries if not entry.context]
        lines += [f"## {provider} ({len(known)} of {len(entries)} known)", ""]
        lines += ["| Model | Context | Source |", "| --- | ---: | --- |"]
        for entry in sorted(known, key=lambda e: (-(e.context or 0), e.model)):
            lines.append(f"| `{entry.model}` | {entry.context_cell} | {entry.source} |")
        for entry in sorted(unknown, key=lambda e: e.model):
            lines.append(f"| `{entry.model}` | unknown | {entry.source} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fcc-context",
        description=(
            "Record the context window of every model you can route to, so "
            "CLIENT_CONTEXT_WINDOW can resolve itself."
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=list(SUPPORTED_PROVIDERS),
        help="Limit to one provider; repeatable. Defaults to both.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Cover every model each provider lists, not just the ones you can "
            "route to. Slow, and mostly records models you will never select."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=(),
        help="Cover only these provider/model refs, ignoring your configuration.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-probe models already recorded instead of reusing their value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown file to write (default ~/.fcc/context.md).",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Concurrent probes (default 6)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help=(
            "Seconds to wait per probe (default 180). A rejection is fast; only "
            "a model that accepts the probe is slow, and that answer is not "
            "worth waiting many minutes for."
        ),
    )
    return parser.parse_args(argv)


def nim_rows(
    args: argparse.Namespace,
    settings: Settings,
    known: dict[tuple[str, str], ModelContext],
    wanted: frozenset[str] | None,
) -> list[ModelContext]:
    keys = nim_keys(settings)
    if not keys:
        print("  no NVIDIA NIM credential configured, skipping", file=sys.stderr)
        return []

    if wanted is None:
        models = [m for m in nim_model_ids(keys[0]) if is_chat_model(m)]
    else:
        models = sorted(wanted)
        if not models:
            print("  no nvidia_nim models routed or pinned", file=sys.stderr)
            return []
    reused = [
        known[("nvidia_nim", model)]
        for model in models
        if not args.refresh
        and ("nvidia_nim", model) in known
        and known[("nvidia_nim", model)].context
    ]
    reused_ids = {row.model for row in reused}
    todo = [model for model in models if model not in reused_ids]
    print(f"  {len(reused)} already known, probing {len(todo)}", file=sys.stderr)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        probed = list(
            pool.map(
                lambda pair: probe_nim_model(
                    pair[1], keys[pair[0] % len(keys)], timeout=args.timeout
                ),
                enumerate(todo),
            )
        )
    for row in probed:
        status = f"{row.context:,}" if row.context else f"? ({row.source})"
        print(f"    {row.model:<48} {status}", file=sys.stderr)
    print(f"  probed in {time.perf_counter() - started:.0f}s", file=sys.stderr)
    return reused + probed


def selected_models_by_provider(
    args: argparse.Namespace, settings: Settings
) -> dict[str, frozenset[str]] | None:
    """Return the per-provider model ids to cover, or None to cover everything.

    An empty mapping means nothing routable belongs to a provider this command
    can measure. That is worth saying out loud: the alternative is a run that
    reports zero models and looks like a bug rather than a routing choice.
    """
    if args.all:
        return None
    selected: dict[str, set[str]] = {}
    for ref in tuple(args.models) or routable_model_refs(settings):
        provider, _, model = ref.partition("/")
        if provider in SUPPORTED_PROVIDERS and model:
            selected.setdefault(provider, set()).add(model)
    if not selected:
        print(
            "None of your pinned models or model routes use a provider this "
            f"command can measure ({', '.join(SUPPORTED_PROVIDERS)}). "
            "Pass --all or --models to choose explicitly.",
            file=sys.stderr,
        )
    return {provider: frozenset(models) for provider, models in selected.items()}


def _in_scope(key: tuple[str, str], selected: dict[str, frozenset[str]] | None) -> bool:
    """Whether a recorded row belongs in the table this run is writing."""
    if selected is None:
        return True
    provider, model = key
    return model in selected.get(provider, frozenset())


def run(argv: Sequence[str] | None = None) -> int:
    """Record context windows and return a shell exit code."""
    args = parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        settings = Settings()
    except ValidationError as error:
        print(f"{Settings.model_config.get('env_file')} was rejected:", file=sys.stderr)
        for problem in error.errors():
            field = ".".join(str(part) for part in problem["loc"]) or "settings"
            print(f"  {field}: {problem['msg']}", file=sys.stderr)
        return 1

    output = args.output or context_windows_path()
    selected = selected_models_by_provider(args, settings)
    if selected == {}:
        return 1
    providers: Iterable[str] = args.provider or SUPPORTED_PROVIDERS
    known = read_existing(output)
    rows: list[ModelContext] = []

    if "open_router" in providers:
        print("open_router: reading published metadata", file=sys.stderr)
        try:
            published = openrouter_contexts()
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            if selected is not None:
                wanted = selected.get("open_router", frozenset())
                published = [row for row in published if row.model in wanted]
            print(f"  {len(published)} models", file=sys.stderr)
            rows += published

    if "nvidia_nim" in providers:
        print("nvidia_nim: no published context length, probing", file=sys.stderr)
        rows += nim_rows(
            args,
            settings,
            known,
            None if selected is None else selected.get("nvidia_nim", frozenset()),
        )

    # The table holds exactly what is in scope and nothing else, so it stays a
    # readable list of the models you actually use. Rows this run did not visit
    # are carried over only when they are still in scope -- that is what keeps
    # `--provider open_router` from wiping your NVIDIA NIM rows, while a model
    # you unpinned drops out instead of accumulating forever.
    covered = {(row.provider, row.model) for row in rows}
    rows += [
        row
        for key, row in known.items()
        if key not in covered and _in_scope(key, selected)
    ]

    if not rows:
        print("nothing to write", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(rows), encoding="utf-8")
    resolved = sum(1 for row in rows if row.context)
    print(
        f"\nwrote {output} ({resolved} of {len(rows)} models resolved)",
        file=sys.stderr,
    )
    return 0

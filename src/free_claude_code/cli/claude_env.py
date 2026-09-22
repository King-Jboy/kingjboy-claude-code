"""Shared Claude Code environment policy for FCC client surfaces."""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from free_claude_code.cli.local_http import with_local_proxy_bypass
from free_claude_code.cli.proxy_auth import proxy_auth_token
from free_claude_code.config.constants import DEFAULT_CLIENT_CONTEXT_WINDOW

CLAUDE_BINARY_NAME = "claude"


def resolve_claude_executable(claude_bin: str = CLAUDE_BINARY_NAME) -> str:
    """Resolve the Claude Code executable path with fallback to common install dirs."""
    found = shutil.which(claude_bin)
    if found:
        return found

    bin_path = Path(claude_bin)
    if bin_path.is_file() and (os.name == "nt" or os.access(bin_path, os.X_OK)):
        return str(bin_path)

    home = Path.home()
    candidates: list[Path] = [
        home / ".local" / "bin" / "claude",
        home / ".local" / "bin" / "claude.exe",
        home / ".npm-global" / "bin" / "claude",
        home / ".npm-global" / "bin" / "claude.cmd",
        home / ".cargo" / "bin" / "claude",
        Path("/usr/local/bin/claude"),
    ]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm" / "claude.cmd")
            candidates.append(Path(appdata) / "npm" / "claude.exe")

    for candidate in candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate)

    return claude_bin


_BLOCKED_ENV_PREFIXES = (
    "ANTHROPIC_",
    "OPENROUTER_",
    "NVIDIA_NIM_",
    "DEEPSEEK_",
    "KIMI_",
    "GEMINI_",
    "GROQ_",
    "ZAI_",
    "CUSTOM_",
    "TOKENROUTER_",
    "NARAROUTE_",
)
_BLOCKED_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "HUGGINGFACE_API_KEY",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    }
)


def build_claude_proxy_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    base_env: Mapping[str, str],
    context_window: int = DEFAULT_CLIENT_CONTEXT_WINDOW,
) -> dict[str, str]:
    """Return the canonical environment for Claude Code proxy sessions.

    ``context_window`` is what Claude Code compacts against. It is a user
    setting rather than a constant because the routed model owns the real
    limit and most OpenAI-compatible ``/v1/models`` responses omit it.
    """

    # Claude's aggregate traffic flag also suppresses gateway model discovery.
    env = with_local_proxy_bypass(
        {
            key: value
            for key, value in base_env.items()
            if not any(key.startswith(p) for p in _BLOCKED_ENV_PREFIXES)
            and key not in _BLOCKED_ENV_KEYS
        },
        proxy_root_url=proxy_root_url,
    )
    env["ANTHROPIC_BASE_URL"] = proxy_root_url
    env["ANTHROPIC_AUTH_TOKEN"] = proxy_auth_token(auth_token)
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    # FCC uses Claude's separate classifier requests, not Anthropic server checks.
    env["CLAUDE_CODE_AUTO_MODE_SERVER"] = "0"
    env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] = "1"
    env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(context_window)
    env["DISABLE_AUTOUPDATER"] = "1"
    env["DISABLE_FEEDBACK_COMMAND"] = "1"
    env["DISABLE_ERROR_REPORTING"] = "1"
    return env

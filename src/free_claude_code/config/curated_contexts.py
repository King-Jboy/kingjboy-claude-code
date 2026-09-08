"""Context windows no endpoint publishes, recorded from provider documentation.

Most OpenAI-compatible providers state a model's context length nowhere on the
wire, but document it. Those numbers change rarely and almost never grow, so a
small curated table beats both probing (which costs requests and breaks when a
provider rewords a rejection) and leaving the window unresolved.

Entries are exact model ids, except a key ending in ``-`` which matches every
id in that family (``codestral-`` covers ``codestral-latest`` and dated
revisions alike). Matching is longest-prefix-wins, so a specific entry always
beats a family entry.

Every value here is overridable: a number recorded in ``~/.fcc/context.md`` -
including one written by hand and marked ``manual`` - is kept as-is on later
``fcc-context`` runs, so a wrong or stale curated figure never wins against
the operator's own table.
"""

CURATED_CONTEXT_WINDOWS: dict[str, dict[str, int]] = {
    # DeepSeek documents 128K for chat/reasoner; V4 family supports 1M.
    "deepseek": {
        "deepseek-chat": 128_000,
        "deepseek-reasoner": 128_000,
        "deepseek-v4-pro-0813": 1_048_576,
        "deepseek-v4-flash-0731": 1_048_576,
        "deepseek-v4-flash": 1_048_576,
        "deepseek-v4": 1_048_576,
    },
    # NVIDIA NIM published documentation & model cards specifications
    "nvidia_nim": {
        "deepseek-ai/deepseek-v4-pro-0813": 1_048_576,
        "deepseek-ai/deepseek-v4-flash-0731": 1_048_576,
        "minimaxai/minimax-m3": 262_144,
        "nvidia/nemotron-3-ultra-550b-a55b": 1_048_576,
        "nvidia/nemotron-3-super-120b-a12b": 1_048_576,
        "moonshotai/kimi-k3": 1_048_576,
        "meta/muse-glimmer-30b": 131_072,
        "meta/llama-3.3-70b-instruct": 131_072,
        "meta/llama-3.1-405b-instruct": 131_072,
        "meta/llama-3.1-70b-instruct": 131_072,
        "meta/llama-3.1-8b-instruct": 131_072,
        "qwen/qwen2.5-72b-instruct": 131_072,
        "mistralai/mistral-large-2-instruct": 131_072,
    },
    # Moonshot Kimi Chat Completions
    "kimi": {
        "kimi-k3": 1_048_576,
        "kimi-k1.5": 128_000,
        "moonshot-v1-128k": 128_000,
        "moonshot-v1-32k": 32_768,
        "moonshot-v1-8k": 8_192,
        "moonshot-v1-auto": 128_000,
        "kimi-": 128_000,
        "moonshot-v1-": 128_000,
    },
    # Zhipu AI / Z.ai GLM Coding Plan models
    "zai": {
        "glm-4-plus": 128_000,
        "glm-4-flash": 128_000,
        "glm-4-long": 1_048_576,
        "glm-4-air": 128_000,
        "glm-4-0520": 128_000,
        "codegeex-4": 128_000,
        "glm-4-": 128_000,
        "glm-5-": 128_000,
    },
    # Google AI Studio / Gemini API
    "gemini": {
        "gemini-2.5-pro": 1_048_576,
        "gemini-2.5-flash": 1_048_576,
        "gemini-2.0-flash": 1_048_576,
        "gemini-2.0-flash-lite": 1_048_576,
        "gemini-1.5-pro": 2_097_152,
        "gemini-1.5-flash": 1_048_576,
    },
    # OpenAI Chat / Codex
    "openai": {
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "chatgpt-4o-latest": 128_000,
        "o1": 200_000,
        "o1-mini": 128_000,
        "o1-preview": 128_000,
        "o3-mini": 200_000,
        "gpt-4.5-preview": 128_000,
    },
    # Fallback for when no GROQ_API_KEY is set to read the live catalog, which
    # also states these. Only families whose figure is stable are recorded.
    "groq": {
        "llama-3.3-70b-versatile": 131_072,
        "llama-3.1-8b-instant": 131_072,
        "openai/gpt-oss-120b": 131_072,
        "openai/gpt-oss-20b": 131_072,
    },
    # NaraRoute gateway models
    "nararoute": {
        "deepseek-v4-flash": 1_048_576,
        "deepseek-v4": 1_048_576,
        "glm-5.3-flash-free": 128_000,
        "qwen3.8-27b": 131_072,
        "tencent-hy3-free": 131_072,
    },
}

_KNOWN_MODEL_FAMILIES: tuple[tuple[str, int], ...] = (
    ("deepseek-v4", 1_048_576),
    ("kimi-k3", 1_048_576),
    ("nemotron-3", 1_048_576),
    ("minimax-m3", 262_144),
    ("gemini-1.5-pro", 2_097_152),
    ("gemini-2.5-pro", 1_048_576),
    ("gemini-2.5-flash", 1_048_576),
    ("gemini-2.0-flash", 1_048_576),
    ("gemini-", 1_048_576),
    ("glm-4-long", 1_048_576),
    ("glm-", 128_000),
    ("moonshot-v1-128k", 128_000),
    ("moonshot-v1-32k", 32_768),
    ("moonshot-v1-8k", 8_192),
    ("kimi-k1.5", 128_000),
    ("deepseek-chat", 128_000),
    ("deepseek-reasoner", 128_000),
    ("llama-3", 131_072),
    ("qwen", 131_072),
    ("gpt-4o", 128_000),
    ("o1", 200_000),
    ("o3-mini", 200_000),
)


def curated_providers() -> tuple[str, ...]:
    """Return every provider the curated table knows about."""

    return tuple(CURATED_CONTEXT_WINDOWS)


def curated_context_window(provider: str, model: str) -> int | None:
    """Return the curated window for a model id, or ``None``.

    An exact id wins; otherwise the longest family prefix (a key ending in
    ``-``) that the id starts with applies. If the provider has no entry or no
    match, known model families are checked for gateway routers.
    """

    entries = CURATED_CONTEXT_WINDOWS.get(provider)
    if entries:
        if model in entries:
            return entries[model]
        best_prefix = ""
        best_value: int | None = None
        for prefix, value in entries.items():
            if (
                prefix.endswith("-")
                and model.startswith(prefix)
                and len(prefix) > len(best_prefix)
            ):
                best_prefix = prefix
                best_value = value
        if best_value is not None:
            return best_value
        return None

    normalized = model.lower()
    for family, window in _KNOWN_MODEL_FAMILIES:
        if family in normalized:
            return window

    return None

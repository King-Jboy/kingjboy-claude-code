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
    # DeepSeek documents 128K for both chat and reasoner across revisions.
    "deepseek": {
        "deepseek-chat": 128_000,
        "deepseek-reasoner": 128_000,
    },
    # Google AI Studio's OpenAI-compatible endpoint publishes no length; the
    # 2.x generation is 1M input across the board.
    "gemini": {
        "gemini-2.5-pro": 1_048_576,
        "gemini-2.5-flash": 1_048_576,
        "gemini-2.0-flash": 1_048_576,
    },
    # Fallback for when no GROQ_API_KEY is set to read the live catalog, which
    # also states these. Only families whose figure is stable are recorded.
    "groq": {
        "llama-3.3-70b-versatile": 131_072,
        "llama-3.1-8b-instant": 131_072,
        "openai/gpt-oss-120b": 131_072,
        "openai/gpt-oss-20b": 131_072,
    },
    # Cerebras caps the 8B model far below its siblings; the rest of the
    # family serves 128K.
    "cerebras": {
        "llama3.1-8b": 8_192,
        "llama-3.3-70b": 128_000,
    },
    "mistral": {
        "mistral-large-latest": 128_000,
        "open-mistral-nemo": 128_000,
    },
    # The Codestral endpoint only ever serves this family; revisions keep the
    # same 256K window.
    "mistral_codestral": {
        "codestral-": 256_000,
    },
}


def curated_providers() -> tuple[str, ...]:
    """Return every provider the curated table knows about."""

    return tuple(CURATED_CONTEXT_WINDOWS)


def curated_context_window(provider: str, model: str) -> int | None:
    """Return the curated window for a model id, or ``None``.

    An exact id wins; otherwise the longest family prefix (a key ending in
    ``-``) that the id starts with applies. ``None`` means the table has no
    opinion, which is honest: an unknown window falls back to the default
    rather than a guess.
    """

    entries = CURATED_CONTEXT_WINDOWS.get(provider)
    if not entries:
        return None
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
    return best_value

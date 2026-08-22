"""Matching model ids against the curated context-window table."""

from free_claude_code.config.curated_contexts import curated_context_window


def test_an_exact_id_wins() -> None:
    assert curated_context_window("deepseek", "deepseek-chat") == 128_000


def test_an_unknown_provider_has_no_opinion() -> None:
    assert curated_context_window("lm_studio", "local-model") is None


def test_an_unlisted_model_is_unknown_rather_than_guessed() -> None:
    assert curated_context_window("deepseek", "deepseek- hypothetical") is None


def test_a_family_prefix_covers_revisions() -> None:
    # A family entry exists so the table does not chase every version string.
    from free_claude_code.config import curated_contexts

    original = curated_contexts.CURATED_CONTEXT_WINDOWS
    curated_contexts.CURATED_CONTEXT_WINDOWS = {"kimi": {"kimi-k2-": 128_000}}
    try:
        assert curated_context_window("kimi", "kimi-k2-0905-preview") == 128_000
        assert curated_context_window("kimi", "kimi-k2-thinking") == 128_000
    finally:
        curated_contexts.CURATED_CONTEXT_WINDOWS = original


def test_a_specific_entry_would_beat_a_family_prefix() -> None:
    # Longest prefix wins, so a dated revision with a different window can be
    # added without disturbing the rest of the family.
    table = {"groq": {"model-": 64_000, "model-v2-": 128_000}}
    from free_claude_code.config import curated_contexts

    original = curated_contexts.CURATED_CONTEXT_WINDOWS
    curated_contexts.CURATED_CONTEXT_WINDOWS = table
    try:
        assert curated_context_window("groq", "model-v2-beta") == 128_000
        assert curated_context_window("groq", "model-classic") == 64_000
    finally:
        curated_contexts.CURATED_CONTEXT_WINDOWS = original

from unittest.mock import MagicMock

from free_claude_code.config.api_key_pool import parse_api_key_pool
from free_claude_code.providers.key_pool import ApiKeyPool


def test_key_pool_parser_deduplicates_credentials_in_first_seen_order() -> None:
    assert parse_api_key_pool("first, second, first, third, second") == (
        "first",
        "second",
        "third",
    )


def test_duplicate_pool_entries_do_not_bypass_a_per_key_failure_hold() -> None:
    pool = ApiKeyPool(("same-key", "same-key"), rate_limit=100, rate_window=60.0)

    for _ in range(3):
        pool.mark_failed("same-key")

    assert pool.get_next_key() is None


def test_key_pool_notifies_capacity_changes_as_credentials_cool_down_or_retire() -> (
    None
):
    recorded_limits: list[int] = []
    pool = ApiKeyPool(
        ("key-1", "key-2", "key-3"),
        rate_limit=20,
        rate_window=60.0,
        on_capacity_change=recorded_limits.append,
    )

    assert pool.active_key_count() == 3
    assert pool.active_rate_limit() == 60

    pool.mark_rate_limited("key-1")
    assert pool.active_key_count() == 2
    assert recorded_limits[-1] == 40

    for _ in range(3):
        pool.mark_failed("key-2")
    assert pool.active_key_count() == 1
    assert recorded_limits[-1] == 20

    pool.mark_rate_limited("key-3")
    assert pool.active_key_count() == 0
    assert recorded_limits[-1] == 20


def test_a_replacement_pool_resumes_key_health_and_recent_sends() -> None:
    """A config reload must not revive retired keys or empty per-key windows."""
    previous = ApiKeyPool(("key-a", "key-b"), rate_limit=1, rate_window=60.0)
    for _ in range(3):
        previous.mark_failed("key-a")
    assert previous.get_next_key() == "key-b"

    replacement = ApiKeyPool(
        ("key-a", "key-b", "key-c"), rate_limit=1, rate_window=60.0
    )
    replacement.import_state(previous.export_state())

    # key-a stays retired and key-b's one send this window still counts, so
    # only the key the old generation never touched is available.
    assert replacement.get_next_key() == "key-c"
    assert replacement.get_next_key() is None


def test_runtime_snapshot_reaches_a_lazily_built_provider() -> None:
    from free_claude_code.providers.runtime.runtime import ProviderRuntime

    old_pool = ApiKeyPool(("key-a", "key-b"), rate_limit=5, rate_window=60.0)
    for _ in range(3):
        old_pool.mark_failed("key-a")
    old_provider = MagicMock()
    old_provider.api_key_pool = old_pool
    previous = ProviderRuntime(MagicMock(), {"open_router": old_provider})

    new_pool = ApiKeyPool(("key-a", "key-b"), rate_limit=5, rate_window=60.0)
    new_provider = MagicMock()
    new_provider.api_key_pool = new_pool
    current = ProviderRuntime(
        MagicMock(), provider_constructor=lambda _id, _settings: new_provider
    )
    current.inherit_credential_state(previous)
    current.resolve_provider("open_router")

    assert new_pool.get_next_key() == "key-b"
    assert new_pool.get_next_key(exclude="key-b") is None

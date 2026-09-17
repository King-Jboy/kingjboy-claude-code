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

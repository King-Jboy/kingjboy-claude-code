"""Property tests for parsers whose input space is far wider than any example.

Each of these was chosen because a wrong answer is silent: a mis-parsed
cooldown idles a healthy key or hammers a throttled one, and a mis-quoted env
value corrupts config on the next Admin save without raising anything.
"""

import json

from hypothesis import given
from hypothesis import strategies as st

from free_claude_code.config.admin.persistence import quote_env_value
from free_claude_code.config.admin.sources import dotenv_values_from_text
from free_claude_code.config.api_keys import parse_api_key_list

# dotenv treats a leading/trailing space and a bare newline as syntax, not
# data, so those are outside the round-trip contract.
ENV_VALUES = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=0, max_size=120
).filter(lambda value: value == value.strip())

KEY_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1, max_size=40
).filter(lambda value: value.strip() != "")


@given(value=ENV_VALUES)
def test_any_quoted_value_survives_a_dotenv_round_trip(value: str) -> None:
    # The Admin save rewrites the whole managed file, so a value that does not
    # survive quoting is silently corrupted rather than rejected.
    rendered = f"SOME_KEY={quote_env_value(value)}"

    assert dotenv_values_from_text(rendered)["SOME_KEY"] == value


@given(keys=st.lists(KEY_TEXT, min_size=0, max_size=12))
def test_pool_parsing_preserves_order_and_drops_duplicates(keys: list[str]) -> None:
    parsed = parse_api_key_list(json.dumps(keys), env_name="POOL")

    stripped = [key.strip() for key in keys if key.strip()]
    expected: list[str] = []
    for key in stripped:
        if key not in expected:
            expected.append(key)
    assert list(parsed) == expected


@given(keys=st.lists(KEY_TEXT, min_size=1, max_size=8))
def test_pool_parsing_is_idempotent(keys: list[str]) -> None:
    once = parse_api_key_list(json.dumps(keys), env_name="POOL")

    assert parse_api_key_list(json.dumps(list(once)), env_name="POOL") == once


@given(raw=st.text(max_size=40))
def test_pool_parsing_either_returns_keys_or_names_the_variable(raw: str) -> None:
    # Malformed input must fail loudly and say which variable to fix, never
    # silently yield a pool that is smaller than the operator configured.
    try:
        parsed = parse_api_key_list(raw, env_name="NVIDIA_NIM_API_KEYS")
    except ValueError as error:
        assert "NVIDIA_NIM_API_KEYS" in str(error)
    else:
        assert all(key.strip() == key and key for key in parsed)

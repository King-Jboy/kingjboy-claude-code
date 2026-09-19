from free_claude_code.core.openai_responses.ids import (
    tool_item_id_for_kind,
    tool_item_id_prefix,
)


def test_tool_item_id_prefix() -> None:
    assert tool_item_id_prefix("function") == "fc_"
    assert tool_item_id_prefix("custom") == "ctc_"


def test_tool_item_id_for_kind_retags_opposite_prefix() -> None:
    assert tool_item_id_for_kind("fc_12345", kind="custom") == "ctc_12345"
    assert tool_item_id_for_kind("ctc_12345", kind="function") == "fc_12345"


def test_tool_item_id_for_kind_preserves_matching_and_unrelated_prefixes() -> None:
    assert tool_item_id_for_kind("fc_12345", kind="function") == "fc_12345"
    assert tool_item_id_for_kind("ctc_12345", kind="custom") == "ctc_12345"
    assert tool_item_id_for_kind("call_12345", kind="custom") == "call_12345"
    assert tool_item_id_for_kind("custom_12345", kind="function") == "custom_12345"

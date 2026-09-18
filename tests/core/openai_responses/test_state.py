from pathlib import Path

import pytest

from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesStore,
)


def _completed_response(response_id: str) -> dict[str, object]:
    return {
        "id": response_id,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Prior answer"}],
            }
        ],
    }


def test_continuation_replays_prior_input_without_carrying_instructions() -> None:
    store = ResponsesStore()
    first = OpenAIResponsesRequest(
        model="nvidia_nim/test-model",
        input="First question",
        instructions="Keep the project context.",
    )
    store.record(first, _completed_response("resp_prior"))

    resolved = store.resolve(
        OpenAIResponsesRequest(
            model="nvidia_nim/test-model",
            input="Continue from there.",
            previous_response_id="resp_prior",
        )
    )

    assert resolved.input == [
        "First question",
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Prior answer"}],
        },
        "Continue from there.",
    ]
    assert resolved.instructions is None


def test_continuation_uses_its_new_instructions() -> None:
    store = ResponsesStore()
    store.record(
        OpenAIResponsesRequest(
            model="nvidia_nim/test-model",
            input="First question",
            instructions="Old instructions.",
        ),
        _completed_response("resp_prior"),
    )

    resolved = store.resolve(
        OpenAIResponsesRequest(
            model="nvidia_nim/test-model",
            input="Continue from there.",
            instructions="New instructions.",
            previous_response_id="resp_prior",
        )
    )

    assert resolved.instructions == "New instructions."


def test_unknown_continuation_id_is_rejected() -> None:
    with pytest.raises(ResponsesConversionError, match="previous_response_id"):
        ResponsesStore().resolve(
            OpenAIResponsesRequest(
                model="nvidia_nim/test-model",
                input="Continue",
                previous_response_id="resp_missing",
            )
        )


def test_store_false_does_not_create_a_continuation() -> None:
    store = ResponsesStore()
    request = OpenAIResponsesRequest(
        model="nvidia_nim/test-model", input="Do not retain this.", store=False
    )
    store.record(request, _completed_response("resp_private"))

    with pytest.raises(ResponsesConversionError, match="previous_response_id"):
        store.resolve(
            OpenAIResponsesRequest(
                model="nvidia_nim/test-model",
                input="Continue",
                previous_response_id="resp_private",
            )
        )


def test_completed_responses_survive_a_store_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "responses.json"
    request = OpenAIResponsesRequest(
        model="nvidia_nim/test-model", input="Persist this."
    )
    ResponsesStore(state_path).record(request, _completed_response("resp_saved"))

    resolved = ResponsesStore(state_path).resolve(
        OpenAIResponsesRequest(
            model="nvidia_nim/test-model",
            input="Continue",
            previous_response_id="resp_saved",
        )
    )

    assert resolved.input[0] == "Persist this."

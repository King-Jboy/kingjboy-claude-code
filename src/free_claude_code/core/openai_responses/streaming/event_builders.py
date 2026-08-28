"""OpenAI Responses SSE event builders."""

from typing import Any

from ..events import format_response_sse_event


def response_created(response: dict[str, Any], *, sequence_number: int = 0) -> str:
    return format_response_sse_event(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": sequence_number,
            "response": response,
        },
    )


def response_completed(response: dict[str, Any], *, sequence_number: int = 0) -> str:
    return format_response_sse_event(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": response,
        },
    )


def response_incomplete(response: dict[str, Any], *, sequence_number: int = 0) -> str:
    return format_response_sse_event(
        "response.incomplete",
        {
            "type": "response.incomplete",
            "sequence_number": sequence_number,
            "response": response,
        },
    )


def response_failed(response: dict[str, Any], *, sequence_number: int = 0) -> str:
    return format_response_sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "sequence_number": sequence_number,
            "response": response,
        },
    )


def output_item_added(
    output_index: int, item: dict[str, Any], *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": item,
        },
    )


def output_item_done(
    output_index: int, item: dict[str, Any], *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "sequence_number": sequence_number,
            "output_index": output_index,
            "item": item,
        },
    )


def content_part_added(
    item_id: str, output_index: int, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )


def output_text_delta(
    item_id: str, output_index: int, text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "delta": text,
        },
    )


def output_text_done(
    item_id: str, output_index: int, text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": text,
        },
    )


def content_part_done(
    item_id: str, output_index: int, text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
    )


def reasoning_text_delta(
    item_id: str, output_index: int, text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.reasoning_text.delta",
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "delta": text,
        },
    )


def reasoning_text_done(
    item_id: str, output_index: int, text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.reasoning_text.done",
        {
            "type": "response.reasoning_text.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": text,
        },
    )


def function_call_arguments_delta(
    item_id: str, output_index: int, arguments: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.function_call_arguments.delta",
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "delta": arguments,
        },
    )


def function_call_arguments_done(
    item_id: str, output_index: int, arguments: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.function_call_arguments.done",
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "arguments": arguments,
        },
    )


def custom_tool_call_input_delta(
    item_id: str, output_index: int, input_text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.custom_tool_call_input.delta",
        {
            "type": "response.custom_tool_call_input.delta",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "delta": input_text,
        },
    )


def custom_tool_call_input_done(
    item_id: str, output_index: int, input_text: str, *, sequence_number: int = 0
) -> str:
    return format_response_sse_event(
        "response.custom_tool_call_input.done",
        {
            "type": "response.custom_tool_call_input.done",
            "sequence_number": sequence_number,
            "item_id": item_id,
            "output_index": output_index,
            "input": input_text,
        },
    )

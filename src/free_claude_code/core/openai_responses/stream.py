"""Translate Anthropic SSE streams into OpenAI Responses SSE streams."""

import asyncio
import sys
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from typing import Any

from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import ExecutionFailure, find_execution_failure
from free_claude_code.core.trace import close_stream_input

from .anthropic_sse import iter_sse_events
from .models import OpenAIResponsesRequest
from .streaming import ResponsesStreamAssembler

PostStartTerminalFailureObserver = Callable[[BaseException], None]
CompletedResponseObserver = Callable[[Mapping[str, Any]], None]


async def iter_responses_sse_from_anthropic(
    chunks: AsyncIterable[Any],
    request: OpenAIResponsesRequest,
    *,
    on_post_start_terminal_failure: PostStartTerminalFailureObserver | None = None,
    on_completed_response: CompletedResponseObserver | None = None,
) -> AsyncIterator[str]:
    """Yield Responses SSE events translated from an Anthropic SSE stream."""
    assembler = ResponsesStreamAssembler(request)
    emitted_any_chunk = False
    completed_observed = False
    events = iter_sse_events(chunks)
    try:
        async for event in events:
            output_chunks = assembler.process_anthropic_event(event)
            if assembler.terminal and not completed_observed:
                _observe_completed_response(
                    on_completed_response, assembler.final_response
                )
                completed_observed = True
            for chunk in output_chunks:
                yield chunk
                emitted_any_chunk = True
            if assembler.terminal:
                return
        output_chunks = assembler.finish_if_needed()
        if assembler.terminal and not completed_observed:
            _observe_completed_response(on_completed_response, assembler.final_response)
        for chunk in output_chunks:
            yield chunk
            emitted_any_chunk = True
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except ExecutionFailure as exc:
        if not emitted_any_chunk:
            raise
        _observe_post_start_terminal_failure(on_post_start_terminal_failure, exc)
        for chunk in assembler.fail_execution(exc):
            yield chunk
    except BaseExceptionGroup as exc:
        # A cancellation bundled with another error is still cancellation
        # control flow.  Turning it into a terminal SSE error leaves work
        # running after the HTTP client has disconnected.
        if exc.subgroup(asyncio.CancelledError) is not None:
            raise
        if not emitted_any_chunk:
            raise
        failure = find_execution_failure(exc)
        if failure is not None:
            _observe_post_start_terminal_failure(
                on_post_start_terminal_failure, failure
            )
            for chunk in assembler.fail_execution(failure):
                yield chunk
        else:
            _observe_post_start_terminal_failure(on_post_start_terminal_failure, exc)
            for chunk in assembler.fail_response(_unexpected_error_data(exc)):
                yield chunk
    except Exception as exc:
        if not emitted_any_chunk:
            raise
        _observe_post_start_terminal_failure(on_post_start_terminal_failure, exc)
        for chunk in assembler.fail_response(_unexpected_error_data(exc)):
            yield chunk
    finally:
        await close_stream_input(
            events,
            owner="openai_responses.stream",
            source="core",
            preserved_error=sys.exception(),
        )


def _observe_post_start_terminal_failure(
    observer: PostStartTerminalFailureObserver | None,
    exc: BaseException,
) -> None:
    if observer is not None:
        observer(exc)


def _observe_completed_response(
    observer: CompletedResponseObserver | None,
    response: Mapping[str, Any] | None,
) -> None:
    if (
        observer is not None
        and response is not None
        and response.get("status") == "completed"
    ):
        observer(response)


def _unexpected_error_data(exc: BaseException) -> dict[str, dict[str, str]]:
    return {
        "error": {
            "type": "api_error",
            "message": safe_exception_message(exc),
        }
    }

from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from free_claude_code.messaging.platforms.ports import OutboundMessenger
from free_claude_code.messaging.transcript import RenderCtx
from free_claude_code.messaging.transcript.buffer import TranscriptBuffer
from free_claude_code.messaging.ui_updates import ThrottledTranscriptEditor


class _StaticTranscript:
    def render(self, _ctx: RenderCtx, *, limit_chars: int, status: str | None) -> str:
        del limit_chars
        return status or "Working"


class _PendingOutbound:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None] | None] = []
        self.failure_callbacks: list[Callable[[Exception], None] | None] = []
        self.background_tasks: list[Awaitable[Any]] = []

    async def queue_edit_message(
        self,
        _chat_id: str,
        _message_id: str,
        _text: str,
        *,
        parse_mode: str | None = None,
        on_delivered: Callable[[], None] | None = None,
        on_delivery_failed: Callable[[Exception], None] | None = None,
    ) -> None:
        del parse_mode
        self.callbacks.append(on_delivered)
        self.failure_callbacks.append(on_delivery_failed)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        self.background_tasks.append(task)


def _render_context() -> RenderCtx:
    return RenderCtx(
        bold=lambda text: text,
        code_inline=lambda text: text,
        escape_code=lambda text: text,
        escape_text=lambda text: text,
        render_markdown=lambda text: text,
    )


@pytest.mark.asyncio
async def test_editor_retries_terminal_display_until_outbox_confirms_delivery() -> None:
    outbound = _PendingOutbound()
    editor = ThrottledTranscriptEditor(
        outbound=cast(OutboundMessenger, outbound),
        parse_mode=None,
        get_limit_chars=lambda: 100,
        transcript=cast(TranscriptBuffer, _StaticTranscript()),
        render_ctx=_render_context(),
        node_id="node",
        chat_id="chat",
        status_msg_id="status",
        debug_platform_edits=False,
    )

    await editor.update("Complete", force=True)
    await editor.update("Complete", force=True)

    assert len(outbound.callbacks) == 2
    assert outbound.callbacks[-1] is not None
    outbound.callbacks[-1]()

    await editor.update("Complete", force=True)

    assert len(outbound.callbacks) == 2


@pytest.mark.asyncio
async def test_editor_retries_a_failed_terminal_edit_once_after_outbox_recovery() -> (
    None
):
    outbound = _PendingOutbound()
    editor = ThrottledTranscriptEditor(
        outbound=cast(OutboundMessenger, outbound),
        parse_mode=None,
        get_limit_chars=lambda: 100,
        transcript=cast(TranscriptBuffer, _StaticTranscript()),
        render_ctx=_render_context(),
        node_id="node",
        chat_id="chat",
        status_msg_id="status",
        debug_platform_edits=False,
    )

    await editor.update("Complete", force=True)

    assert outbound.failure_callbacks[-1] is not None
    outbound.failure_callbacks[-1](RuntimeError("connection reset"))
    assert len(outbound.background_tasks) == 1

    retry = outbound.background_tasks.pop()
    await retry
    assert len(outbound.callbacks) == 2
    assert outbound.callbacks[-1] is not None
    outbound.callbacks[-1]()

    await editor.update("Complete", force=True)

    assert len(outbound.callbacks) == 2

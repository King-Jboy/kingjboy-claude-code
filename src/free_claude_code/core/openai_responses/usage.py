"""Usage helpers for OpenAI Responses payloads."""

import threading
from typing import Protocol

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


class _TokenEncoder(Protocol):
    def encode(
        self, text: str, *, disallowed_special: tuple[str, ...]
    ) -> list[int]: ...


_encoder: _TokenEncoder | None = None
_encoder_loaded = False
_encoder_lock = threading.Lock()


def _load_encoder() -> _TokenEncoder | None:
    from free_claude_code.core.anthropic.tokens import get_tiktoken_encoder

    return get_tiktoken_encoder()


def _get_encoder() -> _TokenEncoder | None:
    global _encoder, _encoder_loaded
    if _encoder_loaded:
        return _encoder
    with _encoder_lock:
        if not _encoder_loaded:
            _encoder = _load_encoder()
            _encoder_loaded = True
        return _encoder


def estimate_text_tokens(text: str) -> int:
    """Return a best-effort token estimate for Responses usage details."""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return max(1, len(text) // 4)

"""Tests for NVIDIA NIM voice transcription and key pool isolation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.providers.key_pool import ApiKeyPool
from free_claude_code.providers.nvidia_nim.voice import NvidiaNimTranscriber


class _RivaCredentialError(Exception):
    def __init__(self, status: str) -> None:
        self._status = status

    def code(self) -> str:
        return self._status


@pytest.mark.asyncio
async def test_voice_failure_does_not_disable_shared_chat_key_pool(tmp_path: Path) -> None:
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"audio data")

    shared_chat_pool = ApiKeyPool(
        ("key1", "key2"),
        rate_limit=40,
        rate_window=60.0,
    )

    async def get_pool():
        return shared_chat_pool

    transcriber = NvidiaNimTranscriber(
        model="openai/whisper-large-v3",
        key_pool_provider=get_pool,
    )

    first_auth = MagicMock()
    second_auth = MagicMock()
    first_service = MagicMock()
    first_service.offline_recognize.side_effect = _RivaCredentialError("UNAUTHENTICATED")
    second_response = SimpleNamespace(
        results=[
            SimpleNamespace(alternatives=[SimpleNamespace(transcript="success")])
        ]
    )
    second_service = MagicMock()
    second_service.offline_recognize.return_value = second_response
    client = SimpleNamespace(
        Auth=MagicMock(side_effect=[first_auth, second_auth]),
        ASRService=MagicMock(side_effect=[first_service, second_service]),
        RecognitionConfig=MagicMock(return_value=object()),
    )
    riva = SimpleNamespace(__path__=[], client=client)

    with patch.dict("sys.modules", {"riva": riva, "riva.client": client}):
        result = await transcriber.transcribe(wav)

    assert result == "success"

    # Shared chat key pool must NOT have key1 marked as failed
    key1_state = shared_chat_pool._by_key["key1"]
    assert key1_state.failed is False
    assert key1_state.consecutive_failures == 0
    assert key1_state.unavailable_until == 0.0

    # Key1 should still be available in the shared chat key pool
    assert shared_chat_pool.get_next_key() == "key1"

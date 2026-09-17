"""Durable continuation state for the supported Responses API subset."""

import copy
import json
import os
import tempfile
import threading
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest


class ResponsesStore:
    """Keep completed Responses turns so ``previous_response_id`` is meaningful."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._loaded = path is None

    def resolve(self, request: OpenAIResponsesRequest) -> OpenAIResponsesRequest:
        """Expand a continuation request, or reject an unknown response id."""
        response_id = request.previous_response_id
        if response_id is None:
            return request
        with self._lock:
            self._load_locked()
            record = self._records.get(response_id)
            if record is None:
                raise ResponsesConversionError(
                    "previous_response_id does not reference a stored completed response."
                )
            history = copy.deepcopy(record["input"])
            history.extend(copy.deepcopy(record["output"]))
            history.extend(_input_items(request.input))
            instructions = request.instructions
            if instructions is None:
                stored_instructions = record.get("instructions")
                if isinstance(stored_instructions, str):
                    instructions = stored_instructions
        return request.model_copy(
            update={
                "input": history,
                "instructions": instructions,
                "previous_response_id": None,
            }
        )

    def record(
        self,
        request: OpenAIResponsesRequest,
        response: Mapping[str, Any],
    ) -> None:
        """Persist one completed response unless the caller explicitly opted out."""
        if request.store is False or response.get("status") != "completed":
            return
        response_id = response.get("id")
        output = response.get("output")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("Completed Responses output must include a response id.")
        if not isinstance(output, list):
            raise ValueError("Completed Responses output must include an output list.")
        record = {
            "input": _input_items(request.input),
            "output": copy.deepcopy(output),
            "instructions": request.instructions,
        }
        # Reject non-JSON state before mutating the in-memory continuation chain.
        json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._load_locked()
            self._records[response_id] = record
            self._persist_locked()

    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        assert self._path is not None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except OSError, json.JSONDecodeError:
            return
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, dict):
            return
        self._records = {
            response_id: record
            for response_id, record in records.items()
            if isinstance(response_id, str) and _valid_record(record)
        }

    def _persist_locked(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"records": self._records}, ensure_ascii=False)
        fd, temporary_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            with suppress(OSError):
                temporary.chmod(0o600)
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)


def _input_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    return [copy.deepcopy(value)]


def _valid_record(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("input"), list)
        and isinstance(value.get("output"), list)
        and (
            value.get("instructions") is None
            or isinstance(value.get("instructions"), str)
        )
    )

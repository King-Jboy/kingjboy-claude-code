"""Provider connection provenance and precise historical request corrections."""

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.history_replay import (
    HistoryProtocol,
    HistoryReplayError,
    ReplayOrigin,
    decode_replay,
    is_replay,
    readable_reasoning,
    resolve_messages_replay,
)
from free_claude_code.core.json_types import JsonValue

from .endpoint_types import HttpEndpoint

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def validate_history(body: Mapping[str, Any]) -> None:
    """Reject malformed persisted carriers before credentials or inference."""
    try:
        for protocol in ("responses", "messages", "chat"):
            for _, item in _reasoning_records(body, protocol):
                for key in ("encrypted_content", "signature", "data"):
                    value = item.get(key)
                    if isinstance(value, str) and is_replay(value):
                        decode_replay(value)
    except HistoryReplayError as error:
        raise InvalidRequestError(str(error)) from error


def _iter_block_replay_values(block: Any) -> Iterator[str]:
    if isinstance(block, Mapping):
        for key in ("encrypted_content", "signature", "data"):
            value = block.get(key)
            if isinstance(value, str) and is_replay(value):
                yield value
        return
    extra = getattr(block, "model_extra", None)
    for key in ("encrypted_content", "signature", "data"):
        value = getattr(block, key, None)
        if value is None and isinstance(extra, Mapping):
            value = extra.get(key)
        if isinstance(value, str) and is_replay(value):
            yield value


def normalize_messages_history(request: MessagesRequest) -> MessagesRequest:
    """Resolve persisted associations before any lossy protocol conversion."""
    extra = getattr(request, "model_extra", None)
    if isinstance(extra, Mapping) and "input" in extra:
        validate_history({"input": extra["input"]})

    messages: list[Message] | None = None
    try:
        for index, message in enumerate(request.messages):
            if message.role != "assistant":
                if messages is not None:
                    messages.append(message)
                continue

            msg_extra = getattr(message, "model_extra", None)
            if isinstance(msg_extra, Mapping):
                details = msg_extra.get("reasoning_details")
                if isinstance(details, list):
                    for detail in details:
                        if isinstance(detail, Mapping):
                            for value in _iter_block_replay_values(detail):
                                decode_replay(value)

            if isinstance(message.content, str):
                if messages is not None:
                    messages.append(message)
                continue

            has_associated = False
            for block in message.content:
                block_type = (
                    block.get("type")
                    if isinstance(block, Mapping)
                    else getattr(block, "type", None)
                )
                if block_type not in {"thinking", "redacted_thinking"}:
                    continue
                for value in _iter_block_replay_values(block):
                    decode_replay(value)
                    if value.startswith("fcc:history:v2:"):
                        has_associated = True

            if not has_associated:
                if messages is not None:
                    messages.append(message)
                continue

            if messages is None:
                messages = list(request.messages[:index])
            body = message.model_dump(mode="json")
            body["content"] = resolve_messages_replay(
                cast(list[JsonValue], body["content"])
            )
            messages.append(Message.model_validate(body))
    except HistoryReplayError as error:
        raise InvalidRequestError(str(error)) from error

    if messages is None:
        return request
    return request.model_copy(update={"messages": messages})


def replay_origin(
    provider: str,
    protocol: HistoryProtocol,
    model: str,
    *,
    client: AsyncOpenAI | None = None,
    endpoint: HttpEndpoint | None = None,
) -> ReplayOrigin:
    """Identify the actual connection without storing its credentials."""
    base_url = (
        endpoint.base_url
        if endpoint is not None
        else str(client.base_url)
        if client is not None
        else ""
    )
    headers: Mapping[str, object] = (
        endpoint.headers
        if endpoint is not None
        else client.default_headers
        if client is not None
        else {}
    )
    normalized_headers = {
        key.lower(): value for key, value in headers.items() if isinstance(value, str)
    }
    key = (
        endpoint.api_key
        if endpoint is not None
        else client.api_key
        if client is not None
        else None
    )
    account = endpoint.account_id if endpoint is not None else None
    credential = (
        "account:" + account
        if account
        else "key:" + key
        if isinstance(key, str) and key
        else "auth:"
        + normalized_headers.get(
            "authorization", normalized_headers.get("x-api-key", "")
        )
    )
    parsed = urlsplit(base_url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host += f":{parsed.port}"
    address = urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    return ReplayOrigin(
        provider,
        protocol,
        address,
        hashlib.sha256(credential.encode()).hexdigest(),
        model,
    )


_INVALID_ID_RE = re.compile(
    r"invalid.+(?:input|messages).+\.id.+(?:letters|characters|ID)",
    re.I,
)
_INVALID_NATIVE_RE = re.compile(
    r"invalid signature in thinking block|referenced reasoning item .+ (?:not found|expired)",
    re.I,
)
_VALID_ID_FULLMATCH_RE = re.compile(r"[A-Za-z0-9_-]+")
_LOCATION_PATH_RE = re.compile(
    r"(?:input|messages)(?:\[\d+\]|\.\d+)(?:(?:\.content|\.reasoning_details)(?:\[\d+\]|\.\d+))?"
)
_PATH_TOKEN_RE = re.compile(r"[a-z_]+|\d+")
_NAMED_ID_RE = re.compile(
    r"\b(?:for item|reasoning item|and item)\s+"
    r"(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z0-9_-]+))(?=\s|[.,:]|$)",
    re.I,
)


def history_retry_body(
    error: Exception, body: Mapping[str, Any], protocol: HistoryProtocol
) -> dict[str, Any] | None:
    """Repair one explicitly rejected historical record on an independent copy."""
    detail = extract_upstream_error_detail(error)
    if detail.status_code not in {None, 200, 400, 422}:
        return None
    try:
        payload = json.loads(detail.body_text or "")
    except ValueError:
        payload = {
            "message": detail.exception_text or str(error),
            "code": getattr(error, "code", None),
        }
    candidates = list(_reasoning_records(body, protocol))
    for record in _error_records(payload):
        message = str(record.get("message", ""))
        code = record.get("code", record.get("type"))
        invalid_id = bool(_INVALID_ID_RE.search(message))
        invalid_native = code in {
            "invalid_encrypted_content",
            "invalid_signature",
        } or bool(_INVALID_NATIVE_RE.search(message))
        if not (invalid_id or invalid_native):
            continue
        matched = _rejected_record(record, candidates)
        if matched is None:
            continue
        path, original = matched
        result = deepcopy(dict(body))
        # A one-object Responses `input` is not a list, so `input[0]` is that object.
        if (
            path[:2] == ("input", 0)
            and len(path) == 2
            and isinstance(result.get("input"), Mapping)
            and not isinstance(result.get("input"), list)
        ):
            path = ("input",)
        parent: Any = result
        for key in path[:-1]:
            parent = parent[key]
        if invalid_id:
            value = original.get("id")
            if not isinstance(value, str) or _VALID_ID_FULLMATCH_RE.fullmatch(value):
                continue
            corrected = "rs_" + hashlib.sha256(value.encode()).hexdigest()[:24]
            parent[path[-1]]["id"] = corrected
            for item in result.get("input", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "item_reference"
                    and item.get("id") == value
                ):
                    item["id"] = corrected
        else:
            context = "\n\n".join(
                text
                for text, _ in readable_reasoning(
                    original
                    if protocol != "chat"
                    else {"reasoning_details": [original]}
                )
            )
            if protocol == "responses" and context:
                parent[path[-1]] = {"role": "assistant", "content": context}
            elif protocol == "messages" and context:
                parent[path[-1]] = {"type": "text", "text": context}
            else:
                del parent[path[-1]]
                if protocol == "chat":
                    assistant = result["messages"][path[1]]
                    if context:
                        existing_content = assistant.get("content")
                        if isinstance(existing_content, list):
                            existing_content.append({"type": "text", "text": context})
                        elif (
                            isinstance(existing_content, str)
                            and existing_content.strip()
                        ):
                            assistant["content"] = f"{existing_content}\n\n{context}"
                        else:
                            assistant["content"] = context
                    if not parent:
                        assistant.pop("reasoning_details", None)
        return result
    return None


def _reasoning_records(
    body: Mapping[str, Any], protocol: HistoryProtocol
) -> Iterator[tuple[tuple[str | int, ...], Mapping[str, Any]]]:
    if protocol == "responses":
        items = body.get("input")
        if isinstance(items, Mapping):
            items = [items]
        for index, item in enumerate(items if isinstance(items, list) else []):
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                yield ("input", index), item
        return
    for index, message in enumerate(body.get("messages", [])):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        field = "content" if protocol == "messages" else "reasoning_details"
        blocks = message.get(field)
        if not isinstance(blocks, list):
            continue
        for block_index, block in enumerate(blocks):
            if isinstance(block, Mapping) and (
                protocol == "chat"
                or block.get("type") in {"thinking", "redacted_thinking"}
            ):
                yield ("messages", index, field, block_index), block


def _error_records(payload: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("message"), str):
            yield payload
        for key in ("error", "errors", "metadata", "raw", "previous_errors"):
            nested = payload.get(key)
            if key == "raw" and isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except ValueError:
                    continue
            yield from _error_records(nested)
    elif isinstance(payload, list):
        for item in payload:
            yield from _error_records(item)


def _rejected_record(
    error: Mapping[str, Any],
    candidates: list[tuple[tuple[str | int, ...], Mapping[str, Any]]],
) -> tuple[tuple[str | int, ...], Mapping[str, Any]] | None:
    location = str(error.get("param") or error.get("loc") or error.get("message") or "")
    match = _LOCATION_PATH_RE.search(location)
    if match:
        path = tuple(
            int(part) if part.isdigit() else part
            for part in _PATH_TOKEN_RE.findall(match[0])
        )
        found = next((entry for entry in candidates if entry[0] == path), None)
        if found is not None:
            return found
        # `messages[0]` names the message, not the thinking block inside it.
        if not any(
            len(entry[0]) > len(path) and entry[0][: len(path)] == path
            for entry in candidates
        ):
            return None
    message = str(error.get("message", ""))
    named_ids = {
        value for match in _NAMED_ID_RE.findall(message) for value in match if value
    }
    identified = [
        entry
        for entry in candidates
        if isinstance(entry[1].get("id"), str)
        and (
            entry[1]["id"] in named_ids
            or f"'{entry[1]['id']}'" in message
            or f'"{entry[1]["id"]}"' in message
        )
    ]
    if len(identified) == 1 and (
        not named_ids or named_ids == {identified[0][1]["id"]}
    ):
        return identified[0]
    if named_ids or identified:
        return None
    opaque = [
        entry
        for entry in candidates
        if any(entry[1].get(key) for key in ("encrypted_content", "signature", "data"))
    ]
    if len(opaque) == 1:
        return opaque[0]
    return None

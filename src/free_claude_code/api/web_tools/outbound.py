"""Outbound HTTP for web_search / web_fetch (client, body caps, logging)."""

import asyncio
import codecs
import socket
from collections.abc import AsyncIterator
from email.message import Message
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from loguru import logger

from . import constants
from .constants import (
    _MAX_FETCH_CHARS,
    _MAX_SEARCH_RESULTS,
    _REDIRECT_RESPONSE_BODY_CAP_BYTES,
    _REQUEST_TIMEOUT_S,
    _WEB_FETCH_REDIRECT_STATUSES,
    _WEB_TOOL_HTTP_HEADERS,
)
from .egress import (
    WebFetchEgressPolicy,
    WebFetchEgressViolation,
    get_validated_stream_addrinfos_for_egress,
)
from .parsers import HTMLTextParser, SearchResultParser


def _safe_public_host_for_logs(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[:253]


def _log_web_tool_failure(
    tool_name: str,
    error: BaseException,
    *,
    fetch_url: str | None = None,
) -> None:
    exc_type = type(error).__name__
    if isinstance(error, WebFetchEgressViolation):
        host = _safe_public_host_for_logs(fetch_url) if fetch_url else ""
        logger.warning(
            "web_tool_egress_rejected tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            host,
        )
        return
    if tool_name == "web_fetch" and fetch_url:
        logger.warning(
            "web_tool_failure tool={} exc_type={} host={!r}",
            tool_name,
            exc_type,
            _safe_public_host_for_logs(fetch_url),
        )
    else:
        logger.warning("web_tool_failure tool={} exc_type={}", tool_name, exc_type)


def _web_tool_client_error_summary(
    tool_name: str,
    error: BaseException,
    *,
    verbose: bool,
) -> str:
    if verbose:
        return f"{tool_name} failed: {type(error).__name__}"
    return "Web tool request failed."


async def _iter_response_body_under_cap(
    response: httpx.Response, max_bytes: int
) -> AsyncIterator[bytes]:
    if max_bytes <= 0:
        return
    received = 0
    async for chunk in response.aiter_bytes(chunk_size=65_536):
        if received >= max_bytes:
            break
        remaining = max_bytes - received
        if len(chunk) <= remaining:
            received += len(chunk)
            yield chunk
            if received >= max_bytes:
                break
        else:
            yield chunk[:remaining]
            break


async def _drain_response_body_capped(response: httpx.Response, max_bytes: int) -> None:
    async for _ in _iter_response_body_under_cap(response, max_bytes):
        pass


async def _read_response_body_capped(response: httpx.Response, max_bytes: int) -> bytes:
    return b"".join(
        [piece async for piece in _iter_response_body_under_cap(response, max_bytes)]
    )


def _charset_from_content_type(content_type: str | None) -> str | None:
    """Extract the charset parameter from a Content-Type header without body sniffing."""
    if not content_type:
        return None
    msg = Message()
    msg["content-type"] = content_type
    charset = msg.get_param("charset", header="content-type")
    if isinstance(charset, tuple):
        charset = charset[2]
    if isinstance(charset, str):
        stripped = charset.strip()
        return stripped or None
    return None


def _pinned_fetch_targets(
    url: str, addrinfos: list[tuple]
) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    """Build DNS-pinned httpx request targets from validated ``getaddrinfo`` rows.

    Connects directly to each validated IP literal while preserving the original
    ``Host`` header and TLS SNI hostname so redirects and connections cannot
    re-resolve to a different address.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "http").lower()
    original_host = parsed.hostname or ""
    default_port = 443 if scheme == "https" else 80
    parsed_port = parsed.port

    formatted_host = f"[{original_host}]" if ":" in original_host else original_host
    if parsed_port is not None and parsed_port != default_port:
        host_header = f"{formatted_host}:{parsed_port}"
    else:
        host_header = formatted_host

    headers = {**_WEB_TOOL_HTTP_HEADERS, "Host": host_header}
    extensions = {"sni_hostname": original_host} if scheme == "https" and original_host else {}

    seen: set[tuple[str, int]] = set()
    targets: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for family, _type, _proto, _canon, sockaddr in addrinfos:
        if family == socket.AF_INET6:
            if len(sockaddr) < 2:
                continue
            resolved_ip = str(sockaddr[0])
            port = int(sockaddr[1])
            ip_host = f"[{resolved_ip}]"
        elif family == socket.AF_INET:
            resolved_ip = str(sockaddr[0])
            port = int(sockaddr[1])
            ip_host = resolved_ip
        else:
            continue

        key = (resolved_ip, port)
        if key in seen:
            continue
        seen.add(key)

        pinned_netloc = f"{ip_host}:{port}" if port != default_port else ip_host
        pinned_url = urlunparse(
            (scheme, pinned_netloc, parsed.path, parsed.params, parsed.query, "")
        )
        targets.append((pinned_url, headers, extensions))

    if not targets:
        raise WebFetchEgressViolation(f"Could not resolve host {original_host!r}")
    return targets


async def _run_web_search(query: str) -> list[dict[str, str]]:
    async with (
        httpx.AsyncClient(
            timeout=_REQUEST_TIMEOUT_S,
            follow_redirects=True,
            headers=_WEB_TOOL_HTTP_HEADERS,
        ) as client,
        client.stream(
            "GET",
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
        ) as response,
    ):
        response.raise_for_status()
        body_bytes = await _read_response_body_capped(
            response, constants._MAX_WEB_FETCH_RESPONSE_BYTES
        )
    text = body_bytes.decode("utf-8", errors="replace")
    parser = SearchResultParser()
    parser.feed(text)
    return parser.results[:_MAX_SEARCH_RESULTS]


def _text_encoding(charset: str | None) -> str:
    """Return a usable codec name from a Content-Type charset, defaulting to utf-8.

    Reads the header charset only without sniffing the unread response body.
    """
    if isinstance(charset, str) and charset:
        try:
            return codecs.lookup(charset).name
        except LookupError, ValueError:
            pass
    return "utf-8"


async def _run_web_fetch(url: str, egress: WebFetchEgressPolicy) -> dict[str, str]:
    """Fetch URL with manual redirects; each hop is DNS-pinned to validated addresses."""
    current_url = url
    redirect_hops = 0
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _REQUEST_TIMEOUT_S

    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError("web_fetch cumulative timeout exceeded")
        addr_infos = await asyncio.to_thread(
            get_validated_stream_addrinfos_for_egress, current_url, egress
        )
        targets = _pinned_fetch_targets(current_url, addr_infos)

        redirect_target: str | None = None
        last_connect_error: Exception | None = None
        final_url = current_url
        content_type = "text/plain"
        encoding = "utf-8"
        body_bytes = b""

        for pinned_url, headers, extensions in targets:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("web_fetch cumulative timeout exceeded")
            try:
                async with (
                    httpx.AsyncClient(
                        timeout=remaining,
                        follow_redirects=False,
                        trust_env=False,
                    ) as client,
                    client.stream(
                        "GET",
                        pinned_url,
                        headers=headers,
                        extensions=extensions or None,
                    ) as response,
                ):
                    if response.status_code in _WEB_FETCH_REDIRECT_STATUSES:
                        await _drain_response_body_capped(
                            response, _REDIRECT_RESPONSE_BODY_CAP_BYTES
                        )
                        if redirect_hops >= constants._MAX_WEB_FETCH_REDIRECTS:
                            raise WebFetchEgressViolation(
                                "web_fetch exceeded maximum redirects "
                                f"({constants._MAX_WEB_FETCH_REDIRECTS})"
                            )
                        location = response.headers.get("location")
                        if not location or not location.strip():
                            raise WebFetchEgressViolation(
                                "web_fetch redirect response missing Location header"
                            )
                        redirect_target = urljoin(current_url, location.strip())
                        break
                    response.raise_for_status()
                    raw_content_type = response.headers.get("content-type")
                    content_type = raw_content_type or "text/plain"
                    encoding = _text_encoding(
                        _charset_from_content_type(raw_content_type)
                    )
                    body_bytes = await _read_response_body_capped(
                        response, constants._MAX_WEB_FETCH_RESPONSE_BYTES
                    )
                    break
            except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as exc:
                last_connect_error = exc
                continue
        else:
            assert last_connect_error is not None
            raise last_connect_error

        if redirect_target is not None:
            current_url = redirect_target
            redirect_hops += 1
            continue

        break

    try:
        text = body_bytes.decode(encoding, errors="replace")
    except LookupError:
        text = body_bytes.decode("utf-8", errors="replace")
    title = final_url
    data = text
    if "html" in content_type.lower():
        parser = HTMLTextParser()
        parser.feed(text)
        title = parser.title or final_url
        data = "\n".join(parser.text_parts)
    return {
        "url": final_url,
        "title": title,
        "media_type": "text/plain",
        "data": data[:_MAX_FETCH_CHARS],
    }

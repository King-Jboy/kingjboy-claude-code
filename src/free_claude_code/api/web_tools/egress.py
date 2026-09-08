"""Egress policy for user-controlled web_fetch URLs (SSRF guard)."""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class WebFetchEgressPolicy:
    """Egress rules for user-influenced web_fetch URLs."""

    allow_private_network_targets: bool
    allowed_schemes: frozenset[str]


class WebFetchEgressViolation(ValueError):
    """Raised when a web_fetch URL is rejected by egress policy (SSRF guard)."""


def web_fetch_allowed_scheme_set(raw_schemes: str) -> frozenset[str]:
    """Return normalized schemes allowed for web_fetch."""

    return frozenset(
        part.strip().lower() for part in raw_schemes.split(",") if part.strip()
    )


def _port_for_url(parsed) -> int:
    if parsed.port is not None:
        return parsed.port
    return 443 if (parsed.scheme or "").lower() == "https" else 80


def _stream_getaddrinfo_or_raise(host: str, port: int) -> list[tuple]:
    try:
        return socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        raise WebFetchEgressViolation(
            f"Could not resolve host {host!r}: {exc}"
        ) from exc


def _is_ip_global(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_global
    return ip.is_global


def get_validated_stream_addrinfos_for_egress(
    url: str, policy: WebFetchEgressPolicy
) -> list[tuple]:
    """Resolve and validate a URL for web_fetch, returning getaddrinfo rows for pinning.

    Returns the raw ``socket.getaddrinfo`` tuples so callers can pin the exact
    resolved addresses rather than re-resolving and exposing themselves to DNS
    rebinding / time-of-check-time-of-use attacks.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        raise WebFetchEgressViolation(
            f"Scheme {scheme!r} is not allowed for web_fetch "
            f"(allowed: {sorted(policy.allowed_schemes)})"
        )

    host = parsed.hostname
    if host is None or host == "":
        raise WebFetchEgressViolation("web_fetch URL must include a host")

    port = _port_for_url(parsed)

    if policy.allow_private_network_targets:
        return _stream_getaddrinfo_or_raise(host, port)

    host_lower = host.lower()
    if host_lower == "localhost" or host_lower.endswith(".localhost"):
        raise WebFetchEgressViolation("localhost targets are not allowed for web_fetch")
    if host_lower.endswith(".local"):
        raise WebFetchEgressViolation(".local hostnames are not allowed for web_fetch")

    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        parsed_ip = None

    if parsed_ip is not None:
        if not _is_ip_global(parsed_ip):
            raise WebFetchEgressViolation(
                f"Non-public IP host {host!r} is not allowed for web_fetch"
            )
        return _stream_getaddrinfo_or_raise(host, port)

    infos = _stream_getaddrinfo_or_raise(host, port)
    for *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not _is_ip_global(resolved):
            raise WebFetchEgressViolation(
                f"Host {host!r} resolves to a non-public address ({resolved})"
            )
    return infos


def enforce_web_fetch_egress(url: str, policy: WebFetchEgressPolicy) -> None:
    """Validate ``url`` (scheme, host, and resolved addresses) for web_fetch."""
    get_validated_stream_addrinfos_for_egress(url, policy)

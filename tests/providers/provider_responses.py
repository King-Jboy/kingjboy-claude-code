"""Upstream responses recorded from real providers, replayed in tests.

Every payload here was captured from the live endpoint on the date noted, not
written by hand. That distinction is the point: the pool's failure policy
branches on which status a provider returns for a credential it refuses, and
providers disagree. Hand-written mocks encode the assumption under test, so
they agree with the code even when the code is wrong.

Re-record with a real key when a provider changes shape; the constants below
are the contract the pool is built against.
"""

from dataclasses import dataclass
from typing import Any

RECORDED_ON = "2026-08-04"


@dataclass(frozen=True, slots=True)
class RecordedResponse:
    """One captured upstream reply."""

    status: int
    json: dict[str, Any]
    headers: dict[str, str]
    note: str


# NVIDIA NIM answers 403 for an invalid credential. KeyPool.record_failure must
# not retire on this, because other providers use 403 to refuse the request.
NVIDIA_INVALID_KEY = RecordedResponse(
    status=403,
    json={"status": 403, "title": "Forbidden", "detail": "Authorization failed"},
    headers={},
    note=f"POST /v1/chat/completions, invalid nvapi- key, {RECORDED_ON}",
)

# OpenRouter answers 401 for an invalid credential, so the same condition takes
# the opposite branch and does retire the key.
OPENROUTER_INVALID_KEY = RecordedResponse(
    status=401,
    json={"error": {"message": "User not found.", "code": 401}},
    headers={},
    note=f"POST /v1/chat/completions, invalid sk-or-v1- key, {RECORDED_ON}",
)

# OpenRouter's free tier is 20 requests/minute per key. It sends no Retry-After
# and expresses the reset as epoch MILLISECONDS, which is the encoding
# _rate_limit_reset_seconds has to disambiguate by magnitude.
OPENROUTER_RATE_LIMITED = RecordedResponse(
    status=429,
    json={"error": {"message": "Rate limit exceeded", "code": 429}},
    headers={
        "X-RateLimit-Limit": "20",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1785829380000",
    },
    note=f"POST /v1/chat/completions, free-tier burst, {RECORDED_ON}",
)

# Both providers serve /v1/models without checking the credential, so a dead
# key "succeeds" there. Model discovery must not treat that as proof of health.
UNAUTHENTICATED_MODEL_LIST = RecordedResponse(
    status=200,
    json={"data": [{"id": "some/model"}]},
    headers={},
    note=f"GET /v1/models with an invalid key on NVIDIA NIM and OpenRouter, {RECORDED_ON}",
)

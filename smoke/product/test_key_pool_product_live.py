"""Live proof that credential-pool rotation survives dead and throttled keys.

The pool's whole failure policy turns on which status a provider returns for a
credential it will not accept, and providers disagree. Observed 2026-08-04:
NVIDIA NIM answers ``403``, OpenRouter answers ``401``. Those codes take
different branches in :func:`KeyPool.record_failure` - ``401`` cools a key on the
authentication ladder, ``403`` cools it briefly as an ambiguous refusal - so a
provider silently changing its answer would quietly degrade rotation without
failing any hermetic test. These probes
pin the observed behaviour against the real endpoints.
"""

import pytest
from openai import AsyncOpenAI

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.providers.key_pool import (
    KeyFailureAction,
    KeyPool,
    PooledKeyLease,
)
from free_claude_code.providers.runtime.config import build_provider_config
from smoke.lib.config import SmokeConfig

pytestmark = [
    pytest.mark.live,
    pytest.mark.provider,
    pytest.mark.smoke_target("providers"),
]

# Status each provider returns for a credential it refuses, observed live.
INVALID_CREDENTIAL_STATUS = {
    "nvidia_nim": 403,
    "open_router": 401,
}

# A refused credential must never escalate: the pool has to keep serving from
# the keys that still work, whichever status the provider chose to send.
EXPECTED_ACTIONS = frozenset({KeyFailureAction.HOP, KeyFailureAction.HOP_AMBIGUOUS})

_DUD = {
    "nvidia_nim": "nvapi-INVALID000000000000000000000000000000000000000",
    "open_router": "sk-or-v1-INVALID00000000000000000000000000000000000000000000000000000",
}


def _smoke_model(provider_id: str, smoke_config: SmokeConfig) -> str:
    """Return the model name this provider is smoke-tested with, or skip."""
    for candidate in smoke_config.provider_smoke_models():
        if candidate.provider == provider_id:
            return candidate.model_name
    pytest.skip(f"missing_env: no smoke model resolved for {provider_id}")


def _pool_for(provider_id: str, keys: list[str], base_url: str) -> KeyPool:
    return KeyPool(
        keys,
        provider_name=provider_id,
        client_factory=lambda key: AsyncOpenAI(
            api_key=key, base_url=base_url, max_retries=0
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", sorted(INVALID_CREDENTIAL_STATUS))
async def test_invalid_credential_status_matches_pool_policy_e2e(
    provider_id: str, smoke_config: SmokeConfig
) -> None:
    if not smoke_config.has_provider_configuration(provider_id):
        pytest.skip(f"missing_env: {provider_id} is not configured")

    config = build_provider_config(PROVIDER_CATALOG[provider_id], smoke_config.settings)
    pool = _pool_for(provider_id, [_DUD[provider_id]], config.base_url)
    lease = await pool.acquire()
    expected = INVALID_CREDENTIAL_STATUS[provider_id]
    try:
        # The SDK raises a different subclass per status, which is the very
        # thing under test, so this cannot narrow to one exception type.
        with pytest.raises(Exception) as error:
            await lease.client.chat.completions.create(
                model=_smoke_model(provider_id, smoke_config),
                messages=[{"role": "user", "content": "x"}],
                max_tokens=1,
            )
        status = getattr(error.value, "status_code", None)
        assert status == expected, (
            f"{provider_id} now answers {status} for an invalid credential, not "
            f"{expected}. KeyPool.record_failure branches on this: 401 cools a "
            f"key on the authentication ladder, 403 cools it briefly. Re-check "
            f"the classification before trusting rotation on this provider."
        )
        action = pool.record_failure(PooledKeyLease(0, lease.client), error.value)
        assert action in EXPECTED_ACTIONS, (
            f"{provider_id} refused a credential and the pool chose {action}; a "
            f"refused key must be skipped, never escalated past the other keys."
        )
    finally:
        await pool.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", sorted(INVALID_CREDENTIAL_STATUS))
async def test_dead_keys_never_strand_a_working_key_e2e(
    provider_id: str, smoke_config: SmokeConfig
) -> None:
    config = build_provider_config(PROVIDER_CATALOG[provider_id], smoke_config.settings)
    real = [key for key in config.api_keys if key]
    if len(real) < 2:
        pytest.skip(f"missing_env: {provider_id} has no configured key pool")

    # Three duds ahead of the real keys: the pool must walk every one of them.
    keys = [f"{_DUD[provider_id]}{index}" for index in range(3)] + list(real[:2])
    pool = _pool_for(provider_id, keys, config.base_url)
    model = _smoke_model(provider_id, smoke_config)
    try:
        completion = await pool.run_key_local(
            lambda client: client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "reply with just: OK"}],
                max_tokens=8,
            )
        )
    finally:
        await pool.aclose()

    assert completion.choices, f"{provider_id} pool returned no choices behind 3 duds"

"""`GET /api/pool-status`: read-only pool health for API clients."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from free_claude_code.api.dependencies import get_settings
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import ProviderConfig
from tests.api.support import create_test_app
from tests.providers.support import profiled_provider


def _pooled_app() -> FastAPI:
    provider = profiled_provider(
        "groq",
        ProviderConfig(
            api_key="primary",
            base_url="https://pool-status.test",
            api_keys=("key-a", "key-b"),
        ),
    )
    return create_test_app(providers={"groq": provider})


def test_pool_status_reports_counts_only() -> None:
    # The payload crosses into client space, so it must stay counts: echoing a
    # key here would undo the point of pooling them behind the proxy.
    response = TestClient(_pooled_app()).get("/api/pool-status")

    assert response.status_code == 200
    assert response.json() == {
        "key_pools": {
            "groq": {
                "size": 2,
                "ready": 2,
                "cooling": 0,
                "retired": 0,
                "soonest_ready_in": None,
            }
        }
    }


def test_pool_status_requires_the_proxy_token_when_configured() -> None:
    # Same gate as /v1/models: open when no token is set, Bearer-only otherwise.
    app = _pooled_app()
    settings = Settings()
    settings.anthropic_auth_token = "s3cr3t"
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    assert client.get("/api/pool-status").status_code == 401

    authorized = client.get(
        "/api/pool-status", headers={"authorization": "Bearer s3cr3t"}
    )
    assert authorized.status_code == 200
    assert "groq" in authorized.json()["key_pools"]

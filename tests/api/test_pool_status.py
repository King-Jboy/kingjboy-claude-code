"""API endpoints exposing pooled-credential health."""

from fastapi.testclient import TestClient

from tests.api.support import create_test_app


def test_pool_status_endpoint():
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/api/pool-status")
    assert response.status_code == 200
    assert "key_pools" in response.json()
    assert isinstance(response.json()["key_pools"], dict)


def test_admin_key_pools_endpoint():
    app = create_test_app()
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.get("/admin/api/providers/key-pools")
    assert response.status_code == 200
    assert "key_pools" in response.json()
    assert isinstance(response.json()["key_pools"], dict)

from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.model_refs import ModelCatalogScope
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, provider_manager_for_app


def _settings(
    *,
    model: str = "deepseek/deepseek-chat",
    model_fable: str | None = None,
    model_opus: str | None = "open_router/anthropic/claude-opus",
    model_haiku: str | None = "deepseek/deepseek-chat",
    model_catalog_scope: ModelCatalogScope = ModelCatalogScope.ALL,
    pinned_models: str = "",
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=None,
        model_haiku=model_haiku,
        model_catalog_scope=model_catalog_scope,
        pinned_models=pinned_models,
        anthropic_auth_token="",
        deepseek_api_key="deepseek-key",
        open_router_api_key="open-router-key",
        zai_api_key="zai-key",
    )


def _cache_models(app, provider_id: str, *model_ids: str) -> None:
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {ProviderModelInfo(model_id) for model_id in model_ids},
    )


def test_models_list_includes_configured_refs_cached_provider_models_and_aliases():
    app = create_test_app(_settings())
    _cache_models(app, "deepseek", "deepseek-chat")
    _cache_models(
        app,
        "open_router",
        "meta/llama-3.3",
        "anthropic/claude-opus",
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    ids = [item["id"] for item in data["data"]]
    assert ids[:6] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
        "anthropic/open_router/meta/llama-3.3",
        "claude-3-freecc-no-thinking/open_router/meta/llama-3.3",
    ]
    assert ids.count("anthropic/deepseek/deepseek-chat") == 1
    assert ids.count("anthropic/open_router/anthropic/claude-opus") == 1
    display_names = {item["id"]: item["display_name"] for item in data["data"]}
    assert (
        display_names["anthropic/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3"
    )
    assert (
        display_names["claude-3-freecc-no-thinking/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3 (no thinking)"
    )
    assert "claude-sonnet-4-20250514" in ids
    assert "claude-fable-5" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]
    assert data["has_more"] is False


def test_models_list_uses_thinking_metadata_for_cached_models():
    app = create_test_app(_settings(model_opus=None))
    manager = provider_manager_for_app(app)
    _cache_models(app, "deepseek", "deepseek-chat")
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/reasoning-model" in ids
    assert "claude-3-freecc-no-thinking/open_router/reasoning-model" in ids
    assert "anthropic/open_router/plain-model" not in ids
    assert "claude-3-freecc-no-thinking/open_router/plain-model" in ids


def test_models_list_uses_cached_metadata_for_configured_refs():
    app = create_test_app(
        _settings(
            model="open_router/plain-model",
            model_opus=None,
            model_haiku=None,
        )
    )
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {ProviderModelInfo("plain-model", supports_thinking=False)},
    )

    response = TestClient(app).get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/plain-model" not in ids
    assert ids[0] == "claude-3-freecc-no-thinking/open_router/plain-model"


def test_models_list_includes_cached_zai_models():
    app = create_test_app(
        _settings(
            model="zai/glm-4.7",
            model_opus=None,
            model_haiku=None,
        )
    )
    _cache_models(app, "zai", "glm-4.7", "glm-4.7-air")

    response = TestClient(app).get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/zai/glm-4.7" in ids
    assert "claude-3-freecc-no-thinking/zai/glm-4.7" in ids
    assert "anthropic/zai/glm-4.7-air" in ids
    assert "claude-3-freecc-no-thinking/zai/glm-4.7-air" in ids


def test_configured_scope_drops_discovered_models_but_keeps_routes_and_aliases():
    """A picker under the configured scope lists routes, not the whole catalog."""

    app = create_test_app(_settings(model_catalog_scope=ModelCatalogScope.CONFIGURED))
    _cache_models(app, "deepseek", "deepseek-chat")
    _cache_models(app, "open_router", "meta/llama-3.3", "anthropic/claude-opus")

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/meta/llama-3.3" not in ids
    assert "claude-3-freecc-no-thinking/open_router/meta/llama-3.3" not in ids
    # Configured routes survive, and so do the Claude aliases the client sends.
    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "anthropic/open_router/anthropic/claude-opus" in ids
    assert "claude-sonnet-4-20250514" in ids


def test_configured_scope_still_honours_cached_thinking_metadata():
    """Scoping changes which models are listed, never how a route is shaped."""

    app = create_test_app(
        _settings(
            model="open_router/plain-model",
            model_opus=None,
            model_haiku=None,
            model_catalog_scope=ModelCatalogScope.CONFIGURED,
        )
    )
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {ProviderModelInfo("plain-model", supports_thinking=False)},
    )

    ids = [item["id"] for item in TestClient(app).get("/v1/models").json()["data"]]

    assert "anthropic/open_router/plain-model" not in ids
    assert ids[0] == "claude-3-freecc-no-thinking/open_router/plain-model"


def test_pinned_models_extend_a_scoped_list_beyond_the_routing_slots():
    """The five MODEL_* slots are not the ceiling on a personalised list."""

    app = create_test_app(
        _settings(
            model_opus=None,
            model_haiku=None,
            model_catalog_scope=ModelCatalogScope.CONFIGURED,
            pinned_models=(
                '["open_router/meta/llama-3.3", "open_router/z-ai/glm-5.2:free"]'
            ),
        )
    )
    _cache_models(app, "open_router", "meta/llama-3.3", "unwanted/model")

    ids = [item["id"] for item in TestClient(app).get("/v1/models").json()["data"]]

    assert "anthropic/open_router/meta/llama-3.3" in ids
    assert "anthropic/open_router/z-ai/glm-5.2:free" in ids
    assert "anthropic/deepseek/deepseek-chat" in ids
    # Pinning is a shortlist, not a second way to list everything discovered.
    assert "anthropic/open_router/unwanted/model" not in ids


def test_a_pinned_model_is_listed_even_when_discovery_never_saw_it():
    """A slug typed by hand must be selectable before any refresh finds it."""

    app = create_test_app(
        _settings(
            model_opus=None,
            model_haiku=None,
            model_catalog_scope=ModelCatalogScope.CONFIGURED,
            pinned_models='["nvidia_nim/brand-new/model"]',
        )
    )

    ids = [item["id"] for item in TestClient(app).get("/v1/models").json()["data"]]

    assert "anthropic/nvidia_nim/brand-new/model" in ids


def test_pinned_models_are_not_duplicated_by_the_configured_routes():
    app = create_test_app(
        _settings(
            model="deepseek/deepseek-chat",
            model_opus=None,
            model_haiku=None,
            model_catalog_scope=ModelCatalogScope.CONFIGURED,
            pinned_models='["deepseek/deepseek-chat"]',
        )
    )

    ids = [item["id"] for item in TestClient(app).get("/v1/models").json()["data"]]

    assert ids.count("anthropic/deepseek/deepseek-chat") == 1


def test_models_list_works_with_empty_discovery_catalog():
    app = create_test_app(_settings())

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids[:4] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
    ]
    assert "claude-sonnet-4-20250514" in ids

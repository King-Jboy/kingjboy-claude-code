"""Local admin UI routes and APIs."""

import asyncio
import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import validate_updates
from free_claude_code.config.admin.values import load_config_response
from free_claude_code.config.model_refs import (
    ModelCatalogScope,
    configured_chat_model_refs,
    pinned_model_refs,
)
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)

from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
LOCAL_PROVIDER_PATHS = {
    "lmstudio": "/models",
    "ollama": "/api/tags",
}
LOCAL_PROBE_TIMEOUT_SECONDS = 1.5


class AdminConfigPayload(BaseModel):
    """Partial config update submitted by the admin UI."""

    values: dict[str, Any] = Field(default_factory=dict)


class ConnectedAccountLoginPayload(BaseModel):
    """Interactive connected-account login selection."""

    mode: ConnectedAccountLoginMode = ConnectedAccountLoginMode.BROWSER


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _origin_is_local(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return _is_loopback_host(parsed.hostname)


def require_loopback_admin(request: Request) -> None:
    """Allow admin access only from the local machine."""

    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")

    origin = request.headers.get("origin")
    if not _origin_is_local(origin):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")


def _asset_response(filename: str) -> FileResponse:
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return FileResponse(path)


@router.get("/admin", include_in_schema=False)
async def admin_page(request: Request):
    require_loopback_admin(request)
    return _asset_response("index.html")


@router.get("/admin/assets/{filename}", include_in_schema=False)
async def admin_asset(filename: str, request: Request):
    require_loopback_admin(request)
    if filename not in {"admin.css", "admin.js"}:
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return _asset_response(filename)


@router.get("/admin/api/config")
async def get_admin_config(request: Request):
    require_loopback_admin(request)
    return load_config_response()


@router.post("/admin/api/config/validate")
async def validate_admin_config(payload: AdminConfigPayload, request: Request):
    require_loopback_admin(request)
    return validate_updates(_filtered_values(payload.values))


@router.post("/admin/api/config/apply")
async def apply_admin_config(
    payload: AdminConfigPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.apply_admin_config(_filtered_values(payload.values))
    restart = result.get("restart")
    if isinstance(restart, dict) and restart.get("automatic"):
        background_tasks.add_task(services.admin.request_restart)
    return result


@router.get("/admin/api/status")
async def admin_status(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return services.admin.admin_status()


@router.post("/admin/api/server/restart")
async def restart_server(
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    """Restart the server that serves this page.

    Dispatched after the response so the browser is told the restart began
    rather than losing the connection mid-request and reporting a failure the
    user cannot distinguish from a crash.
    """

    require_loopback_admin(request)
    background_tasks.add_task(services.admin.request_restart)
    return {"server": "restarting"}


@router.post("/admin/api/server/stop")
async def stop_server(
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    """Stop the server, ending the Admin UI with it.

    Deferred for the same reason as restart. Nothing here can start the server
    again, which is why the UI has to say where the next start comes from.
    """

    require_loopback_admin(request)
    background_tasks.add_task(services.admin.request_stop)
    return {"server": "stopping"}


@router.get("/admin/api/providers/local-status")
async def local_provider_status(request: Request):
    require_loopback_admin(request)
    config = load_config_response()
    values = {field["key"]: field["value"] for field in config["fields"]}
    # Probe every local provider at once and over one connection pool. Serial
    # probes made an all-offline check cost the sum of every timeout.
    async with httpx.AsyncClient(timeout=LOCAL_PROBE_TIMEOUT_SECONDS) as client:
        checks = await asyncio.gather(
            *(
                _check_local_provider(
                    client,
                    provider_id,
                    _local_provider_url(provider_id, values),
                    path,
                )
                for provider_id, path in LOCAL_PROVIDER_PATHS.items()
            )
        )
    return {"providers": list(checks)}


@router.get("/admin/api/providers/key-pools")
async def key_pool_status(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    """Report pooled-credential health so silent capacity loss is visible."""
    require_loopback_admin(request)
    return {"key_pools": services.admin.key_pool_status()}


@router.post("/admin/api/providers/{provider_id}/test")
async def test_provider(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return await services.admin.test_provider(provider_id)


@router.get("/admin/api/providers/{provider_id}/auth")
async def connected_account_status(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.connected_account_status(provider_id)
    return _no_store(status.as_dict())


@router.post("/admin/api/providers/{provider_id}/auth/login")
async def start_connected_account_login(
    provider_id: str,
    payload: ConnectedAccountLoginPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    try:
        status = await services.admin.start_connected_account_login(
            provider_id, payload.mode
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(f"Could not start connected-account login ({type(exc).__name__})."),
        ) from exc
    return _no_store(status.as_dict())


@router.post("/admin/api/providers/{provider_id}/auth/cancel")
async def cancel_connected_account_login(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.cancel_connected_account_login(provider_id)
    return _no_store(status.as_dict())


@router.delete("/admin/api/providers/{provider_id}/auth")
async def disconnect_connected_account(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.disconnect_connected_account(provider_id)
    return _no_store(status.as_dict())


@router.get("/admin/api/models")
async def models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return _model_options(services)


@router.post("/admin/api/models/refresh")
async def refresh_models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.refresh_models()
    # Refreshing is an explicit "show me what the providers have" action, so it
    # ignores MODEL_CATALOG_SCOPE. That keeps the scoped default short without
    # stranding anyone who wants to switch to a model they have not configured.
    return _model_options(services, refresh_result=result, include_discovered=True)


def _model_options(
    services: ApiServices,
    *,
    refresh_result: ProviderModelRefreshResult | None = None,
    include_discovered: bool | None = None,
) -> dict[str, list[str]]:
    settings = services.requests.current_settings()
    configured = {ref.model_ref for ref in configured_chat_model_refs(settings)}
    configured |= set(pinned_model_refs(settings))
    if include_discovered is None:
        include_discovered = settings.model_catalog_scope is ModelCatalogScope.ALL
    discovered = (
        {info.model_id for info in services.requests.cached_prefixed_model_infos()}
        if include_discovered
        else set()
    )
    failed_provider_ids = (
        refresh_result.failed_provider_ids if refresh_result is not None else ()
    )
    return {
        "models": sorted(configured | discovered, key=str.casefold),
        "failed_providers": list(failed_provider_ids),
    }


def _filtered_values(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in FIELD_BY_KEY}


def _local_provider_url(provider_id: str, values: dict[str, str]) -> str:
    if provider_id == "lmstudio":
        return values.get("LM_STUDIO_BASE_URL", "")
    if provider_id == "ollama":
        return values.get("OLLAMA_BASE_URL", "")
    return ""


async def _check_local_provider(
    client: httpx.AsyncClient, provider_id: str, base_url: str, path: str
) -> dict[str, Any]:
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return {
            "provider_id": provider_id,
            "status": "missing_url",
            "label": "Missing URL",
            "base_url": base_url,
        }

    url = f"{clean_url}{path}"
    try:
        response = await client.get(url)
        ok = 200 <= response.status_code < 300
        return {
            "provider_id": provider_id,
            "status": "reachable" if ok else "offline",
            "label": "Reachable" if ok else "Offline",
            "base_url": base_url,
            "status_code": response.status_code,
        }
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "status": "offline",
            "label": "Offline",
            "base_url": base_url,
            "error_type": type(exc).__name__,
        }


def _require_connected_account_provider(provider_id: str) -> None:
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if (
        descriptor is None
        or descriptor.auth_kind is not ProviderAuthKind.CONNECTED_ACCOUNT
    ):
        raise HTTPException(
            status_code=404,
            detail="Provider does not support connected-account login.",
        )


def _no_store(payload: Any) -> JSONResponse:
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})

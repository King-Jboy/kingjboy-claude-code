"""OpenAI browser and device authorization flows."""

import asyncio
import base64
import contextlib
import hashlib
import html
import secrets
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

OPENAI_AUTH_ISSUER = "https://auth.openai.com"
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_ORIGINATOR = "codex_cli_rs"
OPENAI_OAUTH_SCOPE = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
OPENAI_TOKEN_URL = f"{OPENAI_AUTH_ISSUER}/oauth/token"
OPENAI_REVOKE_URL = f"{OPENAI_AUTH_ISSUER}/oauth/revoke"
OPENAI_DEVICE_USER_CODE_URL = f"{OPENAI_AUTH_ISSUER}/api/accounts/deviceauth/usercode"
OPENAI_DEVICE_TOKEN_URL = f"{OPENAI_AUTH_ISSUER}/api/accounts/deviceauth/token"
OPENAI_DEVICE_VERIFICATION_URL = f"{OPENAI_AUTH_ISSUER}/codex/device"
OPENAI_DEVICE_REDIRECT_URI = f"{OPENAI_AUTH_ISSUER}/deviceauth/callback"
OPENAI_CALLBACK_HOST = "localhost"
LOGIN_LIFETIME_SECONDS = 15 * 60

_HTTP_REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
}


class OpenAILoginError(RuntimeError):
    """An interactive authorization flow could not complete."""


@dataclass(frozen=True, slots=True, repr=False)
class AuthorizationGrant:
    """OAuth authorization code and matching PKCE material."""

    code: str
    redirect_uri: str
    code_verifier: str


@dataclass(frozen=True, slots=True, repr=False)
class _DeviceAuthorization:
    verification_url: str
    user_code: str
    expires_at: int
    interval: float
    device_auth_id: str


class BrowserAuthorization:
    """Short-lived loopback callback server for one PKCE login."""

    def __init__(
        self,
        *,
        auth_url: str,
        redirect_uri: str,
        code_verifier: str,
        server: asyncio.Server,
        result: asyncio.Future[AuthorizationGrant],
    ) -> None:
        self.auth_url = auth_url
        self.redirect_uri = redirect_uri
        self.code_verifier = code_verifier
        self._server = server
        self._result = result
        self._closed = False

    @classmethod
    async def start(cls) -> BrowserAuthorization:
        """Bind an allow-listed loopback port and construct the authorization URL."""

        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
        loop = asyncio.get_running_loop()
        result: asyncio.Future[AuthorizationGrant] = loop.create_future()
        server: asyncio.Server | None = None
        redirect_uri = ""
        for candidate in (1455, 1457):
            candidate_redirect_uri = (
                f"http://{OPENAI_CALLBACK_HOST}:{candidate}/auth/callback"
            )

            async def handle_client(
                reader: asyncio.StreamReader,
                writer: asyncio.StreamWriter,
                callback_redirect_uri: str = candidate_redirect_uri,
            ) -> None:
                try:
                    try:
                        raw = await asyncio.wait_for(
                            reader.readuntil(b"\r\n\r\n"),
                            timeout=5.0,
                        )
                    except asyncio.IncompleteReadError as exc:
                        raw = exc.partial
                    except TimeoutError, asyncio.LimitOverrunError, OSError:
                        return
                    if not raw:
                        return

                    request_line = raw.split(b"\r\n", 1)[0].decode(
                        "latin-1", errors="replace"
                    )
                    parts = request_line.split(" ", 2)
                    if len(parts) < 2:
                        response = _http_response(400, "Bad Request")
                    else:
                        method, target = parts[0], parts[1]
                        parsed = urlsplit(target)
                        if parsed.path != "/auth/callback":
                            response = _http_response(404, "Not Found")
                        elif method != "GET":
                            response = _http_response(405, "Method Not Allowed")
                        else:
                            query = parse_qs(parsed.query, keep_blank_values=True)
                            req_state = query.get("state", [None])[0]
                            req_error = query.get("error", [None])[0]
                            req_code = query.get("code", [None])[0]
                            if req_state != state:
                                response = _http_response(400, "State mismatch")
                            elif req_error:
                                description = (
                                    query.get("error_description", [None])[0]
                                    or req_error
                                )
                                if not result.done():
                                    result.set_exception(OpenAILoginError(description))
                                response = _callback_page(
                                    "Sign-in failed", description, status=400
                                )
                            elif not req_code:
                                response = _http_response(
                                    400, "Missing authorization code"
                                )
                            else:
                                if not result.done():
                                    result.set_result(
                                        AuthorizationGrant(
                                            code=req_code,
                                            redirect_uri=callback_redirect_uri,
                                            code_verifier=verifier,
                                        )
                                    )
                                response = _callback_page(
                                    "Connected",
                                    "OpenAI is connected to Free Claude Code. You can close this tab.",
                                    close_window=True,
                                )

                    writer.write(response)
                    await writer.drain()
                except OSError:
                    return
                finally:
                    writer.close()
                    with contextlib.suppress(OSError):
                        await writer.wait_closed()

            try:
                candidate_server = await asyncio.start_server(
                    handle_client,
                    OPENAI_CALLBACK_HOST,
                    candidate,
                )
            except OSError:
                continue
            server = candidate_server
            redirect_uri = candidate_redirect_uri
            break
        if server is None:
            raise OpenAILoginError(
                "OpenAI sign-in could not bind localhost ports 1455 or 1457."
            )

        auth_url = f"{OPENAI_AUTH_ISSUER}/oauth/authorize?" + urlencode(
            {
                "response_type": "code",
                "client_id": OPENAI_OAUTH_CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": OPENAI_OAUTH_SCOPE,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": state,
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
                "originator": OPENAI_CODEX_ORIGINATOR,
            }
        )
        return cls(
            auth_url=auth_url,
            redirect_uri=redirect_uri,
            code_verifier=verifier,
            server=server,
            result=result,
        )

    async def wait(self) -> AuthorizationGrant:
        """Wait for one valid callback, bounded by the login lifetime."""

        try:
            return await asyncio.wait_for(self._result, timeout=LOGIN_LIFETIME_SECONDS)
        except TimeoutError as exc:
            raise OpenAILoginError("OpenAI sign-in timed out.") from exc

    async def close(self) -> None:
        """Stop the callback listener."""

        if self._closed:
            return
        self._closed = True
        if not self._result.done():
            self._result.cancel()
        self._server.close()
        await self._server.wait_closed()


async def request_device_authorization(
    client: httpx.AsyncClient,
) -> _DeviceAuthorization:
    """Request a device code from OpenAI's documented Codex flow."""

    response = await client.post(
        OPENAI_DEVICE_USER_CODE_URL,
        json={"client_id": OPENAI_OAUTH_CLIENT_ID},
    )
    response.raise_for_status()
    payload = response.json()
    device_auth_id = payload.get("device_auth_id")
    user_code = payload.get("user_code", payload.get("usercode"))
    if not isinstance(device_auth_id, str) or not isinstance(user_code, str):
        raise OpenAILoginError("OpenAI returned an invalid device-code response.")
    try:
        interval = max(1.0, float(payload.get("interval", 5)))
    except TypeError, ValueError:
        interval = 5.0
    return _DeviceAuthorization(
        verification_url=OPENAI_DEVICE_VERIFICATION_URL,
        user_code=user_code,
        expires_at=int(time.time()) + LOGIN_LIFETIME_SECONDS,
        interval=interval,
        device_auth_id=device_auth_id,
    )


async def wait_for_device_grant(
    client: httpx.AsyncClient,
    authorization: _DeviceAuthorization,
) -> AuthorizationGrant:
    """Poll until OpenAI issues the PKCE-backed authorization code."""

    while time.time() < authorization.expires_at:
        response = await client.post(
            OPENAI_DEVICE_TOKEN_URL,
            json={
                "device_auth_id": authorization.device_auth_id,
                "user_code": authorization.user_code,
            },
        )
        if response.is_success:
            payload = response.json()
            code = payload.get("authorization_code")
            verifier = payload.get("code_verifier")
            if not isinstance(code, str) or not isinstance(verifier, str):
                raise OpenAILoginError(
                    "OpenAI returned an invalid device authorization response."
                )
            return AuthorizationGrant(
                code=code,
                redirect_uri=OPENAI_DEVICE_REDIRECT_URI,
                code_verifier=verifier,
            )
        if response.status_code not in {403, 404}:
            response.raise_for_status()
        await asyncio.sleep(authorization.interval)
    raise OpenAILoginError("OpenAI device sign-in timed out.")


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _http_response(
    status: int,
    body: str,
    *,
    content_type: str = "text/plain; charset=utf-8",
) -> bytes:
    payload = body.encode("utf-8")
    reason = _HTTP_REASONS.get(status, "OK")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    return head + payload


def _callback_page(
    title: str,
    message: str,
    *,
    status: int = 200,
    close_window: bool = False,
) -> bytes:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    close_script = "<script>window.close()</script>" if close_window else ""
    return _http_response(
        status,
        (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{safe_title}</title>"
            "<style>body{font-family:system-ui;max-width:36rem;margin:12vh auto;"
            "padding:2rem;color:#202124}h1{font-size:1.6rem}</style>"
            f"<h1>{safe_title}</h1><p>{safe_message}</p>"
            f"{close_script}"
        ),
        content_type="text/html; charset=utf-8",
    )

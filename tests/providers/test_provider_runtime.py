import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import (
    UnknownProviderError,
)
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import (
    HUGGINGFACE_DEFAULT_BASE,
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ZAI_DEFAULT_BASE,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.deepseek import DeepSeekProvider
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.lmstudio import LMStudioProvider
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    OpenAIChatProvider,
)
from free_claude_code.providers.openai_codex import OpenAICodexProvider
from free_claude_code.providers.runtime import (
    ProviderRuntime,
    build_provider_config,
    create_provider,
)


def _make_settings(**overrides):
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
    mock.model_fable = None
    mock.model_opus = None
    mock.model_sonnet = None
    mock.model_haiku = None
    mock.nvidia_nim_api_key = "test_key"
    mock.open_router_api_key = "test_openrouter_key"
    mock.deepseek_api_key = "test_deepseek_key"
    mock.huggingface_api_key = "test_huggingface_key"
    mock.zai_api_key = "test_zai_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.ollama_base_url = "http://localhost:11434"
    mock.nvidia_nim_proxy = ""
    mock.open_router_proxy = ""
    mock.lmstudio_proxy = ""
    mock.kimi_proxy = ""
    mock.kimi_api_key = "test_kimi_key"
    mock.huggingface_proxy = ""
    mock.groq_api_key = ""
    mock.groq_proxy = ""
    mock.gemini_api_key = "test_gemini_key"
    mock.gemini_proxy = ""
    mock.bedrock_api_key = "test_bedrock_key"
    mock.bedrock_base_url = "https://bedrock-mantle.us-east-1.api.aws/v1"
    mock.bedrock_proxy = ""
    mock.tokenrouter_api_key = "test_tokenrouter_key"
    mock.tokenrouter_base_url = "https://api.tokenrouter.com/v1"
    mock.tokenrouter_proxy = ""
    mock.nararoute_api_key = "test_nararoute_key"
    mock.nararoute_base_url = "https://router.bynara.id/v1"
    mock.nararoute_proxy = ""
    mock.openai_proxy = ""
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.log_raw_sse_events = False
    mock.log_api_error_tracebacks = False
    mock.nim = NimSettings()
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def test_importing_runtime_does_not_eager_load_other_adapters() -> None:
    """Runtime metadata must not import every provider adapter up front."""
    code = (
        "import sys\n"
        "import free_claude_code.providers.runtime\n"
        "assert 'free_claude_code.providers.open_router' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_provider_catalog_covers_advertised_provider_ids():
    assert set(PROVIDER_CATALOG) == set(SUPPORTED_PROVIDER_IDS)
    assert set(OPENAI_CHAT_PROFILES) < set(PROVIDER_CATALOG)
    for descriptor in PROVIDER_CATALOG.values():
        assert descriptor.provider_id


def test_ollama_descriptor_uses_local_openai_endpoint_semantics():
    descriptor = PROVIDER_CATALOG["ollama"]

    assert descriptor.default_base_url == "http://localhost:11434"
    assert descriptor.local is True


@pytest.mark.parametrize(
    ("provider_id", "expected_api_key"),
    [
        ("lmstudio", "lm-studio"),
        ("ollama", "ollama"),
    ],
)
def test_local_provider_factory_resolves_catalog_static_credential(
    provider_id: str,
    expected_api_key: str,
) -> None:
    descriptor = PROVIDER_CATALOG[provider_id]
    settings = _make_settings()

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider(provider_id, settings)

    assert config.api_key == expected_api_key
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._api_key == expected_api_key


def test_zai_descriptor_uses_fixed_cloud_base_url():
    descriptor = PROVIDER_CATALOG["zai"]

    assert descriptor.default_base_url == ZAI_DEFAULT_BASE
    assert descriptor.base_url_attr is None


def test_zai_provider_config_ignores_stale_base_url_setting():
    descriptor = PROVIDER_CATALOG["zai"]

    config = build_provider_config(
        descriptor,
        _make_settings(zai_base_url="https://custom.zai.invalid/v1"),
    )

    assert config.base_url == ZAI_DEFAULT_BASE


def test_huggingface_descriptor_uses_openai_chat_router() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]

    assert descriptor.default_base_url == HUGGINGFACE_DEFAULT_BASE
    assert descriptor.credential_env == "HUGGINGFACE_API_KEY"
    assert descriptor.proxy_attr == "huggingface_proxy"


def test_build_provider_config_huggingface_uses_api_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]
    settings = _make_settings(
        huggingface_api_key="hf-token",
        huggingface_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "hf-token"
    assert config.proxy == "http://proxy.test:8080"


def test_create_provider_uses_openai_chat_openrouter_by_default():
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("open_router", _make_settings())

    assert isinstance(provider, OpenRouterProvider)


def test_create_provider_instantiates_each_builtin():
    settings = _make_settings(
        groq_api_key="test_groq_key",
        huggingface_api_key="test_huggingface_key",
        kimi_api_key="test_kimi_key",
        zai_api_key="test_zai_key",
        deepseek_api_key="test_deepseek_key",
        provider_rate_limit=7,
        provider_rate_window=11,
        provider_max_concurrency=3,
    )
    cases = {
        "nvidia_nim": NvidiaNimProvider,
        "openai": OpenAICodexProvider,
        "open_router": OpenRouterProvider,
        "deepseek": DeepSeekProvider,
        "huggingface": OpenAIChatProvider,
        "kimi": OpenAIChatProvider,
        "groq": OpenAIChatProvider,
        "zai": OpenAIChatProvider,
        "gemini": GeminiProvider,
        "bedrock": OpenAIChatProvider,
        "tokenrouter": OpenAIChatProvider,
        "nararoute": OpenAIChatProvider,
        "lmstudio": LMStudioProvider,
        "ollama": OpenAIChatProvider,
    }
    sentinel_admission = MagicMock(spec=ProviderAdmissionController)
    auth = MagicMock()
    injected_factories = {
        "openai": lambda config, _settings, admission: OpenAICodexProvider(
            config,
            auth=auth,
            admission=admission,
        )
    }

    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
        patch(
            "free_claude_code.providers.runtime.factory.ProviderAdmissionController",
            return_value=sentinel_admission,
        ) as admission_factory,
    ):
        for provider_id, provider_cls in cases.items():
            provider = create_provider(
                provider_id,
                settings,
                injected_factories=injected_factories,
            )

            assert isinstance(provider, provider_cls)
            assert provider._admission is sentinel_admission
            admission_factory.assert_called_once_with(
                provider_name=provider_id,
                # An explicitly configured quota overrides the provider's own,
                # then the safety margin holds back one whole request from it.
                rate_limit=6,
                rate_window=11,
                max_concurrency=3,
            )
            admission_factory.reset_mock()

    assert set(cases) == set(PROVIDER_CATALOG)


def test_provider_runtime_caches_by_provider_id():
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert first is second


def test_provider_runtime_provider_owns_one_admission_controller() -> None:
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert isinstance(first, NvidiaNimProvider)
    assert isinstance(second, NvidiaNimProvider)
    assert first._admission is second._admission


def test_separate_provider_runtimes_never_share_admission_controllers() -> None:
    first_runtime = ProviderRuntime(_make_settings())
    second_runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = first_runtime.resolve_provider("nvidia_nim")
        second = second_runtime.resolve_provider("nvidia_nim")

    assert isinstance(first, NvidiaNimProvider)
    assert isinstance(second, NvidiaNimProvider)
    assert first is not second
    assert first._admission is not second._admission


def test_different_providers_have_independent_admission_controllers() -> None:
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        nim = runtime.resolve_provider("nvidia_nim")
        open_router = runtime.resolve_provider("open_router")

    assert isinstance(nim, NvidiaNimProvider)
    assert isinstance(open_router, OpenRouterProvider)
    assert nim._admission is not open_router._admission


def test_unknown_provider_raises_unknown_provider_type_error():
    with pytest.raises(UnknownProviderError, match="Unknown provider_type"):
        create_provider("unknown", _make_settings())


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_runs_all_even_if_one_fails() -> None:
    """Successful providers leave the cache while failed providers remain retryable."""
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("first"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock()
    runtime = ProviderRuntime(_make_settings(), {"a": p1, "b": p2})

    with pytest.raises(RuntimeError, match="first"):
        await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    p2.cleanup.assert_awaited_once()
    assert runtime.is_cached("a")
    assert not runtime.is_cached("b")

    p1.cleanup = AsyncMock()
    await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    assert not runtime.is_cached("a")


@pytest.mark.asyncio
async def test_cancelled_cleanup_retains_current_and_unvisited_providers() -> None:
    first = MagicMock()
    second = MagicMock()
    third = MagicMock()
    second_started = asyncio.Event()
    second_attempts = 0

    async def cleanup_second() -> None:
        nonlocal second_attempts
        second_attempts += 1
        if second_attempts == 1:
            second_started.set()
            await asyncio.Event().wait()

    first.cleanup = AsyncMock()
    second.cleanup = AsyncMock(side_effect=cleanup_second)
    third.cleanup = AsyncMock()
    runtime = ProviderRuntime(
        _make_settings(),
        {"first": first, "second": second, "third": third},
    )
    cleanup_task = asyncio.create_task(runtime.cleanup())
    await second_started.wait()

    cleanup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert runtime.is_cached("first") is False
    assert runtime.is_cached("second") is True
    assert runtime.is_cached("third") is True
    first.cleanup.assert_awaited_once_with()
    third.cleanup.assert_not_awaited()

    await runtime.cleanup()

    first.cleanup.assert_awaited_once_with()
    assert second.cleanup.await_count == 2
    third.cleanup.assert_awaited_once_with()
    assert runtime.is_cached("first") is False
    assert runtime.is_cached("second") is False
    assert runtime.is_cached("third") is False


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_exceptiongroup_on_multiple_failures() -> None:
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("a"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock(side_effect=RuntimeError("b"))
    runtime = ProviderRuntime(_make_settings(), {"x": p1, "y": p2})

    with pytest.raises(ExceptionGroup) as exc_info:
        await runtime.cleanup()

    assert len(exc_info.value.exceptions) == 2
    assert runtime.is_cached("x")
    assert runtime.is_cached("y")

    p1.cleanup = AsyncMock()
    p2.cleanup = AsyncMock()
    await runtime.cleanup()

    assert not runtime.is_cached("x")
    assert not runtime.is_cached("y")

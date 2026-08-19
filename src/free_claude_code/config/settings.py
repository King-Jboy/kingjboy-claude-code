"""Flat application settings schema loaded by Pydantic Settings."""

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# ``parse_env_vars`` applies this source's own case, empty, and none-string
# options to whatever the file yielded. Reimplementing it here would drift from
# the version installed, so it is imported and only the read beneath it is
# replaced. `test_the_literal_source_still_matches_pydantic_settings` fails
# loudly if a pydantic-settings upgrade moves this or the hook below it.
from pydantic_settings.sources.providers.dotenv import parse_env_vars

from .api_keys import parse_api_key_list
from .constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from .env_files import (
    ANTHROPIC_AUTH_TOKEN_ENV,
    env_file_override,
    read_dotenv_file,
    settings_env_files,
)
from .model_refs import ModelCatalogScope, parse_model_ref_list
from .nim import NimSettings
from .provider_catalog import SUPPORTED_PROVIDER_IDS
from .reasoning import ReasoningPreference


class LiteralDotEnvSettingsSource(DotEnvSettingsSource):
    """Load env files as written, without POSIX ``${VAR}`` expansion.

    pydantic-settings reads dotenv files through ``dotenv_values``, which
    expands ``${NAME}`` in every value and has no escape for a literal one. An
    FCC env file is data the Admin UI rewrites wholesale, so a saved value
    containing ``${`` reached the runtime expanded -- or emptied, when the name
    was undefined -- and the proxy ran on a credential nobody entered. The
    Admin UI's own reads go through
    :func:`free_claude_code.config.env_files.read_dotenv_file`; this is the
    same read for the settings load, so both agree with the file.
    """

    @classmethod
    def replacing(cls, source: DotEnvSettingsSource) -> LiteralDotEnvSettingsSource:
        """Return a literal-reading twin of an already-resolved dotenv source.

        The source pydantic builds carries what the caller passed to
        ``Settings(...)`` -- ``_env_file`` above all -- so a replacement built
        from ``settings_cls`` alone would quietly read a different file.
        Options are forwarded rather than copied afterwards because the base
        class reads the files during ``__init__``.
        ``test_the_literal_source_forwards_every_dotenv_option`` fails if
        pydantic-settings grows one this misses.
        """

        return cls(
            source.settings_cls,
            env_file=source.env_file,
            env_file_encoding=source.env_file_encoding,
            dotenv_filtering=source.dotenv_filtering,
            case_sensitive=source.case_sensitive,
            env_prefix=source.env_prefix,
            env_prefix_target=source.env_prefix_target,
            env_nested_delimiter=source.env_nested_delimiter,
            env_nested_max_split=source.env_nested_max_split,
            env_ignore_empty=source.env_ignore_empty,
            env_parse_none_str=source.env_parse_none_str,
            env_parse_enums=source.env_parse_enums,
        )

    def _read_env_file(self, file_path: Path) -> Mapping[str, str | None]:
        return parse_env_vars(
            read_dotenv_file(file_path, encoding=self.env_file_encoding or "utf-8"),
            self.case_sensitive,
            self.env_ignore_empty,
            self.env_parse_none_str,
        )


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== OpenRouter Config ====================
    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    # Optional pool of interchangeable keys as a JSON list; overrides the single
    # key above. Stored raw so the Admin textarea round-trips exactly what the
    # user typed; ``config.api_keys`` owns parsing.
    open_router_api_keys: str = Field(
        default="", validation_alias="OPENROUTER_API_KEYS"
    )
    # Usage budget per pooled key per 24h window. OpenRouter's free tier caps
    # each key daily; 0 disables local metering.
    open_router_key_usage_limit: int = Field(
        default=1000, validation_alias="OPENROUTER_KEY_USAGE_LIMIT"
    )

    # ==================== DeepSeek Config ====================
    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")

    # ==================== Kimi Config ====================
    kimi_api_key: str = Field(default="", validation_alias="KIMI_API_KEY")

    # ==================== Hugging Face Inference Providers ====================
    huggingface_api_key: str = Field(default="", validation_alias="HUGGINGFACE_API_KEY")

    # ==================== Z.ai Config ====================
    zai_api_key: str = Field(default="", validation_alias="ZAI_API_KEY")

    # ==================== Groq (OpenAI-compatible) ====================
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord" | "none"
    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )
    messaging_rate_limit: int = Field(
        default=1, validation_alias="MESSAGING_RATE_LIMIT"
    )
    messaging_rate_window: float = Field(
        default=1.0, validation_alias="MESSAGING_RATE_WINDOW"
    )

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: str = ""
    # Optional pool of interchangeable keys as a JSON list; overrides the single
    # key above. See ``open_router_api_keys`` for why this stays a raw string.
    nvidia_nim_api_keys: str = Field(default="", validation_alias="NVIDIA_NIM_API_KEYS")
    # NIM's free tier is rate-limited per minute rather than by a consumable
    # budget, so no local usage metering by default; set a positive value to
    # self-impose a per-key budget anyway.
    nvidia_nim_key_usage_limit: int = Field(
        default=0, validation_alias="NVIDIA_NIM_KEY_USAGE_LIMIT"
    )

    # ==================== LM Studio Config ====================
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )

    # ==================== Ollama Config ====================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model (fallback)
    # Format: provider_type/model/name
    model: str = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"

    # Per-model overrides (optional, falls back to MODEL)
    # Each can use a different provider
    model_fable: str | None = Field(default=None, validation_alias="MODEL_FABLE")
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    # Context window advertised to launched client CLIs. Providers rarely
    # publish a per-model context length (NVIDIA NIM's /v1/models carries only
    # id/object/created/owned_by), so it cannot be discovered at runtime.
    # Blank resolves it from the routed model's row in ~/.fcc/context.md, so
    # changing MODEL changes the window; a value here overrides that lookup.
    client_context_window: int | None = Field(
        default=None,
        validation_alias="CLIENT_CONTEXT_WINDOW",
        gt=0,
    )

    # Whether client and admin model lists include every discovered provider
    # model or only the configured routes and the pinned list.
    model_catalog_scope: ModelCatalogScope = Field(
        default=ModelCatalogScope.ALL,
        validation_alias="MODEL_CATALOG_SCOPE",
    )

    # Extra provider/model refs to keep in the model lists, as a JSON list.
    # Stored raw so the Admin textarea round-trips what the user typed;
    # ``pinned_model_refs`` owns parsing.
    pinned_models: str = Field(default="", validation_alias="PINNED_MODELS")

    # ==================== Per-Provider Proxy ====================
    openai_proxy: str = Field(default="", validation_alias="OPENAI_PROXY")
    nvidia_nim_proxy: str = Field(default="", validation_alias="NVIDIA_NIM_PROXY")
    open_router_proxy: str = Field(default="", validation_alias="OPENROUTER_PROXY")
    lmstudio_proxy: str = Field(default="", validation_alias="LMSTUDIO_PROXY")
    kimi_proxy: str = Field(default="", validation_alias="KIMI_PROXY")
    huggingface_proxy: str = Field(default="", validation_alias="HUGGINGFACE_PROXY")
    zai_proxy: str = Field(default="", validation_alias="ZAI_PROXY")
    groq_proxy: str = Field(default="", validation_alias="GROQ_PROXY")
    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    # Fraction of a provider's published quota held back as headroom. We count a
    # request when we send it; the provider counts it on arrival, so latency and
    # clock skew can push our last request of a window into the provider's next
    # one. The cushion also covers the same key being used outside this proxy.
    provider_rate_margin: float = Field(
        default=0.05, validation_alias="PROVIDER_RATE_MARGIN"
    )
    # Ceiling on simultaneous upstream requests for a pooled provider. Quota is
    # per key and multiplies with the pool, but concurrency is bounded by local
    # sockets and event-loop work. Streaming responses stay open for tens of
    # seconds, so too low a ceiling - not the rate limit - becomes the real
    # throughput bound and queues callers until they time out.
    provider_max_pooled_concurrency: int = Field(
        default=64, validation_alias="PROVIDER_MAX_POOLED_CONCURRENCY"
    )
    reasoning_policy: ReasoningPreference = Field(
        default=ReasoningPreference.CLIENT,
        validation_alias="REASONING_POLICY",
    )
    reasoning_fable: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_FABLE",
    )
    reasoning_opus: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_OPUS",
    )
    reasoning_sonnet: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_SONNET",
    )
    reasoning_haiku: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_HAIKU",
    )

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=HTTP_CONNECT_TIMEOUT_DEFAULT,
        validation_alias="HTTP_CONNECT_TIMEOUT",
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== Local web server tools (web_search / web_fetch) ====================
    # On by default to match Claude Code's normal web-tool availability.
    enable_web_server_tools: bool = Field(
        default=True, validation_alias="ENABLE_WEB_SERVER_TOOLS"
    )
    # Comma-separated URL schemes allowed for web_fetch (default: http,https).
    web_fetch_allowed_schemes: str = Field(
        default="http,https", validation_alias="WEB_FETCH_ALLOWED_SCHEMES"
    )
    # When true, skip private/loopback/link-local IP blocking for web_fetch (lab only).
    web_fetch_allow_private_networks: bool = Field(
        default=False, validation_alias="WEB_FETCH_ALLOW_PRIVATE_NETWORKS"
    )

    # ==================== Debug / diagnostic logging (avoid sensitive content) ====================
    # Minimum log level for the JSON file sink (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    # When false (default), API and SSE helpers log only metadata (counts, lengths, ids).
    log_raw_api_payloads: bool = Field(
        default=False, validation_alias="LOG_RAW_API_PAYLOADS"
    )
    log_raw_sse_events: bool = Field(
        default=False, validation_alias="LOG_RAW_SSE_EVENTS"
    )
    # When false (default), unhandled exceptions log only type + route metadata (no message/traceback).
    log_api_error_tracebacks: bool = Field(
        default=False, validation_alias="LOG_API_ERROR_TRACEBACKS"
    )
    # When false (default), messaging logs omit text/transcription previews (metadata only).
    log_raw_messaging_content: bool = Field(
        default=False, validation_alias="LOG_RAW_MESSAGING_CONTENT"
    )
    # When true, log full Claude CLI stderr, non-JSON lines, and parser error text.
    log_raw_cli_diagnostics: bool = Field(
        default=False, validation_alias="LOG_RAW_CLI_DIAGNOSTICS"
    )
    # When true, log exception text / CLI error strings in messaging (may leak user content).
    log_messaging_error_details: bool = Field(
        default=False, validation_alias="LOG_MESSAGING_ERROR_DETAILS"
    )
    debug_platform_edits: bool = Field(
        default=False, validation_alias="DEBUG_PLATFORM_EDITS"
    )
    debug_subagent_stack: bool = Field(
        default=False, validation_alias="DEBUG_SUBAGENT_STACK"
    )

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Voice Note Transcription ====================
    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    # Device: "cpu" | "cuda" | "nvidia_nim"
    # - "cpu"/"cuda": local Whisper (requires voice_local extra: uv sync --extra voice_local)
    # - "nvidia_nim": NVIDIA NIM Whisper API (requires voice extra: uv sync --extra voice)
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    # Whisper model ID or short name (for local Whisper) or NVIDIA NIM model (for nvidia_nim)
    # Local Whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
    # NVIDIA NIM: "nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3", etc.
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")
    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    telegram_proxy_url: str = Field(default="", validation_alias="TELEGRAM_PROXY_URL")
    discord_bot_token: str | None = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: str | None = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    allowed_dir: str = ""
    max_message_log_entries_per_chat: int | None = Field(
        default=None, validation_alias="MAX_MESSAGE_LOG_ENTRIES_PER_CHAT"
    )

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    open_admin_browser: bool = Field(default=True, validation_alias="FCC_OPEN_BROWSER")
    # Optional proxy bearer token protecting public API endpoints.
    # Set via env `ANTHROPIC_AUTH_TOKEN`. When empty, no auth is required.
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )

    # ==================== Browser Shell Bridge ====================
    # Off by default, and deliberately not implied by installing the Chrome
    # extension: this is the switch that lets a browser run commands on this
    # machine. Turning it on is a separate, deliberate act.
    browser_shell_enabled: bool = Field(
        default=False, validation_alias="BROWSER_SHELL_ENABLED"
    )
    # Commands may only run inside this directory tree. Blank means the home
    # directory, which is broad; point it at a project root in practice.
    browser_shell_root: str = Field(default="", validation_alias="BROWSER_SHELL_ROOT")

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        "model_fable",
        "model_opus",
        "model_sonnet",
        "model_haiku",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("max_message_log_entries_per_chat", mode="before")
    @classmethod
    def parse_optional_log_cap(cls, v: Any) -> Any:
        if v == "" or v is None:
            return None
        return v

    @field_validator("client_context_window", mode="before")
    @classmethod
    def parse_optional_context_window(cls, v: Any) -> Any:
        """Treat a blank field as "resolve it from context.md"."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("nvidia_nim_api_keys", "open_router_api_keys")
    @classmethod
    def validate_api_key_pool(cls, value: str, info: ValidationInfo) -> str:
        """Fail startup on a malformed pool rather than quietly serving one key."""
        parse_api_key_list(value, env_name=(info.field_name or "").upper())
        return value

    @field_validator("pinned_models")
    @classmethod
    def validate_pinned_models(cls, value: str) -> str:
        """Fail startup on a malformed pinned list rather than dropping it."""
        parse_model_ref_list(value, env_name="PINNED_MODELS")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, got {v!r}")
        return upper

    @field_validator("reasoning_policy")
    @classmethod
    def validate_root_reasoning_policy(
        cls, value: ReasoningPreference
    ) -> ReasoningPreference:
        if value is ReasoningPreference.INHERIT:
            raise ValueError("REASONING_POLICY cannot inherit")
        return value

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("messaging_platform")
    @classmethod
    def validate_messaging_platform(cls, v: str) -> str:
        if v not in ("telegram", "discord", "none"):
            raise ValueError(
                f"messaging_platform must be 'telegram', 'discord', or 'none', got {v!r}"
            )
        return v

    @field_validator("messaging_rate_limit")
    @classmethod
    def validate_messaging_rate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("messaging_rate_limit must be > 0")
        return v

    @field_validator("messaging_rate_window")
    @classmethod
    def validate_messaging_rate_window(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("messaging_rate_window must be > 0")
        return float(v)

    @field_validator("web_fetch_allowed_schemes")
    @classmethod
    def validate_web_fetch_allowed_schemes(cls, v: str) -> str:
        schemes = [part.strip().lower() for part in v.split(",") if part.strip()]
        if not schemes:
            raise ValueError("web_fetch_allowed_schemes must list at least one scheme")
        for scheme in schemes:
            if not scheme.isascii() or not scheme.isalpha():
                raise ValueError(
                    f"Invalid URL scheme in web_fetch_allowed_schemes: {scheme!r}"
                )
        return ",".join(schemes)

    @field_validator(
        "model", "model_fable", "model_opus", "model_sonnet", "model_haiku"
    )
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(SUPPORTED_PROVIDER_IDS)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in SUPPORTED_PROVIDER_IDS:
            supported = ", ".join(f"'{p}'" for p in SUPPORTED_PROVIDER_IDS)
            raise ValueError(f"Invalid provider: '{provider}'. Supported: {supported}")
        return v

    @model_validator(mode="after")
    def check_nvidia_nim_api_key(self) -> Settings:
        if (
            self.voice_note_enabled
            and self.whisper_device == "nvidia_nim"
            and not self.nvidia_nim_api_key.strip()
        ):
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when WHISPER_DEVICE is 'nvidia_nim'. "
                "Set it in your .env file."
            )
        return self

    @model_validator(mode="after")
    def prefer_dotenv_anthropic_auth_token(self) -> Settings:
        """Let explicit .env auth config override stale shell/client tokens."""
        dotenv_value = env_file_override(self.model_config, ANTHROPIC_AUTH_TOKEN_ENV)
        if dotenv_value is not None:
            self.anthropic_auth_token = dotenv_value
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read dotenv files literally, keeping the default source precedence."""

        if not isinstance(dotenv_settings, DotEnvSettingsSource):
            return (init_settings, env_settings, dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            LiteralDotEnvSettingsSource.replacing(dotenv_settings),
            file_secret_settings,
        )

    model_config = SettingsConfigDict(
        env_file=settings_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

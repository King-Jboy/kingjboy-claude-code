# Free Claude Code (`v6.20.92`) — Complete Codebase Summary & Module Reference

## 1. Executive Overview

**Free Claude Code (`fcc`)** is an async Python 3.14+ FastAPI local proxy and messaging bridge that connects agentic coding CLIs (**Claude Code**, **Codex CLI**, **Codex Desktop App**, **Pi**, **Hermes**, **DSH**, **Grok CLI**) to 12+ OpenAI-compatible AI providers (**NVIDIA NIM**, **OpenAI / ChatGPT Codex Subscription**, **OpenRouter**, **Google Gemini**, **DeepSeek**, **Groq**, **Cerebras**, **Mistral**, **SambaNova**, **Together AI**, **Fireworks AI**, **xAI**, **LM Studio**, **Llama.cpp**), while optionally bridging **Discord** and **Telegram** conversations into managed local Claude Code subprocesses.

### Enforced Layered Package Dependency Graph

The codebase enforces a strict DAG (verified by `tests/contracts/test_import_boundaries.py`):

```text
config/                      (Layer 0 — Leaf: settings, provider catalog, model refs, admin persistence)
  ^
core/                        (Layer 1 — Wire protocols, Anthropic <-> OpenAI conversion, SSE ledgers, rate limiters)
  ^
  +-- application/           (Layer 2a — ModelRouter, ReasoningPolicy, ProviderExecutor, structural ports)
  +-- providers/             (Layer 2b — Admission controller, API key pool, stream recovery, vendor adapters)
  +-- messaging/             (Layer 2c — Conversation tree queue, turn intake, node runner, Discord/Telegram bridges)
  ^
api/                         (Layer 3 — FastAPI routes, handlers, fast-path optimizations, local WebSearch/WebFetch)
  ^
cli/                         (Layer 4 — Managed Claude subprocess pool, agent CLI launchers, OAuth helpers)
  ^
runtime/                     (Layer 5 — Composition root, ProviderManager, uvicorn lifecycle, startup diagnostics)
```

---

## 2. Package-by-Package & Module-by-Module Breakdown

### 2.1 `src/free_claude_code/config/` — Configuration, Catalog & Admin Persistence

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `settings.py` | `Settings`, `LiteralDotEnvSettingsSource`, `get_settings()`, `reload_settings()` | Flat Pydantic `BaseSettings` schema reading `.env` files literally (preventing JSON parsing of comma-separated API keys) and coercing empty strings to field defaults. Exposes `.nim` (`NimSettings`) and `.bot` (`BotSettings`) cached sub-views. |
| `provider_catalog.py` | `ProviderDescriptor`, `PROVIDER_CATALOG`, `get_provider_descriptor()`, `canonical_provider_id()` | Authoritative metadata registry for all 14 supported providers: base URLs, env var names, rate-limit defaults, capability flags (`supports_key_pool`, `supports_reasoning`, `local_default_api_key`). |
| `model_refs.py` | `ModelReference`, `parse_model_ref()`, `resolve_model_alias()`, `split_provider_model()` | Parses `provider/model` strings, resolves user-defined model aliases, and validates provider prefixes against the catalog. |
| `context_windows.py` & `curated_contexts.py` | `resolve_context_window()`, `CURATED_CONTEXT_WINDOWS` | Resolves accurate context-window token limits per model family for Claude Code auto-compaction (`CLAUDE_CODE_AUTO_COMPACT_WINDOW`) and `/v1/models` metadata. |
| `nim.py` & `bot.py` | `NimSettings`, `BotSettings` | Grouped immutable views for upstream generation hyperparameters (temperature, top_p, thinking budgets) and messaging bot configuration (Discord/Telegram tokens, voice note transcription, workspace paths). |
| `admin/manifest.py` | `ADMIN_CONFIG_MANIFEST`, `AdminFieldSpec` | Declarative schema for the loopback Admin Web UI (`/ui`), classifying every `.env` setting by section, input type, secret redaction policy, and hot-reload vs. restart requirement. |
| `admin/persistence.py` | `read_env_document()`, `apply_admin_patch()`, `atomic_write_env()` | Round-trip `.env` file parser and atomic writer that updates configuration keys in-place while preserving user comments, blank lines, and ordering. |

---

### 2.2 `src/free_claude_code/core/` — Wire Protocols, Conversion & Rate Limiting

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `anthropic/models.py` | `MessagesRequest`, `Message`, `ContentBlock`, `Tool`, `ThinkingConfig`, `TokenCountRequest` | Pydantic v2 wire models for the Anthropic `/v1/messages` and `/v1/messages/count_tokens` APIs. |
| `anthropic/conversion.py` | `convert_messages_to_openai()`, `convert_tools_to_openai()`, `extract_system_text()` | Translates Anthropic multi-block messages (text, thinking, base64/URL images, `tool_use`, `tool_result`) and tool JSON schemas into OpenAI Chat Completions format. Pairs orphan `tool_use`/`tool_result` blocks so strict upstreams (MiniMax, DeepSeek, Mistral) do not reject history. |
| `anthropic/openai_tool_names.py` | `OpenAIToolNameCodec` | Reversible, collision-free codec that maps Anthropic tool names >64 chars (e.g. long MCP server tool names) into deterministic SHA-256 suffixed 64-char OpenAI function names and restores them on output. |
| `anthropic/streaming/ledger.py` | `AnthropicStreamLedger`, `ToolBlockRecord` | Single source of truth for an active Anthropic SSE response stream. Tracks `message_start`, active thinking/text/tool content block indices, accumulated text/reasoning, and token estimation. |
| `anthropic/streaming/emitter.py` | `AnthropicSSEEmitter`, `format_sse_event()`, `anthropic_ping_frame()` | Emits spec-compliant Anthropic SSE frames (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, `error`) while mutating `AnthropicStreamLedger`. |
| `anthropic/streaming/recovery.py` | `repair_incomplete_tool_json()`, `all_emitted_tools_complete()`, `build_continuation_messages()` | Parses and repairs truncated tool-call JSON fragments when an upstream stream drops mid-generation, and constructs synthetic assistant-prefill continuation prompts. |
| `anthropic/tokens.py` | `count_tokens_for_anthropic_request()`, `estimate_tokens()` | Fast local token counter using `tiktoken` (`cl100k_base`) with structural overhead estimation for tools, system blocks, and base64 images. |
| `openai_responses/adapter.py` | `adapt_responses_request()`, `build_responses_json_from_anthropic_sse()` | Translates OpenAI `/v1/responses` requests (used by Codex CLI & Codex Desktop) into internal `MessagesRequest` objects and converts Anthropic SSE streams back into `/v1/responses` JSON or SSE events. |
| `openai_responses/stream.py` | `AnthropicToResponsesStreamAssembler` | Stateful streaming translator converting Anthropic SSE events into OpenAI Responses API events (`response.created`, `response.output_item.added`, `response.output_text.delta`, `response.function_call_arguments.delta`, `response.completed`). |
| `rate_limit.py` | `StrictSlidingWindowLimiter` | Lock-protected sliding-window rate limiter supporting blocking `acquire()`, conditional `acquire_if()`, non-blocking `try_acquire()`, dynamic `set_rate_limit()`, and `headroom()` / `next_available_in()` inspection. |
| `failures.py` | `ExecutionFailure`, `FailureKind`, `classify_provider_failure()` | Provider-neutral failure taxonomy (`RATE_LIMIT`, `AUTHENTICATION`, `PERMISSION_DENIED`, `INVALID_REQUEST`, `CONTEXT_LENGTH`, `TIMEOUT`, `UPSTREAM_OVERLOAD`, `TRANSIENT_STREAM`) mapped to Anthropic HTTP status codes and error types. |
| `json_utils.py` | `fast_json_loads()`, `fast_json_dumps()` | High-speed JSON serialization/deserialization backed by `orjson`. |

---

### 2.3 `src/free_claude_code/application/` — Use-Case & Routing Layer

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `routing.py` | `ModelRouter`, `RoutedModelRequest` | Resolves incoming client model names (e.g. `claude-sonnet-4-5`, `gpt-5-codex`, or custom aliases) to target `(provider_id, model_name)` pairs, applying subagent model overrides when subagent system markers are detected. |
| `reasoning.py` | `ReasoningPolicy`, `resolve_reasoning_policy()` | Computes effective reasoning/thinking configuration from request `thinking` blocks, `output_config.effort`, model capabilities, and `.env` thinking overrides. |
| `execution.py` | `ProviderExecutor`, `_stream_with_progress_timeout()` | Wraps provider stream execution with a per-chunk progress watchdog (`stream_progress_timeout`) and converts unexpected exceptions into typed `ExecutionFailure` instances. |
| `ports.py` | `ProviderPort`, `ProviderResolverPort` | Structural `Protocol` interfaces decoupling the API and application layers from concrete provider implementations. |

---

### 2.4 `src/free_claude_code/providers/` — Upstream Adapters & Resilience

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `admission.py` | `ProviderAdmissionController`, `ProviderRetrySession`, `ProviderAttempt`, `_RecoveryEpisode` | Coordinates provider concurrency bulkheads (`asyncio.Semaphore`), sliding-window rate limits, per-session retry budgets, and per-model `_RecoveryEpisode` backoff gates so a single 429/503 pauses sibling requests for the same model instead of triggering retry storms. |
| `key_pool.py` | `ApiKeyPool`, `_KeyState` | Thread-safe LRU credential rotator with independent per-key sliding-window rate limiters, 60s rate-limit cooldowns, and 3-strike auth failure retirement. Dynamically updates the provider admission rate limit as keys enter/leave cooldown. |
| `stream_recovery.py` | `RecoveryController`, `RecoveryHoldbackBuffer`, `HoldbackSignal` | Buffers early SSE output behind a commit boundary (`0.75s` / `64 KiB`) so early upstream failures can be transparently retried before HTTP 200 headers commit, and coordinates mid-stream continuation or tool JSON repair after commit. |
| `openai_chat/provider.py` | `OpenAIChatProvider`, `_OpenAIChatStreamRunner`, `create_openai_chat_provider()` | Full-featured OpenAI Chat Completions streaming engine supporting sequential & hedged multi-key opening, `<think>` tag extraction, `<function=...>` XML fallback tool parsing, heuristic tool repair, schema downgrade retries, and mid-stream continuation. |
| `openai_chat/profiles.py` | `OpenAIChatProfile`, `PROFILES` | Declarative vendor profiles for Groq, Cerebras, Mistral, SambaNova, Together, Fireworks, xAI, DeepSeek, OpenRouter, LM Studio, and Llama.cpp (configuring reasoning encoders, unsupported parameters, and auth rotation behavior). |
| `nvidia_nim/` | `NvidiaNimProvider`, `NimToolNormalizer`, `NimCapabilityCache` | Specialized NIM adapter handling MiniMax M2.1/M2.5 native tool stream normalization, parameter alias recovery, and per-model capability downgrade caching (`thinking`, `parallel_tool_calls`, `reasoning_effort`). |
| `openai_codex/` | `OpenAICodexProvider`, `CodexAuthSource` | Connects directly to ChatGPT Plus/Pro/Team subscriptions via OAuth tokens (`~/.codex/auth.json`) over the OpenAI Responses API endpoint (`chatgpt.com/backend-api/codex/responses`). |
| `gemini/`, `google_openai/`, `deepseek/`, `open_router/`, `lmstudio/` | Vendor-specific provider subclasses | Handle Gemini thought signatures & schema sanitization, DeepSeek `reasoning_content` history replay, OpenRouter provider routing headers, and LM Studio local model discovery. |

---

### 2.5 `src/free_claude_code/api/` — FastAPI HTTP Layer & Local Web Tools

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `app.py` | `create_app()`, `RequestCorrelationMiddleware` | FastAPI factory, lifespan wiring, pure ASGI correlation middleware, and disconnect-aware cancellation propagation. |
| `routes.py` | `/v1/messages`, `/v1/responses`, `/v1/models`, `/v1/messages/count_tokens`, `/health`, `/ready` | Primary API endpoints serving both Anthropic Messages clients (Claude Code) and OpenAI Responses clients (Codex CLI/App). |
| `admin_routes.py` | `/api/admin/*`, `/ui` | Loopback-only (`127.0.0.1` / `::1`) Admin API for live configuration inspection, `.env` editing, provider health checks, and Codex OAuth login flows. |
| `handlers/messages.py` | `MessagesHandler` | Orchestrates `/v1/messages`: runs fast-path optimization checks, intercepts server-side `WebSearch`/`WebFetch` tools, routes models, opens `ManagedStreamingResponse`, and collects non-streaming responses. |
| `optimization_handlers.py` & `detection.py` | `try_handle_optimizations()`, `is_quota_check_request()`, `is_prefix_detection_request()`, `is_title_generation_request()`, `is_suggestion_mode_request()`, `is_filepath_extraction_request()` | Detects and locally short-circuits auxiliary Claude Code background requests (quota probes, bash command prefix extraction, conversation title generation, prompt suggestions, file path extraction) to save upstream API quota and latency. |
| `response_streams.py` | `ManagedStreamingResponse`, `open_managed_stream()` | Prefetches the first committed SSE chunk *before* sending HTTP 200 headers so pre-commit `ExecutionFailure` exceptions return proper non-200 JSON error responses. |
| `web_tools/search.py` & `web_tools/fetch.py` | `execute_web_search()`, `execute_web_fetch()`, `validate_public_url()` | Executes local DuckDuckGo HTML searches and SSRF-hardened web page fetches (blocking RFC1918/loopback/link-local/cloud-metadata IPs on every redirect hop) when the upstream model invokes `web_search` or `web_fetch`. |

---

### 2.6 `src/free_claude_code/messaging/` — Discord & Telegram Bridge

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `workflow.py` | `MessagingWorkflow` | Top-level coordinator uniting turn intake, conversation tree queueing, node execution, crash snapshot recovery, and `/stop`, `/clear`, `/model`, `/stats` bot commands. |
| `trees/queue_manager.py` & `trees/data.py` | `TreeQueueManager`, `ConversationTree`, `MessageNode` | Manages branching reply trees per chat scope. Each root conversation has a FIFO execution queue; replying to any message in a tree forks the Claude session from that parent node's `session_id` and queues within the same tree. |
| `node_runner.py` & `node_event_pipeline.py` | `MessagingNodeRunner`, `process_parsed_cli_event()` | Claims a queued node, acquires a `ManagedClaudeSession`, streams JSON-lines stdout events into a `TranscriptBuffer`, and throttles live message edits to Discord/Telegram via `ThrottledTranscriptEditor`. |
| `platforms/discord/` & `platforms/telegram/` | `DiscordPlatform`, `TelegramPlatform`, `OutboxWorker` | Platform-specific event listeners, rate-limit-aware message outbox workers, and voice-note downloaders (`transcription.py` via Whisper/OpenAI audio). |

---

### 2.7 `src/free_claude_code/cli/` & `src/free_claude_code/runtime/` — Launchers & Runtime

| Module | Key Classes & Functions | Responsibility |
| :--- | :--- | :--- |
| `cli/managed/session.py` & `manager.py` | `ManagedClaudeSession`, `ManagedClaudeSessionManager` | Spawns persistent `claude -p --output-format stream-json` subprocesses with bounded 256 KiB stdout/stderr pipe draining, benign stderr filtering, and cross-platform PID tree cleanup (`cli/process_registry.py`). |
| `cli/launchers/` | `claude.py`, `codex.py`, `pi.py`, `hermes.py`, `dsh.py`, `grok.py`, `desktop.py` | Entrypoints for `fcc-claude`, `fcc-codex`, `fcc-pi`, `fcc-hermes`, `fcc-dsh`, `fcc-grok`, and `fcc-desktop`. Automatically start or reuse a local `fcc-server` daemon and inject proxy environment variables into the target coding CLI. |
| `runtime/bootstrap.py` & `runtime/application.py` | `RuntimeBootstrap`, `RuntimeApplication` | Composition root managing server startup, provider warmup, messaging platform lifecycle, hot reload on `.env` changes, and graceful shutdown. |
| `runtime/provider_manager.py` | `ProviderManager` | Lazy-initializes and caches `ProviderPort` instances per provider, manages shared `ProviderAdmissionController`s, and refreshes dynamic `/v1/models` catalogs. |

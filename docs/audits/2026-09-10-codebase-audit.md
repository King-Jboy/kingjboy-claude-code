# Codebase Audit

## Scope and evidence

Reviewed commit `07190a8`: 455 Python files and 348 collected tests. Evidence consists of source tracing, direct reproductions where noted, focused tests, full-project Ruff/type checks, and the production FCC journal for the NIM incident. Findings below are concrete defects or configuration hazards, not general code-quality suggestions.

The live NIM incident was not key rotation: NIM returned HTTP 200, then its SSE stream went silent beyond the configured 30-second read timeout. The live timeout has been raised to 300 seconds and health was verified.

## Findings

| Priority | Finding | Consequence |
| --- | --- | --- |
| P1 | Admin template defaults drift from Settings defaults. | First Admin save can public-bind FCC, change auth, and alter rate/timeouts. |
| P1 | A saturated LRU key blocks a key with immediate RPM headroom. | Avoidable pool latency; not seamless rotation. |
| P1 | NIM request-level 403 cools every key. | One invalid request can disable the full pool for five minutes. |
| P1 | Mid-stream recovery bypasses admission backoff. | Immediate retry storms after partial output. |
| P1 | Automatic web search buffers an unlimited stream. | Unbounded memory and indefinite first-byte delay. |
| P1 | Codex launcher combines incompatible provider authentication modes. | `fcc-codex` can fail before sending a request when durable FCC auth is configured. |
| P1, conditional | Local-only Admin assumes no loopback reverse proxy. | Unsafe proxy deployment exposes unauthenticated Admin actions. |
| P2 | Hedge opens are not physically admission-accounted. | Nearly 2x upstream concurrency/RPM with hedging enabled. |
| P2 | Aggregate admission does not shrink with cooled keys. | Requests queue inside a degraded pool. |
| P2 | NIM capability fallback is provider-global. | One model can permanently downgrade other models until restart. |
| P2 | Responses SSE splitting rejects CRLF framing. | Valid streams can be malformed or lost. |
| P2 | Responses adapter turns grouped cancellation into a provider failure. | Teardown can emit a false `response.failed` event. |
| P2 | Orphan Responses tool output reaches providers. | Client mistake becomes an upstream failure instead of 400. |
| P2 | Admin accepts values that fail on first provider use. | A successful-looking save can cause later 500s. |
| P2 | Foreground client launchers do not force-stop children. | Ctrl-C may hang indefinitely. |

## Detailed findings

### P1 — Admin default drift

`target_values_with_updates()` seeds a missing managed `.env` from the template rather than `Settings`. Those defaults conflict: template `HOST=0.0.0.0` versus Settings loopback; template `ANTHROPIC_AUTH_TOKEN=freecc` versus empty; and template rate, timeout, and voice defaults differ. An unrelated first Admin save writes all values, and a restart changes runtime behavior.

Evidence: `config/admin/persistence.py:68-95,200-213`, `config/admin/sources.py:62-68`, `config/admin/manifest.py:319-348`, `config/settings.py:245-304,366-400`, `.env.example`.

Required test: first Admin save followed by restart must preserve each unedited Settings default.

### P1 — Saturated LRU blocks capacity-ready key

Selection checks cooldown/quota but not a key's sliding-window headroom. The sequential path selects the oldest key and waits for its limiter instead of trying a newer ready key. Direct reproduction with two keys: saturated A, ready B; the request selected A and waited 0.266 seconds.

Evidence: `providers/key_pool.py:280-288,377-383`.

Required test: an older saturated key and newer ready key must select the newer key immediately.

### P1 — NIM 403 can cool the full pool

Every `PermissionDeniedError` is classified as a key failure. The pool then walks all keys and applies a five-minute cooldown. NIM can use 403 for a model/request/policy refusal, so a forbidden request can sideline healthy credentials and make the next valid request fail pool-wide.

Evidence: `providers/key_pool.py:120-144,325-332,377-394`; `README.md:323`.

Required test: identical NIM 403 responses across a pool must return a request error without sidelining keys.

### P1 — Mid-stream recovery ignores admission backoff

The first output chunk marks an attempt successful. A later retryable error creates recovery streams without entering admission retry/backoff or its shared recovery episode. A partially streamed timeout can therefore retry immediately, ignoring `Retry-After` and exponential delay.

Evidence: `providers/openai_chat/provider.py:720-721,901-907,946-967,1210-1223`; the separately implemented OpenAI Codex Responses provider repeats the same pattern at `providers/openai_codex/provider.py:265-326`.

Required test: first-chunk-then-`ReadTimeout` or 429 recovery must respect admission delay.

### P1 — Automatic web search has no bounded decision buffer

The automatic-search path collects all provider chunks before emitting any public frame or deciding to search. There is no byte, event-count, or total-duration cap. A long or endless upstream decision stream can grow memory without bound and leave the client waiting indefinitely.

Evidence: `api/web_tools/automatic_search.py:42-50`.

Required test: oversized and never-ending streams must stop at a fixed bound.

### P1 — Codex launcher authentication-mode conflict

The launcher always passes `model_providers.fcc.env_key=FCC_CODEX_API_KEY`. A persistent Codex configuration may instead correctly use the command-based `[model_providers.fcc.auth]` setup. Current Codex rejects a provider that has both modes: `provider auth cannot be combined with env_key`. That means `fcc-codex` can fail at process startup, before it contacts FCC.

Evidence: `cli/launchers/codex.py:193-210`; `tests/cli/test_codex_model_catalog.py:214-269`. The full test suite reproduced it on the installed Codex CLI.

Required fix/test: select exactly one authentication mechanism, or intentionally replace/clear inherited provider auth before adding `env_key`; test both a blank and a durable command-auth Codex configuration against the supported Codex CLI.

### P1, conditional — Reverse-proxy Admin trust boundary

Admin access trusts the immediate TCP peer being loopback and has no Admin authentication. Direct loopback deployment is safe. A public reverse proxy connected through loopback makes all requests appear local unless that proxy is explicitly trusted and protected.

Evidence: `api/admin_routes.py:73-86`.

Required test: reverse-proxy topology must be denied without explicit trusted-proxy/Admin-auth configuration.

### P2 — Hedge attempts bypass physical accounting

One logical request acquires one admission permit, but hedging can open two physical upstream requests. With nonzero `KEY_HEDGE_DELAY_SECONDS`—1.5 seconds on the live server—actual upstream concurrency and RPM can approach double the configured limit.

Evidence: `providers/openai_chat/provider.py:444-453`, `providers/key_pool.py:435-467`.

### P2 — Admission never follows usable key count

Aggregate admission is calculated once from configured pool size. Its retuning method has no caller when keys cool or recover, so traffic is admitted at the old N-key rate and waits inside a degraded pool.

Evidence: `providers/runtime/factory.py:128-153`, `providers/admission.py:237-245`.

### P2 — NIM capability cache is global

A rejection flips provider-wide feature flags. Subsequent requests for every NIM model have reasoning/chat-template fields stripped until restart, although the rejection may be model-specific or transient.

Evidence: `providers/nvidia_nim/client.py:68-71,86-103,126-157`.

### P2 — CRLF Responses SSE is not split

The Responses adapter splits only on `\n\n`; valid SSE can be delimited by `\r\n\r\n`. The parser itself understands CRLF lines, so the delimiter causes the defect and multiple events can be buffered into one malformed event.

Evidence: `core/openai_responses/anthropic_sse.py:26-39`.

### P2 — Grouped cancellation becomes a false Responses failure

The Responses adapter catches `BaseExceptionGroup` after output begins but does not preserve an included `CancelledError`. It serializes the group as `response.failed`, unlike the outer streaming wrapper which explicitly treats grouped cancellation as control flow. A direct reproduction yielded `PROPAGATED=none` and `TERMINAL_FAILED=True` for an upstream `BaseExceptionGroup` containing only `asyncio.CancelledError`.

Evidence: `core/openai_responses/stream.py:45-57`; contrast `api/response_streams.py:224-231`. Existing cancellation coverage is only for the Anthropic wrapper: `tests/api/test_response_streams.py:219-239`.

Required test: a post-start cancellation-containing exception group through `/v1/responses` must propagate cancellation and emit no `response.failed` frame.

### P2 — Orphan tool output is not rejected

Responses input turns arbitrary tool output call IDs into Anthropic tool_result blocks without verifying a matching tool call. It also does not resolve `previous_response_id`, so malformed input reaches the provider rather than returning deterministic 400.

Evidence: `core/openai_responses/input.py:153-167`, `core/openai_responses/models.py:24`.

### P2 — Admin validates too late

`PROVIDER_MAX_CONCURRENCY=0` passes Settings/Admin validation and persists, but lazy provider construction later raises `ValueError`. Several related timeout/rate/concurrency settings also lack consistent finite positive bounds.

Evidence: `config/settings.py:245-247,290-304`, `config/admin/validation.py:20-35`, `providers/runtime/runtime.py:42-48`, `providers/runtime/factory.py:149-154`, `providers/admission.py:200-205`.

### P2 — Foreground Ctrl-C can hang

The launcher sends a best-effort termination signal then does an unbounded wait. POSIX termination sends SIGTERM only; an uncooperative child never receives forced termination, unlike the bounded managed-session path.

Evidence: `cli/launchers/common.py:97-101`, `cli/process_registry.py:62-67`, `cli/managed/session.py:285-294`.

## Verification record

- Full-project Ruff check: passed.
- Full-project type check: passed.
- Focused provider, pool, configuration, and strict-window tests: 57 passed.
- Complete suite: **1 failed, 3137 passed, 59 skipped** in 399.27 seconds. The failure is `tests/cli/test_codex_model_catalog.py::test_launcher_config_composes_with_persistent_codex_config`, reproduced against the installed Codex CLI; it is the authentication-mode conflict above.
- Live FCC health: passed after setting `HTTP_READ_TIMEOUT=300`.

The listed defects are not covered by existing tests. No production code was changed during this audit.

## Sources

1. Current source checkout at commit `07190a8`; paths and lines are cited inline.
2. FCC EC2 `fcc.service` journal and `~/.fcc/logs/server.log`, accessed 2026-09-10; used only for the NIM timeout timeline.

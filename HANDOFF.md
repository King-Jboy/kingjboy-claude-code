# Handoff — free-claude-code key pool & rate-limiting overhaul

**Repo:** `C:\Users\Maduabuna Josiah\documents\claude-code\free-claude-code`
**Version:** `6.7.0` → `6.9.1` (bumped incrementally across the work)
**State:** All work complete. **Uncommitted** working-tree changes across 17 files.
**Last verification:** all 5 CI checks green — `2998 passed, 53 skipped in 187s`.

---

## 1. What this work was

The project is a local proxy that connects coding agents (Claude Code, Codex) to
OpenAI-compatible providers. The user runs **10+ API keys per provider** against
**NVIDIA NIM** (40 RPM/key, no daily cap) and **OpenRouter** (20 RPM/key, daily cap
they explicitly do *not* want respected).

The goal, in the user's words: Claude Code is fragile and spams network errors
across subagent/workflow fan-outs, so the proxy must **never surface a rate limit**.
Keys should all be live from proxy start, rotation should be effectively instant,
and the aggregate RPM should be `per_key × key_count`.

**Two constraints the user set explicitly — do not violate these:**
- **No cross-provider fallback.** "my model should remain."
- **Neither provider enforces TPM** (tokens/min). Only RPM is modelled.

The user also asked for explanations in **simple, non-jargon English** (a
shop/cashier analogy was used, with problems labelled A–E). Keep that register.

---

## 2. Scope of change

Read the diff for specifics — `git diff` in the repo root. Do not re-derive it here.
The load-bearing files:

| File | Role |
|---|---|
| `src/free_claude_code/providers/key_pool.py` | The centre of the work. ~361 lines changed. |
| `src/free_claude_code/providers/runtime/config.py` | `resolve_rate_policy`, `rate_with_margin`, `operator_configured` |
| `src/free_claude_code/providers/runtime/factory.py` | `MAX_POOLED_CONCURRENCY` 20 → 64 |
| `src/free_claude_code/config/provider_catalog.py` | Per-provider `rate_limit` / `rate_window` |
| `src/free_claude_code/core/rate_limit.py` | `StrictSlidingWindowLimiter.set_rate_limit()` |
| `src/free_claude_code/providers/admission.py` | `ProviderAdmissionController.set_rate_limit()` |
| `src/free_claude_code/providers/openai_chat/provider.py` | `_retune_admission_rate` wiring |

Tests: `tests/providers/test_key_pool.py` (+287 lines, 54 tests in file),
`tests/config/test_api_keys.py`, `tests/providers/test_provider_runtime.py`.

---

## 3. Bugs fixed (all reproduced before fixing)

Round 1 — five confirmed defects:

- **(A) Unbounded retry loop from `Retry-After: 0`.** Repro made 2000 upstream calls
  in 3s and never terminated. Three causes compounded: `_cooldown_seconds` obeyed a
  stated reset verbatim *including zero*, so `_cool` did not actually cool;
  `run_key_local` never excluded HOP keys; the strike ladder only incremented on the
  guessed path. Fixed by treating a zero/elapsed reset as "no timing stated" and
  adding an unconditional attempt budget (`_MAX_ATTEMPTS_PER_KEY = 2`).
- **(A2)** Same loop reachable via a stale `x-ratelimit-reset` epoch under clock skew.
- **(B)** `_restore` clobbered a concurrent 600s cooldown, producing `-81677s`.
  Fixed with `_HealthRollback` compare-and-swap — a field is rolled back only while
  it still holds the value that refusal wrote.
- **(C)** `status()` reported `ready=2, soonest_ready_in=None` while `acquire()` blocked.
  Now measured through `available_in()`, the same way `acquire` measures it.
- **(D)** `messaging/ui_updates.py` used wall-clock `time.time()` for a 1s debounce →
  `time.monotonic()`.

Post-fix, all four repro cases terminate in ≤4 attempts with a real retryable 429.

## 4. Design problems A–E (the user's own labels — reuse them)

- **A — concurrency ceiling.** `MAX_POOLED_CONCURRENCY = 20` capped throughput
  regardless of key count: 20 concurrent ÷ ~20s streams ≈ 60 RPM actual against 380
  allowed. Now 64, configurable via `PROVIDER_MAX_POOLED_CONCURRENCY`.
- **B — one global rate limit for all providers.** `.env.example` shipped
  `PROVIDER_RATE_LIMIT=1`. Now per-provider values live in the catalog, with
  operator settings still winning.
- **C — frozen gate.** `pool_scale` was fixed at the *configured* key count, so the
  provider-wide gate kept admitting at full rate into a pool that had lost keys.
  Now `KeyPool(on_capacity_change=...)` → `_publish_capacity()` →
  `_retune_admission_rate()`. **Only retirement retunes it** — cooldowns are short
  and self-clearing and would make the gate flap.
- **D — the error storm.** `_pool_exhausted_failure(retryable=False)` handed every
  queued request a hard 429 at once. Self-imposed waits now never error, and the
  failure is `retryable=True`.
- **E — no queue.** Thundering herd with an arbitrary winner and real starvation.
  Now FIFO `_Waiter` tickets with a head-of-line exception (an older waiter that
  can't use any ready key must not block one that can), plus immediate wakeup
  handoff via `asyncio.Future`.

**C and D only work as a pair.** Once waiting stops producing errors, an
over-admitting gate no longer causes errors — it causes the queue to grow without
bound until Claude Code times out, which is *worse* than a clean error. The gate
throttling to what the pool can actually serve is what keeps that queue bounded.

## 5. Reusable keys

A `401` previously set `dead = True` **permanently until proxy restart**. Providers
answer 401 during their own auth outages and while a new key propagates, so one bad
moment cost a tenth of capacity for the life of the process.

Now: `retired_until` / `retirements` — an expiring quarantine at
`RETIREMENT_PROBE_SECONDS = 300`, escalating ×10 to `MAX_RETIREMENT_SECONDS = 3600`.
Any success clears the ladder (`record_success`). **Deliberately kept:** if *every*
key fails auth, `_wait_for_capacity` still raises `FailureKind.AUTHENTICATION`
immediately — that is a config mistake and must not hide behind a 5-minute stall.

## 6. Polling spin (found in this re-audit, fixed)

The FIFO queue added in E had a defect. A caller that was *not* next in line still
fell through to `_wait_for_capacity`, which projected a **zero** wait — the key is
free, it is simply owed to someone earlier — and so slept for zero and re-checked.
Every queued caller re-ran `_may_attempt` (which is O(waiters × keys)) on every
event-loop tick until the head caller happened to be scheduled.

Measured with a temporary revert: **5342 `_select` calls in a 0.3s window**, versus
~1 with the fix. Not a hang — it always drained — but a real CPU hot spot under
exactly the fan-out load this project exists to survive.

Fix in `KeyPool.acquire`: hoist `my_turn = self._may_attempt(...)`, and when it is
false park on `_sleep_until_capacity(_WAIT_SLICE_SECONDS)` instead of polling.
Safe because if all keys were retired, `_select` returns `None` for every waiter, so
`_may_attempt` is true for everyone and the retired check is still reached.

Regression test: `test_a_caller_held_back_by_the_queue_sleeps_instead_of_polling`.
It was verified to **fail on the pre-fix code**, not just pass on the new code.

---

## 7. Config state — verified, no action needed

An earlier version of this doc claimed the user had to delete
`PROVIDER_RATE_LIMIT=1` / `PROVIDER_RATE_WINDOW=3` from their `.env`. **That was
wrong** — inferred from `.env.example` shipping those values, never checked.

Verified: **no `.env` exists in any location the proxy loads from** —
`free-claude-code\.env`, `~\.fcc\.env`, and `FCC_ENV_FILE` are all absent, and
neither variable is in the process environment. So the new per-provider catalog
defaults apply directly and the overhaul is fully live. Confirmed by running
`resolve_rate_policy` against a default `Settings()`:

```
nvidia_nim   38 req / 60s per key   -> 10 keys = 380/min
open_router  19 req / 60s per key   -> 10 keys = 190/min
pooled concurrency ceiling: 64      margin: 0.05
```

**Open question for the next session:** with no `.env`, the user's API keys are not
configured on this machine either (no `NVIDIA_NIM_API_KEYS` / `OPENROUTER_API_KEYS`).
Either they configure it somewhere not yet found, or the pooling has never actually
run here. The user was offered a scaffolded `.env` and had not answered as of the end
of the session. Resolve this before assuming any runtime behaviour was observed.

Expected capacity for their 10-key setup:

| Provider | Per key (after 5% margin) | Total | Concurrency |
|---|---|---|---|
| `nvidia_nim` | 38/min | 380/min | 50 |
| `open_router` | 19/min | 190/min | 50 |

## 8. Known limitations — state these honestly, do not paper over them

- **No queue-depth limit.** Above ~380/min sustained, requests queue and wait rather
  than failing. The gate bounds this in normal operation, but genuine sustained
  overload shows up as slow responses, not errors. This is the tradeoff the user
  asked for; they have been told.
- **OpenRouter's daily cap cannot be made seamless.** All keys share one account
  limit, and the user ruled out cross-provider fallback. The pool can only make it
  fail cleanly and retryably — not invisibly.
- **RPM pacing cannot cover limits that are not modelled.** Currently moot: the user
  confirmed neither provider enforces TPM. Revisit if that changes.
- `test_probe_exhaustion_fails_waiters_and_opens_after_cooldown` (in the admission
  suite, untouched by this work) flaked **once** under full parallel load early on
  and has not recurred in any subsequent full run. Timing-sensitive; watch it.

---

## 9. Next steps

1. **Nothing is committed.** Review `git diff`, then commit. `CLAUDE.md` requires the
   semver bump in the *same* commit as the production change — `6.9.1` and `uv.lock`
   are already staged in the working tree, so commit them together.
2. Run against the real workload. If errors persist, the trace events
   `provider.key_pool.wait`, `provider.key_pool.key_retired`, and
   `provider.key_pool.budget_exhausted` identify which mechanism is firing.
3. Consider live smoke coverage — `CLAUDE.md` asks for it and
   `smoke/product/test_key_pool_product_live.py` already pins the 401-vs-403
   provider split.

## 10. Conventions that will bite you

From `CLAUDE.md` (read it fully before editing):

- **No `# type: ignore` / `# ty: ignore`** and **no `from __future__ import annotations`** —
  both are grep-enforced as a CI check.
- Any production-path change needs a `pyproject.toml` semver bump **plus `uv lock`**
  in the same commit.
- `src/free_claude_code/cli/extension_assets/manifest.json` must always match the
  `pyproject.toml` version — pinned by `test_the_manifest_version_tracks_the_package_version`.
- Verify with `.\scripts\ci.ps1` (Windows). Note `pytest-timeout` is **not** installed —
  `--timeout=` will error.
- There is **no `.codegraph/` directory** in this repo, so skip CodeGraph entirely
  despite the global instruction; use Grep/Read.

## 11. Suggested skills

- **`impeccable`** — for the commit itself and any follow-up production edit. This
  repo's bar is explicitly zero-defect, root-cause-oriented, with dense
  rationale-carrying comments; the existing `key_pool.py` prose is the house style to
  match.
- **`improve-codebase-architecture`** — only if the next session takes on the
  queue-depth bound or a broader admission-layer refactor. Not needed to commit
  what exists.

No frontend, design, or media skills apply here.

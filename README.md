<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="assets/free-claude-code-wordmark-light.svg">
    <img src="assets/free-claude-code-wordmark-dark.svg" alt="Free Claude Code" width="610">
  </picture>
</h1>

Use Claude Code, Codex, Pi, or their IDE extensions through your own provider-backed proxy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Tested with Pytest](https://img.shields.io/badge/testing-Pytest-00c0ff.svg?style=for-the-badge)](https://github.com/King-Jboy/kingjboy-claude-code/actions/workflows/tests.yml)
[![Type checking: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00.svg?style=for-the-badge)](https://pypi.org/project/ty/)
[![Code style: Ruff](https://img.shields.io/badge/code%20formatting-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Logging: Loguru](https://img.shields.io/badge/logging-loguru-4ecdc4.svg?style=for-the-badge)](https://github.com/Delgan/loguru)

Run your coding agents with free, paid, or local models. Choose and validate providers from one local Admin UI.

[Quick Start](#quick-start) · [Providers](#choose-a-provider) · [Clients](#connect-your-client) · [Integrations](#optional-integrations) · [Manage](#manage-your-installation)

> A personal fork of [Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code), MIT licensed to Ali Khokhar.
> This fork adds credential pooling, `fcc-doctor`, and a Claude-Code-only installer. See [What This Fork Changes](#what-this-fork-changes).

</div>

<div align="center">
  <img src="assets/pic.png" alt="Free Claude Code in action" width="700">
  <p><em>Claude Code running through the Free Claude Code proxy.</em></p>
</div>

<div align="center">
  <img src="assets/codex.png" alt="Codex CLI in action through Free Claude Code" width="700">
  <p><em>Codex CLI using the local FCC Responses provider.</em></p>
</div>

<a id="model-picker"></a>

<div align="center">
  <img src="assets/cc-model-picker.png" alt="Claude Code model picker showing gateway models" width="700">
  <p><em>Claude Code native <code>/model</code> picker with FCC gateway models.</em></p>
</div>

<div align="center">
  <img src="assets/codex-model-picker.png" alt="Codex model picker showing generated FCC model catalog" width="700">
  <p><em>Codex native <code>/model</code> picker with the generated FCC catalog.</em></p>
</div>

## What You Get

- Launch Claude Code with `fcc-claude`, Codex with `fcc-codex`, or Pi with `fcc-pi`.
- Run FCC in the background from a desktop launcher on Windows or macOS.
- Switch among 31 cloud and local providers from the Admin UI.
- Use each coding agent's native model picker.
- Route Fable, Opus, Sonnet, Haiku, and fallback traffic to different models.
- Pool many NVIDIA NIM or OpenRouter keys into one self-healing virtual key.
- Diagnose a broken setup in one command with `fcc-doctor`.
- Keep streaming, tool use, reasoning, and image input across compatible models.
- Connect Claude Code and Codex in VS Code or Claude Code through JetBrains ACP.
- Debug the page you are on from a Chrome side panel with `fcc-extension`, optionally running approved shell commands.
- Optionally run Claude Code sessions through Discord or Telegram with voice-note transcription.
- Protect the local proxy with optional token authentication.

## Quick Start

<a id="install"></a>

### 1. Install Or Update

macOS/Linux:

```bash
curl -fsSL "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.sh" | sh
```

Windows PowerShell:

```powershell
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.ps1")))
```

Re-run the same command whenever you want to update. You can review the installers before running them: [install.sh](scripts/install.sh) and [install.ps1](scripts/install.ps1).

The installer sets up Claude Code alongside FCC. Other coding agents are left unchanged; install them yourself if you want to use `fcc-codex` or `fcc-pi`.

### 2. Start FCC

#### Windows

Open **Free Claude Code** from your desktop or Start menu.

#### macOS

Open **Free Claude Code** from your desktop or Applications folder.

#### Linux

Run:

```bash
fcc-server
```

On Windows and macOS, FCC runs in the system tray or menu bar without opening a
terminal. Use its menu to open Admin, check server status, restart, or quit. On
Windows, left-clicking the tray icon opens Admin directly.

To print the installed Free Claude Code version without starting the server,
run `fcc-server --version`.

When using `fcc-server`, keep the terminal open. The Admin UI opens in your
browser once the server is healthy by default. Its address is shown in the
startup log:

```text
INFO:     Admin UI: http://127.0.0.1:8082/admin (local-only)
```

Use the port shown in your terminal if it differs from `8082`.

<a id="nvidia-nim-provider"></a>

### 3. Configure NVIDIA NIM

1. Create an API key at [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys).
2. Open the Admin UI URL from the server log.
3. Paste the key into `NVIDIA_NIM_API_KEY`.
4. Leave `MODEL` on the default `nvidia_nim/nvidia/nemotron-3-super-120b-a12b`, or search the model dropdown and select another model.
5. Click **Validate**, then **Apply**.

<div align="center">
  <img src="assets/admin-page.png" alt="Local admin UI for proxy settings" width="700">
</div>

### 4. Run Your Coding Agent

Claude Code:

```bash
fcc-claude
```

Codex:

```bash
fcc-codex
```

Pi:

```bash
fcc-pi
```

All three launchers use the current Admin UI settings. Use the agent's model picker to choose from the models FCC exposes. Normal CLI arguments still work, for example:

```bash
fcc-codex exec "hello"
```

`fcc-pi` registers FCC only for that Pi process; your existing Pi settings, sessions, credentials, and extensions remain unchanged.

## Choose A Provider

1. Open a provider link below for its key, models, or setup instructions.
2. In the Admin UI, configure the listed setting. For OpenAI, use
   **Providers → Connected accounts** instead.
3. Search the `MODEL` dropdown and select a model. If the provider cannot list
   models, enter `<provider-id>/<exact-provider-model-id>` manually.
4. Click **Validate**, then **Apply**.

| Provider | Admin UI setting | Example `MODEL` |
| --- | --- | --- |
| [NVIDIA NIM](https://build.nvidia.com/settings/api-keys) | `NVIDIA_NIM_API_KEY` | `nvidia_nim/nvidia/nemotron-3-super-120b-a12b` |
| [OpenAI / ChatGPT](https://learn.chatgpt.com/docs/auth) | Connect ChatGPT in the Admin UI | `openai/<model-id>` |
| [Azure OpenAI](https://learn.microsoft.com/azure/foundry/openai/how-to/chatgpt) | `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_BASE_URL` | `azure_openai/<deployment-name>` |
| [OpenRouter](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` | `open_router/openrouter/free` |
| [Google AI Studio (Gemini)](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | `gemini/models/gemini-3.1-flash-lite` |
| [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/start/openai) | `VERTEX_PROJECT_ID` + ADC | `vertex/google/gemini-3.5-flash` |
| [DeepSeek](https://platform.deepseek.com/api_keys) | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| [Mistral La Plateforme](https://console.mistral.ai/) | `MISTRAL_API_KEY` | `mistral/devstral-small-latest` |
| [Mistral Codestral](https://console.mistral.ai/) | `CODESTRAL_API_KEY` | `mistral_codestral/codestral-latest` |
| [OpenCode Zen](https://opencode.ai/auth) | `OPENCODE_API_KEY` | `opencode/gpt-5.3-codex` |
| [OpenCode Go](https://opencode.ai/auth) | `OPENCODE_API_KEY` | `opencode_go/minimax-m2.7` |
| [Vercel AI Gateway](https://vercel.com/docs/ai-gateway/models-and-providers) | `AI_GATEWAY_API_KEY` | `vercel/openai/gpt-5.5` |
| [Amazon Bedrock](https://console.aws.amazon.com/bedrock/) | `AWS_BEARER_TOKEN_BEDROCK` | `bedrock/openai.gpt-oss-120b` |
| [Hugging Face Inference Providers](https://huggingface.co/settings/tokens) | `HUGGINGFACE_API_KEY` | `huggingface/Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest` |
| [Cohere](https://dashboard.cohere.com/api-keys) | `COHERE_API_KEY` | `cohere/command-a-plus-05-2026` |
| [GitHub Models](https://github.com/marketplace?type=models) | `GITHUB_MODELS_TOKEN` | `github_models/openai/gpt-4.1` |
| [Wafer](https://wafer.ai/) | `WAFER_API_KEY` | `wafer/DeepSeek-V4-Pro` |
| [Kimi API](https://platform.moonshot.ai/console/api-keys) | `KIMI_API_KEY` | `kimi/kimi-k2.5` |
| [Kimi Code](https://www.kimi.com/code/console) | `KIMI_CODE_API_KEY` | `kimi_code/k3` |
| [MiniMax](https://platform.minimax.io/user-center/basic-information/interface-key) | `MINIMAX_API_KEY` | `minimax/MiniMax-M3` |
| [Cerebras Inference](https://cloud.cerebras.ai/) | `CEREBRAS_API_KEY` | `cerebras/gpt-oss-120b` |
| [Groq](https://console.groq.com/keys) | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| [SambaNova](https://cloud.sambanova.ai/apis) | `SAMBANOVA_API_KEY` | `sambanova/Meta-Llama-3.3-70B-Instruct` |
| [Kilo.ai](https://kilo.ai) | `KILO_API_KEY` | `kilo/kilo-auto/free` |
| [Fireworks AI](https://fireworks.ai/account/api-keys) | `FIREWORKS_API_KEY` | `fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct` |
| [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) | `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` | `cloudflare/@cf/moonshotai/kimi-k2.6` |
| [Z.ai](https://z.ai/manage-apikey/apikey-list) | `ZAI_API_KEY` | `zai/glm-5.2` |
| [Ollama Cloud](https://ollama.com/settings/keys) | `OLLAMA_API_KEY` | `ollama_cloud/qwen3-coder:480b` |
| [LM Studio](https://lmstudio.ai/) | `LM_STUDIO_BASE_URL` | `lmstudio/<model-id>` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `LLAMACPP_BASE_URL` | `llamacpp/<model-id>` |
| [Ollama](https://ollama.com/) | `OLLAMA_BASE_URL` | `ollama/<model-tag>` |

Important provider notes:

- OpenAI uses your ChatGPT subscription rather than an API key. Connect from
  **Providers → Connected accounts**; browser PKCE is the default and device
  code is available for headless setups. FCC stores its own renewable
  credentials under `~/.fcc/auth/` and leaves Codex login untouched. Restart
  an already-running agent after connecting to refresh its model picker.
- Azure OpenAI uses the deployment names from your resource. Set
  `AZURE_OPENAI_BASE_URL` to its complete v1 endpoint, such as
  `https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/`, and select a
  deployment that supports Chat Completions. Azure does not expose custom
  deployment names through its data-plane model list, so enter the deployment
  name as a custom model slug.
- Mistral Codestral uses a separate key from Mistral La Plateforme.
- Kimi Code subscription keys use `kimi_code/`; Kimi API credit keys use
  `kimi/`. Kimi Code plans are for personal interactive coding-agent use under
  [Kimi's community guidelines](https://www.kimi.com/code/docs/en/kimi-code/community-guidelines.html).
- OpenCode Zen and OpenCode Go share `OPENCODE_API_KEY` but use different model prefixes.
- Amazon Bedrock uses its Mantle OpenAI-compatible endpoint. Set
  `BEDROCK_BASE_URL` to the endpoint for the same region as the API key and
  select one of the models returned by FCC's model picker.
- Vertex AI uses Google Application Default Credentials instead of an API key.
  Locally, run `gcloud auth application-default login` once; service-account
  files and attached service accounts also work. Set `VERTEX_PROJECT_ID`, and
  optionally change `VERTEX_LOCATION` from its `global` default. FCC refreshes
  expiring access tokens automatically.
- Cloudflare requires both its API token and account ID.
- Ollama Cloud connects directly to `ollama.com`; use the exact model IDs shown
  by FCC's model picker. Local Ollama remains available through the separate
  `ollama/` prefix.
- Prefer tool-capable models for coding agents. Local models also need enough context for the agent's system prompt and tool definitions.

<details>
<summary><strong>Local provider setup</strong></summary>

### LM Studio

Start LM Studio's local server, load a tool-capable model, and use the model identifier shown by LM Studio with the `lmstudio/` prefix. The default URL is `http://localhost:1234/v1`.

### llama.cpp

Start `llama-server` with its OpenAI-compatible Chat Completions API and enough context for the model. Use the local model ID with the `llamacpp/` prefix. `LLAMACPP_BASE_URL` defaults to `http://localhost:8080/v1`; FCC accepts either the server root or an explicit `/v1` suffix.

### Ollama

```bash
ollama pull llama3.1
ollama serve
```

Use the tag shown by `ollama list` with the `ollama/` prefix. `OLLAMA_BASE_URL` defaults to `http://localhost:11434`; FCC accepts either the root URL or an explicit `/v1` suffix.

</details>

### Optional Model-Tier Routing

`MODEL` is the fallback for every request. Select a model for `MODEL_FABLE`, `MODEL_OPUS`, `MODEL_SONNET`, or `MODEL_HAIKU` to override an individual Claude Code tier; select **None** to use `MODEL`.

For example, route Opus to `nvidia_nim/nvidia/nemotron-3-super-120b-a12b`, Sonnet to `open_router/openrouter/free`, Haiku to `lmstudio/qwen3.5-coder`, and keep `MODEL` on `zai/glm-5.2`.

### Client Context Window

The context window is the token budget FCC hands to a CLI it launches. Claude Code compacts against exactly this number, so it decides when your session starts dropping history.

**Leave Client Context Window (`CLIENT_CONTEXT_WINDOW`) blank and it resolves itself** from the routed model's row in `~/.fcc/context.md`. Switch `MODEL` and the window switches with it — there is no second setting to keep in sync. Generate that table once with [the probe script](#finding-a-models-context-window):

```bash
fcc-context
```

The number cannot be discovered at request time, which is why a recorded table exists at all: most OpenAI-compatible `/v1/models` responses carry no context length, and NVIDIA NIM returns only `id`, `object`, `created`, and `owned_by`.

Resolution order:

| Situation | Window used |
| --- | --- |
| `CLIENT_CONTEXT_WINDOW` set to a number | that number, always |
| Blank, routed model is in `context.md` | the recorded value |
| Blank, several routes recorded | the **smallest**, since one number covers every tier the client can select |
| Blank, nothing recorded | `262144` — a guess, so run `fcc-context` instead of relying on it |

`fcc-doctor` prints the window and where it came from:

```
[  ok  ] context window: 262,144 tokens (from context.md for nvidia_nim/deepseek-ai/deepseek-v4-pro)
```

> The VS Code and JetBrains integrations below launch Claude Code themselves, so they never read any of this. Set `CLAUDE_CODE_AUTO_COMPACT_WINDOW` in their own config to the value `fcc-doctor` reports.

### Your Own Model List

Two configured providers can advertise several hundred models, and a client model picker lists each one twice (thinking and no-thinking). Most are not models you would route to, and some no longer serve requests.

Two settings turn that into a shortlist you control:

**Model List Scope** (`MODEL_CATALOG_SCOPE`) — set it to **Only my configured models** and both `/v1/models` and the Admin pickers stop listing discovered models.

**My Models** (`PINNED_MODELS`) — a JSON list of `provider/model` refs that are *always* offered. This is what makes the shortlist yours rather than just your five routing slots:

```json
[
  "nvidia_nim/deepseek-ai/deepseek-v4-flash",
  "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
  "open_router/z-ai/glm-5.2:free"
]
```

Add a line to add a model, delete a line to remove one. Entries do not have to be discovered yet — a slug you type is selectable immediately, which is how you reach a model the provider added since your last refresh.

So a scoped list contains your `MODEL` and `MODEL_*` routes, plus everything in **My Models**, plus the Claude alias ids clients need. Nothing is locked away: the Admin model fields still accept any slug you type, and **Refresh Models** lists everything your providers report whatever the scope says.

### Finding A Model's Context Window

`fcc-context` fills in `~/.fcc/context.md`, the table FCC reads to resolve the window automatically:

```bash
fcc-context
```

**The table holds exactly the models you can route to** — everything in `PINNED_MODELS` plus your `MODEL` / `MODEL_*` routes, and nothing else. It stays a short list of models you actually use rather than a catalogue that only grows.

So the loop is: add a model to **My Models**, re-run, its window joins the table. Unpin one and it leaves on the next run. Models already recorded are not probed again, so adding a model costs one probe rather than a full sweep:

```
nvidia_nim: no published context length, probing
  7 already known, probing 1
    stepfun-ai/step-3.7-flash                        262,144
```

It reads the number where a provider publishes it and measures it where one does not. OpenRouter reports `context_length` for every model, so those are instant and need no key. NVIDIA NIM publishes nothing, so each model gets one deliberately oversized request and the script reads the ceiling out of the rejection:

```
This model's maximum context length is 262144 tokens. However, your messages
resulted in 360007 tokens. Please reduce the length of the messages.
```

That costs no inference, because the probe deliberately overshoots — a rejection states the ceiling no matter how far over you were. Only a model that *accepts* the probe has to be re-probed larger.

```bash
fcc-context --all        # whole catalog, slow
fcc-context --refresh    # re-measure recorded models
fcc-context --models nvidia_nim/meta/llama-3.3-70b-instruct
```

`--provider` limits what gets re-measured, not what the table contains, so an OpenRouter-only run leaves your NVIDIA NIM rows alone. Keys come from `~/.fcc/.env` and a configured pool is used round-robin, so one key's rate limit does not stall the run. A model a provider lists but does not serve to your account is recorded with the reason instead of a number — on NVIDIA NIM that turns out to be most of the catalog, which is why `--all` is rarely worth running.

### Reasoning Control

Open **Admin UI → Model Config → Reasoning** and select the behavior you want.

| Selection | Behavior |
| --- | --- |
| **From client** (default) | Use the effort sent by Claude Code, Codex, or Pi. If none is sent, keep the provider default. |
| **Off** | Request reasoning to be disabled. |
| **Low**, **Medium**, **High**, **X-High**, or **Max** | Override the client with the selected reasoning level. |
| **Inherit** (Fable, Opus, Sonnet, and Haiku only) | Use the root Reasoning selection. |

Providers that do not support a selected control retain their own behavior.

### Key Pools (NVIDIA NIM and OpenRouter)

If you hold several API keys for NVIDIA NIM or OpenRouter, FCC can treat them as one virtual key. Open **Admin UI → Providers**, find **NVIDIA NIM API Key Pool** or **OpenRouter API Key Pool**, and paste a JSON list:

```json
["key-one", "key-two", "key-three"]
```

Click **Validate**, then **Apply**. There is no limit on how many keys you add, and the pool replaces the single API key field for that provider.

Each key gets its own rate-limit window, and all keys run at the same time, so the pool's throughput is the sum of its keys rather than one key's ceiling. Per request, FCC picks the key with the most headroom left:

| Upstream response | What FCC does |
| --- | --- |
| `401` | Retires that key for the session, then immediately tries another. |
| `403` | Sidelines that key and tries another, but does **not** retire it. |
| `429` | Cools that key until the reset time the provider reported, then immediately tries another. |
| `5xx`, timeout, connection error | Treats it as a backend problem, not a key problem, and applies the normal shared backoff. |

`401` and `403` are handled differently because providers disagree about which one means "bad key". OpenRouter answers `401`, which is unambiguous. NVIDIA NIM answers `403` — but other providers use `403` to refuse the *request* (content policy, or a model the account cannot reach). Retiring on `403` would let a single refused prompt walk the pool and kill every key. So a `403` only sidelines its key; the error is reported to you unchanged once **every** key has refused the same request alike, which is the only proof that the request, not the keys, was at fault.

When a provider states its own reset time (`Retry-After` or `X-RateLimit-Reset`) FCC obeys it exactly. When there is no timing at all — the shape of a dead credential — each further refusal from the same key multiplies the wait (3s, 30s, 5min, capped at 10min), so a key that is never coming back drops out of rotation instead of costing a wasted round-trip forever. Any successful request clears that backoff, so a key that starts working again returns on its own.

Switching keys is instant and never uses up a request's retry budget. If every key is briefly rate limited, FCC waits for the first one to free up, up to a total of 60 seconds. If the wait would be longer — for example a daily cap that resets tomorrow — the request fails right away instead of hanging.

One key alone behaves exactly as before, so there is nothing to change if you only have one. Key health is held in memory and resets when FCC restarts. Live pool health appears on each provider card in **Admin UI → Providers** (`Key pool: 14 keys · 12 ready · 2 cooling`), and `fcc-doctor` reports the configured pool sizes.

> **OpenRouter note:** OpenRouter applies free-tier limits per *account*. Keys minted from separate accounts therefore multiply your throughput; several keys on one account give you redundancy rather than more headroom.

### Checking Your Setup

Run `fcc-doctor` when something is off, or before you rely on a long session:

```bash
fcc-doctor
```

```
[  ok  ] managed env: /home/you/.fcc/.env
[  ok  ] server port: 8082 is free
[  ok  ] claude cli: /home/you/.local/bin/claude
[  ok  ] MODEL: nvidia_nim/nvidia/nemotron-3-super-120b-a12b
[  ok  ] NVIDIA_NIM_API_KEYS: 13 keys pooled
[ FAIL ] MODEL catalog: open_router no longer advertises moonshotai/kimi-k2.6:free
         -> Pick a current model in the Admin UI; providers retire these silently.
```

It checks that your managed env file exists, that every routed model points at a configured provider, that each key pool parses to the size you expect, whether the port is already serving, and whether the Claude Code CLI is on PATH. It also asks each provider whether your configured model is *still* in its catalog — providers move models off their free tiers without warning, and otherwise you only find out when a request fails mid-session.

| Flag | Effect |
| --- | --- |
| `--offline` | Skip every check that contacts a provider. |
| `--json` | Emit findings as JSON for scripting. |

It exits non-zero if anything failed, so you can gate a script on it. Doctor never builds a provider or sends a completion, so it costs no credits.

<a id="connect-your-client"></a>

## Connect Your Client

For terminal use, start `fcc-server`, then run `fcc-claude`, `fcc-codex`, or `fcc-pi`. Use the guides below for editor integrations.

<details>
<summary><strong>Chrome side panel</strong></summary>

A Manifest V3 extension that puts a chat panel beside the page you are working on, with tools that read the page for you. Useful for front-end debugging: it can read the DOM and the tab's console output without you pasting anything.

Print the directory to load and the details to paste:

```bash
fcc-extension
```

Then load it once:

1. Open `chrome://extensions`
2. Turn on **Developer mode**
3. **Load unpacked**, and choose the directory `fcc-extension` printed

Open the panel from the toolbar. It auto-connects to the proxy at the saved URL and fills its model picker from `/v1/models`, so `MODEL_CATALOG_SCOPE` and `PINNED_MODELS` shape the dropdown exactly as they do in the Admin UI.

If your proxy has `ANTHROPIC_AUTH_TOKEN` set, `fcc-extension` masks it by default; add `--show-token` to print it. Leave the panel's token field blank when the variable is unset.

**Tools the model can call**

| Tool | What it reads |
| --- | --- |
| `page_info` | URL, title, viewport of the active tab |
| `read_page` | Rendered text or raw HTML, whole document or one CSS selector |
| `read_console` | Console output and uncaught errors recorded since page load |

Untick **Let the model read the active tab** in the panel's settings to send no tools at all.

**Running shell commands (optional, off by default)**

Manifest V3 cannot spawn a process, so this goes through a Chrome Native Messaging host — `fcc-bridge`. That moves the boundary from the network to the OS: only the browser can reach it, and only for the one extension ID named in the host manifest. An HTTP endpoint on `fcc-server` would not be equivalent, because that server binds `0.0.0.0` by default and skips auth entirely when `ANTHROPIC_AUTH_TOKEN` is blank.

Three things must all be true before a single command runs:

1. **The bridge is registered for your extension.** The panel's settings pane prints the exact command, with your ID already filled in:
   ```bash
   fcc-extension install --extension-id <id>
   ```
   Restart the browser afterwards. `fcc-extension uninstall` removes it.
2. **`BROWSER_SHELL_ENABLED=true`** in `~/.fcc/.env`. Registering the bridge is deliberately not enough on its own.
3. **You approve the command.** The panel shows the exact string and waits. Nothing runs until you click **Run**.

Commands are confined to `BROWSER_SHELL_ROOT` (default: your home directory — narrow it to a project root). A `cwd` that resolves outside that tree is refused, not clamped. Output is capped, there is no interactive stdin, and every command is appended to `~/.fcc/logs/bridge.log` with its exit code.

The shell is PowerShell on Windows (`pwsh`, falling back to `powershell`) and `$SHELL` elsewhere — not `cmd.exe`, which is what the naive choice would have given you.

**Limits worth knowing**

- **The console recorder attaches at page load.** Tabs already open when you installed the extension record nothing until you reload them.
- **`chrome://`, the Web Store, and other extensions are closed to it** by Chrome policy, not by choice.
- **Codespaces and other web IDEs** are readable as pages, but their terminals run on a remote container the extension cannot reach.

</details>

<details>
<summary><strong>Claude Code in VS Code</strong></summary>

Install the [Claude Code extension](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code). Open VS Code's user settings as JSON and add:

```json
"claudeCode.disableLoginPrompt": true,
"claudeCode.environmentVariables": [
  { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
  { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" },
  { "name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "value": "1" },
  { "name": "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "value": "262144" },
  { "name": "DISABLE_AUTOUPDATER", "value": "1" },
  { "name": "DISABLE_FEEDBACK_COMMAND", "value": "1" },
  { "name": "DISABLE_ERROR_REPORTING", "value": "1" }
]
```

Match the port and authentication token to the Admin UI, then reload the extension. Match `CLAUDE_CODE_AUTO_COMPACT_WINDOW` to **Client Context Window** in the Admin UI — these editor integrations do not go through `fcc-claude`, so they cannot read the setting themselves.

</details>

<details>
<summary><strong>Codex App</strong></summary>

Start FCC, then add its provider and generated model catalog to your user-level Codex configuration.

**Windows** — edit `%USERPROFILE%\.codex\config.toml` and replace `YOUR_USERNAME`:

```toml
model_provider = "fcc"
model = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
model_catalog_json = "C:/Users/YOUR_USERNAME/.fcc/codex-model-catalog.json"

[model_providers.fcc]
name = "Free Claude Code"
base_url = "http://127.0.0.1:8082/v1"
http_headers = { Authorization = "Bearer freecc" }
wire_api = "responses"
```

**macOS** — edit `~/.codex/config.toml` and replace `YOUR_USERNAME`:

```toml
model_provider = "fcc"
model = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
model_catalog_json = "/Users/YOUR_USERNAME/.fcc/codex-model-catalog.json"

[model_providers.fcc]
name = "Free Claude Code"
base_url = "http://127.0.0.1:8082/v1"
http_headers = { Authorization = "Bearer freecc" }
wire_api = "responses"
```

Match the model, port, and bearer token to the Admin UI. Restart the Codex App after setup or model changes, then use its model picker to select any FCC provider/model slug.

</details>

<details>
<summary><strong>Codex in VS Code</strong></summary>

Install the [Codex extension](https://marketplace.visualstudio.com/items?itemName=openai.chatgpt). Create or edit `~/.codex/config.toml` (`%USERPROFILE%\.codex\config.toml` on Windows):

```toml
model_provider = "fcc"
model = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"

[model_providers.fcc]
name = "Free Claude Code"
base_url = "http://127.0.0.1:8082/v1"
http_headers = { Authorization = "Bearer freecc" }
wire_api = "responses"
```

Match `model`, the port, and bearer token to the Admin UI, then restart VS Code. For WSL-backed Codex, edit the file inside WSL.

</details>

<details>
<summary><strong>Claude Code in JetBrains ACP</strong></summary>

Edit the installed Claude ACP configuration:

- Windows: `C:\Users\%USERNAME%\AppData\Roaming\JetBrains\acp-agents\installed.json`
- Linux/macOS: `~/.jetbrains/acp.json`

Set the environment for `acp.registry.claude-acp`:

```json
"env": {
  "ANTHROPIC_BASE_URL": "http://localhost:8082",
  "ANTHROPIC_AUTH_TOKEN": "freecc",
  "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "262144",
  "DISABLE_AUTOUPDATER": "1",
  "DISABLE_FEEDBACK_COMMAND": "1",
  "DISABLE_ERROR_REPORTING": "1"
}
```

Match the port and token to the Admin UI, then restart the IDE. Match `CLAUDE_CODE_AUTO_COMPACT_WINDOW` to **Client Context Window** in the Admin UI, as above.

</details>

<details>
<summary><strong>Claude Code still asks you to log in</strong></summary>

If Claude Code asks you to log in after you configure the FCC URL and token, open its state file:

- Windows: `%USERPROFILE%\.claude.json`
- macOS/Linux/WSL: `~/.claude.json`

Merge this property into the existing JSON without removing its other fields:

```json
"hasCompletedOnboarding": true
```

If the file does not exist, create it with a complete JSON object:

```json
{
  "hasCompletedOnboarding": true
}
```

Restart Claude Code or the IDE after saving the file.

</details>

<a id="optional-integrations"></a>

## Optional Integrations

Configure integrations from **Admin UI → Messaging**, then click **Validate** and **Apply**.

<div align="center">
  <img src="assets/admin-messaging.png" alt="Admin UI Messaging view with bot and voice settings" width="700">
</div>

<details>
<summary><strong>Discord bot</strong></summary>

1. Create a bot in the [Discord Developer Portal](https://discord.com/developers/applications).
2. Enable **Message Content Intent** and invite it with read, send,
   message-history, and **Manage Messages** permissions so `/clear` can remove
   user prompts.
3. Set **Messaging Platform** to **discord**.
4. Enter **Discord Bot Token**, **Allowed Discord Channels**, and an absolute **Allowed Directory**.
5. Apply the settings and restart the server if requested.

</details>

<details>
<summary><strong>Telegram bot</strong></summary>

1. Create a bot with [@BotFather](https://t.me/BotFather).
2. Get your numeric user ID from [@userinfobot](https://t.me/userinfobot).
   In groups, grant the bot permission to delete messages.
3. Set **Messaging Platform** to **telegram**.
4. Enter **Telegram Bot Token**, **Allowed Telegram User ID**, and an absolute **Allowed Directory**.
5. Apply the settings and restart the server if requested.

</details>

### Messaging commands

| Usage | Behavior |
| --- | --- |
| `/stats` | Show session state. |
| Standalone `/stop` | Cancel all work. |
| Reply with `/stop` | Cancel only the selected request while other queued requests continue. |
| Standalone `/clear` | Reset all FCC state and remove every tracked message in that chat, including user prompts, voice notes, FCC replies, Telegram's online notice, and the clear command itself. |
| Reply with `/clear` | Delete the selected message and its literal platform reply subtree while preserving its ancestors and siblings. |

<details>
<summary><strong>Voice notes</strong></summary>

Re-run the installer with the voice backend you need.

macOS/Linux:

```bash
# NVIDIA NIM transcription
curl -fsSL "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.sh" | sh -s -- --voice-nim

# Local Whisper on CPU or CUDA
curl -fsSL "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.sh" | sh -s -- --voice-local

# Both backends
curl -fsSL "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.sh" | sh -s -- --voice-all

# Local Whisper with the CUDA 13.0 PyTorch backend
curl -fsSL "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.sh" | sh -s -- --voice-local --torch-backend cu130
```

Windows PowerShell:

```powershell
# NVIDIA NIM transcription
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.ps1"))) -VoiceNim

# Local Whisper on CPU or CUDA
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.ps1"))) -VoiceLocal

# Both backends
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.ps1"))) -VoiceAll

# Local Whisper with the CUDA 13.0 PyTorch backend
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/install.ps1"))) -VoiceLocal -TorchBackend cu130
```

Restart `fcc-server`. In **Admin UI → Messaging → Voice**, enable voice notes, select `cpu`, `cuda`, or `nvidia_nim`, and choose the Whisper model. Local gated models need `HUGGINGFACE_API_KEY`; NVIDIA NIM transcription needs `NVIDIA_NIM_API_KEY`.

</details>

## Manage Your Installation

### Update

Re-run the matching command from [Install Or Update](#install).

### Uninstall

Stop every running FCC command first. The uninstaller verifies every FCC command is gone
before deleting its managed data.

**Removes**

- Free Claude Code, including its desktop launcher and commands
- `~/.fcc/`

**Keeps**

- uv and Python
- Claude Code, Codex, and Pi
- Shared PATH entries

macOS/Linux:

```bash
curl -fsSL "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/uninstall.sh" | sh
```

Windows PowerShell:

```powershell
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/King-Jboy/kingjboy-claude-code/main/scripts/uninstall.ps1")))
```

## Project Links

- [Report bugs or request features](https://github.com/King-Jboy/kingjboy-claude-code/issues)
- [Architecture and extension guide](ARCHITECTURE.md)
- [Contributing guide](CONTRIBUTING.md)

<a id="what-this-fork-changes"></a>

## What This Fork Changes

Everything below is additional to upstream [Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code).

**Credential pooling.** `NVIDIA_NIM_API_KEYS` and `OPENROUTER_API_KEYS` accept a JSON list of keys that behave as one high-throughput, self-healing virtual key. Dead keys are walked past, rate-limited keys are cooled for exactly as long as the provider asked, and repeat offenders back off on an escalating schedule. See [Key Pools](#key-pools-nvidia-nim-and-openrouter).

Verified live against both providers: 20 concurrent requests spread across all 14 OpenRouter keys, and a request still succeeded with three dead keys sitting in front of the working ones.

**`fcc-doctor`.** A one-command health check for config, pools, ports, and — the useful part — whether your configured model is still in its provider's catalog. See [Checking Your Setup](#checking-your-setup).

**Pool visibility in the Admin UI.** Each pooled provider card shows live key health, so silent capacity loss is visible instead of showing up as unexplained slowness.

**Admin saves no longer delete unrecognised variables.** Upstream rewrites `~/.fcc/.env` from its field manifest alone, which silently dropped any variable it did not own. Hand-added variables are now preserved; genuinely retired settings are still cleaned up.

**A context window that follows the model.** Upstream hardcodes the launched CLI's context budget at 190,000 tokens. On a larger model that silently throws the difference away — DeepSeek V4 Flash on NVIDIA NIM accepts 1,048,576, so five sixths of the context went unused and sessions compacted early. The window is now read from the routed model's measured entry, so switching `MODEL` switches it. See [Client Context Window](#client-context-window).

**A model list you curate.** `MODEL_CATALOG_SCOPE=configured` narrows the client and Admin model lists to what you route to, and `PINNED_MODELS` is a shortlist you add to and remove from freely. See [Your Own Model List](#your-own-model-list).

**A context-window table you can regenerate.** `fcc-context` measures your routable models and records them in `~/.fcc/context.md`, reading published metadata where a provider offers it. See [Finding A Model's Context Window](#finding-a-models-context-window).

**A Chrome side panel.** `fcc-extension` ships a Manifest V3 extension that talks to your local proxy from a panel beside the page, with tools that read the DOM and the tab's console. Optionally it runs shell commands too, through the `fcc-bridge` native messaging host — gated behind a registration step, a `BROWSER_SHELL_ENABLED` switch, per-command approval, and a directory confinement, because the obvious alternative of an exec endpoint on `fcc-server` would be unauthenticated LAN-reachable RCE. See [Connect Your Client](#connect-your-client).

**`count_tokens` off the event loop.** The token-count endpoint ran tiktoken inline in the async handler, stalling every in-flight stream for the duration (~90ms on a 100k-token request). It now runs in a worker thread.

**Claude-Code-only installer.** `install.sh` / `install.ps1` no longer offer to install Codex or Pi. The `fcc-codex` and `fcc-pi` entry points still ship and still work if you install those agents yourself.

**Testing.** Provider behaviour is asserted against responses recorded from the real endpoints rather than hand-written mocks, and the parsers are covered by property tests. Recorded status codes differ by provider — NVIDIA NIM answers `403` for a bad key, OpenRouter answers `401` — and the pool depends on that distinction.

## License

MIT License, unchanged from upstream. Copyright (c) 2026 Ali Khokhar. See [LICENSE](LICENSE) for details.

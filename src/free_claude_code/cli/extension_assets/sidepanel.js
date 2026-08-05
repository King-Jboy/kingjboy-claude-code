// Side panel: a Messages-API client for the local Free Claude Code proxy.
//
// The panel calls the proxy directly. It is an extension page, so it runs on
// the extension origin and Chrome grants it the host_permissions CORS bypass;
// no preflight is sent and the proxy needs no CORS middleware. Routing this
// through the service worker instead would only add a message hop.

import { PAGE_TOOLS, runPageTool } from "./page_tools.js";

const SETTINGS_KEY = "fcc.connection";
const DEFAULT_BASE_URL = "http://127.0.0.1:8081";
const MAX_TOKENS = 8192;
// A tool_use turn that never terminates would loop against the provider until
// the user's quota ran out. Real page-debugging turns settle in two or three.
const MAX_TOOL_ROUNDS = 12;

const SYSTEM_PROMPT =
  "You are Free Claude Code running in a Chrome side panel, helping the user with the web " +
  "page they are looking at. When they refer to 'this page', 'the console', or 'the error', " +
  "use your tools to look rather than asking them to paste. Read the smallest region that " +
  "answers the question -- prefer a CSS selector over the whole document. You cannot click, " +
  "type, navigate, or run shell commands; say so plainly if asked, and tell the user what to " +
  "do instead. Be concise: this panel is narrow.";

const ui = {
  status: document.getElementById("status"),
  settings: document.getElementById("settings"),
  settingsToggle: document.getElementById("settings-toggle"),
  settingsNote: document.getElementById("settings-note"),
  baseUrl: document.getElementById("base-url"),
  authToken: document.getElementById("auth-token"),
  model: document.getElementById("model"),
  pageTools: document.getElementById("page-tools"),
  transcript: document.getElementById("transcript"),
  composer: document.getElementById("composer"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  clear: document.getElementById("clear"),
};

/** Conversation history in Messages API shape, replayed on every request. */
let history = [];
let busy = false;

// ---------- connection settings ----------

async function loadSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  return {
    baseUrl: DEFAULT_BASE_URL,
    authToken: "",
    model: "",
    pageTools: true,
    ...(stored[SETTINGS_KEY] ?? {}),
  };
}

async function saveSettings(settings) {
  await chrome.storage.local.set({ [SETTINGS_KEY]: settings });
}

function currentSettings() {
  return {
    baseUrl: ui.baseUrl.value.trim().replace(/\/+$/, ""),
    authToken: ui.authToken.value.trim(),
    model: ui.model.value,
    pageTools: ui.pageTools.checked,
  };
}

function authHeaders(settings) {
  const headers = { "content-type": "application/json", "anthropic-version": "2023-06-01" };
  // Blank is a legitimate configuration: the proxy skips the check entirely
  // when ANTHROPIC_AUTH_TOKEN is unset. Sending "Bearer " would not be.
  if (settings.authToken) headers.authorization = `Bearer ${settings.authToken}`;
  return headers;
}

function setStatus(text, state) {
  ui.status.textContent = text;
  ui.status.dataset.state = state;
}

function setNote(text, state = "") {
  ui.settingsNote.textContent = text;
  ui.settingsNote.dataset.state = state;
}

// ---------- model catalog ----------

/**
 * Group the proxy catalog the way the picker should read.
 *
 * The catalog advertises each provider model twice -- once under `anthropic/`
 * and once under `claude-3-freecc-no-thinking/`, which exists only to trip
 * Claude Code's client-side "no thinking" heuristic. Showing both would double
 * a list the user pinned precisely to keep short, so the no-thinking twin is
 * dropped wherever the normal one exists.
 */
function groupCatalog(payload) {
  const entries = Array.isArray(payload?.data) ? payload.data : [];
  const routed = entries.filter((entry) => entry.id?.startsWith("anthropic/"));
  const routedRefs = new Set(routed.map((entry) => entry.id.slice("anthropic/".length)));

  const noThinking = entries.filter((entry) => {
    const prefix = "claude-3-freecc-no-thinking/";
    return entry.id?.startsWith(prefix) && !routedRefs.has(entry.id.slice(prefix.length));
  });

  const aliases = entries.filter((entry) => entry.id && !entry.id.includes("/"));

  return [
    { label: "Your models", options: [...routed, ...noThinking] },
    { label: "Claude aliases", options: aliases },
  ].filter((group) => group.options.length > 0);
}

function renderCatalog(groups, selected) {
  ui.model.replaceChildren();
  for (const group of groups) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group.label;
    for (const entry of group.options) {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = entry.display_name || entry.id;
      optgroup.append(option);
    }
    ui.model.append(optgroup);
  }
  if (selected && groups.some((g) => g.options.some((o) => o.id === selected))) {
    ui.model.value = selected;
  }
}

async function connect(settings) {
  setNote("Connecting…");
  let response;
  try {
    response = await fetch(`${settings.baseUrl}/v1/models`, { headers: authHeaders(settings) });
  } catch {
    // A refused localhost connection is overwhelmingly "the server is not
    // running", which is a different fix from a wrong URL. Say both.
    setStatus("not connected", "error");
    setNote(`Could not reach ${settings.baseUrl}. Is fcc-server running?`, "error");
    return false;
  }

  if (response.status === 401) {
    setStatus("unauthorized", "error");
    setNote("The proxy rejected the token. Check ANTHROPIC_AUTH_TOKEN.", "error");
    return false;
  }
  if (!response.ok) {
    setStatus("error", "error");
    setNote(`The proxy returned HTTP ${response.status}.`, "error");
    return false;
  }

  const groups = groupCatalog(await response.json());
  if (!groups.length) {
    setStatus("no models", "error");
    setNote("The proxy advertises no models. Check MODEL and PINNED_MODELS.", "error");
    return false;
  }

  renderCatalog(groups, settings.model);
  await saveSettings({ ...settings, model: ui.model.value });
  setStatus("connected", "ok");
  setNote("");
  return true;
}

// ---------- transcript ----------

function addTurn(role, text) {
  const empty = ui.transcript.querySelector(".empty");
  if (empty) empty.remove();

  const node = document.createElement("div");
  node.className = "turn";
  node.dataset.role = role;
  node.textContent = text;
  ui.transcript.append(node);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  return node;
}

function showEmptyState() {
  history = [];
  ui.transcript.replaceChildren();
  const node = document.createElement("p");
  node.className = "empty";
  node.textContent =
    "Ask about the page you are on. The model can read its content and console output.";
  ui.transcript.append(node);
}

// ---------- streaming ----------

/**
 * Parse one Anthropic SSE frame set into the content blocks it completes.
 *
 * Blocks arrive as a start event, a run of deltas, and a stop -- text as
 * `text_delta` fragments, tool calls as `input_json_delta` fragments that only
 * become valid JSON once concatenated. `onText` fires per fragment so the
 * panel can render while the response is still arriving.
 */
async function streamAssistantTurn(response, onText) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const blocks = [];
  let buffer = "";
  let stopReason = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.split("\n").find((candidate) => candidate.startsWith("data:"));
      if (!line) continue;

      let event;
      try {
        event = JSON.parse(line.slice("data:".length).trim());
      } catch {
        continue;
      }

      if (event.type === "content_block_start") {
        const block = event.content_block;
        blocks[event.index] =
          block.type === "tool_use"
            ? { type: "tool_use", id: block.id, name: block.name, json: "" }
            : { type: "text", text: "" };
      } else if (event.type === "content_block_delta") {
        const block = blocks[event.index];
        if (!block) continue;
        if (event.delta.type === "text_delta") {
          block.text += event.delta.text;
          onText(event.delta.text);
        } else if (event.delta.type === "input_json_delta") {
          block.json += event.delta.partial_json;
        }
      } else if (event.type === "message_delta") {
        stopReason = event.delta?.stop_reason ?? stopReason;
      } else if (event.type === "error") {
        throw new Error(event.error?.message ?? "The proxy reported a stream error.");
      }
    }
  }

  const content = blocks.filter(Boolean).map((block) => {
    if (block.type !== "tool_use") return { type: "text", text: block.text };
    let input = {};
    try {
      // An empty-object tool call streams as "" rather than "{}".
      input = block.json ? JSON.parse(block.json) : {};
    } catch {
      input = {};
    }
    return { type: "tool_use", id: block.id, name: block.name, input };
  });

  return { content, stopReason };
}

async function requestTurn(settings) {
  const body = {
    model: settings.model,
    max_tokens: MAX_TOKENS,
    system: SYSTEM_PROMPT,
    messages: history,
    stream: true,
  };
  if (settings.pageTools) body.tools = PAGE_TOOLS;

  const response = await fetch(`${settings.baseUrl}/v1/messages`, {
    method: "POST",
    headers: authHeaders(settings),
    body: JSON.stringify(body),
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Proxy returned HTTP ${response.status}. ${detail.slice(0, 400)}`);
  }
  return response;
}

// ---------- turn loop ----------

function describeToolCall(name, input) {
  const detail = Object.entries(input ?? {})
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
  return detail ? `↳ ${name} ${detail}` : `↳ ${name}`;
}

async function runConversation(settings) {
  for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
    const response = await requestTurn(settings);

    let node = null;
    const { content, stopReason } = await streamAssistantTurn(response, (text) => {
      if (!node) node = addTurn("assistant", "");
      node.textContent += text;
      ui.transcript.scrollTop = ui.transcript.scrollHeight;
    });

    if (!content.length) {
      addTurn("error", "The model returned an empty response.");
      return;
    }
    history.push({ role: "assistant", content });

    const toolCalls = content.filter((block) => block.type === "tool_use");
    if (stopReason !== "tool_use" || !toolCalls.length) return;

    const results = [];
    for (const call of toolCalls) {
      addTurn("tool", describeToolCall(call.name, call.input));
      const { content: text, is_error } = await runPageTool(call.name, call.input);
      results.push({ type: "tool_result", tool_use_id: call.id, content: text, is_error });
    }
    history.push({ role: "user", content: results });
  }

  addTurn("error", `Stopped after ${MAX_TOOL_ROUNDS} tool rounds without a final answer.`);
}

async function send() {
  const text = ui.prompt.value.trim();
  if (!text || busy) return;

  const settings = currentSettings();
  if (!settings.model) {
    setNote("Connect to the proxy first.", "error");
    ui.settings.hidden = false;
    return;
  }

  busy = true;
  ui.send.disabled = true;
  ui.prompt.value = "";
  addTurn("user", text);
  history.push({ role: "user", content: [{ type: "text", text }] });

  try {
    await runConversation(settings);
  } catch (error) {
    addTurn("error", error instanceof Error ? error.message : String(error));
  } finally {
    busy = false;
    ui.send.disabled = false;
    ui.prompt.focus();
  }
}

// ---------- wiring ----------

ui.settingsToggle.addEventListener("click", () => {
  ui.settings.hidden = !ui.settings.hidden;
});

ui.settings.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (await connect(currentSettings())) ui.settings.hidden = true;
});

ui.model.addEventListener("change", async () => {
  await saveSettings(currentSettings());
});

ui.pageTools.addEventListener("change", async () => {
  await saveSettings(currentSettings());
});

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  void send();
});

ui.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void send();
  }
});

ui.clear.addEventListener("click", showEmptyState);

(async function start() {
  const settings = await loadSettings();
  ui.baseUrl.value = settings.baseUrl;
  ui.authToken.value = settings.authToken;
  ui.pageTools.checked = settings.pageTools;
  showEmptyState();

  // Auto-connect: the common case is a proxy already running at the saved URL,
  // and making the user press Connect every time the panel reopens is friction
  // for nothing. Failure just opens the settings pane with the reason.
  if (!(await connect(settings))) ui.settings.hidden = false;
})();

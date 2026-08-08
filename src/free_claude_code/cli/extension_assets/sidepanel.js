// Side panel: a Messages-API client for the local Free Claude Code proxy.
//
// The panel calls the proxy directly. It is an extension page, so it runs on
// the extension origin and Chrome grants it the host_permissions CORS bypass;
// no preflight is sent and the proxy needs no CORS middleware. Routing this
// through the service worker instead would only add a message hop.

import { PAGE_TOOLS, runPageTool } from "./page_tools.js";
import { SHELL_TOOL, bridgeStatus, runShellCommand } from "./shell_tool.js";

const SETTINGS_KEY = "fcc.connection";
// Must match the server's default port. Pointing at a port nothing serves made
// a fresh install fail with "Is fcc-server running?" while it was running.
const DEFAULT_BASE_URL = "http://127.0.0.1:8082";
const MAX_TOKENS = 8192;
// A tool_use turn that never terminates would loop against the provider until
// the user's quota ran out. Real page-debugging turns settle in two or three.
const MAX_TOOL_ROUNDS = 12;

const BASE_PROMPT =
  "You are Free Claude Code running in a Chrome side panel, helping the user with the web " +
  "page they are looking at. When they refer to 'this page', 'the console', or 'the error', " +
  "use your tools to look rather than asking them to paste. Read the smallest region that " +
  "answers the question -- prefer a CSS selector over the whole document. You cannot click, " +
  "type, or navigate; say so plainly if asked, and tell the user what to do instead. " +
  "Be concise: this panel is narrow.";

const SHELL_PROMPT =
  " You can also run shell commands on the user's machine with run_command. Each one is " +
  "shown to them for approval before it runs, so send one purposeful command at a time and " +
  "explain what you expect from it. Assume nothing about the result until you see it.";

function systemPrompt() {
  return shell.usable ? BASE_PROMPT + SHELL_PROMPT : BASE_PROMPT;
}

const ui = {
  status: document.getElementById("status"),
  settings: document.getElementById("settings"),
  settingsToggle: document.getElementById("settings-toggle"),
  settingsNote: document.getElementById("settings-note"),
  baseUrl: document.getElementById("base-url"),
  authToken: document.getElementById("auth-token"),
  model: document.getElementById("model"),
  pageTools: document.getElementById("page-tools"),
  shellState: document.getElementById("shell-state"),
  transcript: document.getElementById("transcript"),
  composer: document.getElementById("composer"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  clear: document.getElementById("clear"),
};

/** Conversation history in Messages API shape, replayed on every request. */
let history = [];
let busy = false;

/**
 * Command bridge state, probed once at startup.
 *
 * `usable` gates whether run_command is advertised at all. Offering a tool that
 * cannot work wastes a round trip and teaches the model to expect a shell that
 * is not there -- the bridge is unregistered by default, and disabled even when
 * registered until BROWSER_SHELL_ENABLED is set.
 */
let shell = { usable: false, root: "" };

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

/** Single owner of sheet visibility, so the toggle never lies to assistive tech. */
function setSettingsOpen(open) {
  ui.settings.hidden = !open;
  ui.settingsToggle.setAttribute("aria-expanded", String(open));
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
  // The chip said "not connected" for the whole round trip, which is a
  // different claim from "trying". Every exit below overwrites it.
  setStatus("connecting", "busy");
  setNote("Reaching the proxy…");
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

/**
 * Openers that each exercise a tool the panel actually has.
 *
 * A first-run user cannot tell from a text box that the model can look at the
 * tab, and the one thing worth teaching here is that they do not have to paste.
 * They fill the composer rather than sending, so the wording stays editable.
 */
const OPENERS = [
  "What is this page for?",
  "Any errors in the console?",
  "Summarize the main content",
];

function showEmptyState() {
  history = [];
  ui.transcript.replaceChildren();

  const node = document.createElement("div");
  node.className = "empty";

  // States the capability rather than repeating the composer's placeholder:
  // that the model can look at the tab is the one thing a first-run user has
  // no way to guess, and the box below already says what to do with it.
  const title = document.createElement("p");
  title.className = "empty-title";
  title.textContent = "Claude can see this tab";

  const body = document.createElement("p");
  body.className = "empty-body";
  body.textContent = "Ask about anything on the page, including its console output.";

  const openers = document.createElement("div");
  openers.className = "empty-suggestions";
  for (const opener of OPENERS) {
    const button = document.createElement("button");
    button.className = "suggestion";
    button.type = "button";
    button.textContent = opener;
    button.addEventListener("click", () => {
      ui.prompt.value = opener;
      resizePrompt();
      ui.prompt.focus();
    });
    openers.append(button);
  }

  node.append(title, body, openers);
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
    system: systemPrompt(),
    messages: history,
    stream: true,
  };
  const tools = [...(settings.pageTools ? PAGE_TOOLS : []), ...(shell.usable ? [SHELL_TOOL] : [])];
  if (tools.length) body.tools = tools;

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

/**
 * Show a command and wait for the user to allow or refuse it.
 *
 * This is the control the shell bridge rests on. Everything else -- the
 * allowed_origins pin, BROWSER_SHELL_ENABLED, the directory confinement --
 * limits the blast radius; this is what decides whether a command runs at all.
 * So it renders the exact string that will be executed, and defaults to nothing
 * happening if the user simply closes the panel.
 */
function requestApproval({ command, cwd }) {
  return new Promise((resolve) => {
    const card = document.createElement("div");
    card.className = "turn approval";

    const line = document.createElement("code");
    line.className = "approval-command";
    line.textContent = command;

    const actions = document.createElement("div");
    actions.className = "approval-actions";
    const deny = document.createElement("button");
    deny.className = "ghost";
    deny.type = "button";
    deny.textContent = "Deny";
    const allow = document.createElement("button");
    allow.type = "button";
    allow.textContent = "Run";

    // The card's heading reports the outcome, so nothing else has to change:
    // rewriting the body left the card saying "Denied" twice.
    const settle = (approved) => {
      actions.remove();
      card.dataset.outcome = approved ? "approved" : "denied";
      resolve(approved);
    };
    deny.addEventListener("click", () => settle(false));
    allow.addEventListener("click", () => settle(true));

    actions.append(deny, allow);
    // The directory only appears when there is one to name.
    if (cwd) {
      const where = document.createElement("p");
      where.className = "approval-label";
      where.textContent = cwd;
      card.append(where);
    }
    card.append(line, actions);
    ui.transcript.append(card);
    ui.transcript.scrollTop = ui.transcript.scrollHeight;
    allow.focus();
  });
}

async function runTool(call) {
  if (call.name === SHELL_TOOL.name) {
    return runShellCommand(call.input, { requestApproval });
  }
  addTurn("tool", describeToolCall(call.name, call.input));
  return runPageTool(call.name, call.input);
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
      const { content: text, is_error } = await runTool(call);
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
    setSettingsOpen(true);
    return;
  }

  busy = true;
  ui.send.disabled = true;
  ui.prompt.value = "";
  resizePrompt();
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

/** Grow the composer with its content, up to the max-height the stylesheet sets. */
function resizePrompt() {
  ui.prompt.style.height = "auto";
  ui.prompt.style.height = `${ui.prompt.scrollHeight}px`;
}

ui.settingsToggle.addEventListener("click", () => {
  setSettingsOpen(ui.settings.hidden);
});

ui.settings.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (await connect(currentSettings())) setSettingsOpen(false);
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

ui.prompt.addEventListener("input", resizePrompt);

ui.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void send();
  }
});

ui.clear.addEventListener("click", showEmptyState);

async function probeBridge() {
  const status = await bridgeStatus();
  shell = { usable: status.available && status.enabled, root: status.root };

  if (shell.usable) {
    ui.shellState.textContent = `Shell commands run under ${shell.root}, and you approve each one.`;
    return;
  }
  if (status.available) {
    ui.shellState.textContent =
      "Shell bridge registered but off. Set BROWSER_SHELL_ENABLED=true in ~/.fcc/.env.";
    return;
  }
  // chrome.runtime.id is this extension's ID, which is otherwise something the
  // user has to go and copy off the chrome://extensions card by hand.
  ui.shellState.textContent =
    `Shell commands off. To enable: fcc-extension install --extension-id ${chrome.runtime.id}`;
}

(async function start() {
  const settings = await loadSettings();
  ui.baseUrl.value = settings.baseUrl;
  ui.authToken.value = settings.authToken;
  ui.pageTools.checked = settings.pageTools;
  showEmptyState();
  resizePrompt();
  await probeBridge();

  // Auto-connect: the common case is a proxy already running at the saved URL,
  // and making the user press Connect every time the panel reopens is friction
  // for nothing. Failure just opens the settings pane with the reason.
  if (!(await connect(settings))) setSettingsOpen(true);
})();

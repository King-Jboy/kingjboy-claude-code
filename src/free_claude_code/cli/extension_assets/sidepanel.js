// Side panel: a Messages-API client for the local Free Claude Code proxy.
//
// The panel calls the proxy directly. It is an extension page, so it runs on
// the extension origin and Chrome grants it the host_permissions CORS bypass;
// no preflight is sent and the proxy needs no CORS middleware. Routing this
// through the service worker instead would only add a message hop.

import { markdownNodes } from "./markdown.js";
import { PAGE_APPROVALS, PAGE_TOOLS, runPageTool } from "./page_tools.js";
import { SHELL_TOOL, bridgeStatus, runShellCommand } from "./shell_tool.js";

const SETTINGS_KEY = "fcc.connection";
const CONVERSATION_KEY = "fcc.conversation";
// Storage carries the replayable history, screenshots included; past this the
// oldest exchanges are dropped rather than risking the 10MB quota mid-chat.
const MAX_STORED_BYTES = 4_000_000;
// Must match the server's default port. Pointing at a port nothing serves made
// a fresh install fail with "Is fcc-server running?" while it was running.
const DEFAULT_BASE_URL = "http://127.0.0.1:8082";
const MAX_TOKENS = 8192;
// A tool_use turn that never terminates would loop against the provider until
// the user's quota ran out. Real page-debugging turns settle in two or three.
const MAX_TOOL_ROUNDS = 12;
// What a tool result shows before it is expanded. Results can be a whole page;
// the transcript would otherwise become a wall of output nobody scrolled past.
const OUTPUT_PREVIEW_CHARS = 400;

const BASE_PROMPT =
  "You are Free Claude Code running in a Chrome side panel, helping the user with the web " +
  "page they are looking at. When they refer to 'this page', 'the console', or 'the error', " +
  "use your tools to look rather than asking them to paste. Read the smallest region that " +
  "answers the question -- prefer a CSS selector over the whole document. You can click and " +
  "type on the page with the click and type_text tools; the user approves each action " +
  "before it runs, so say what you are about to do and why, one action at a time. You " +
  "cannot navigate to another URL or reload the page; say so plainly if asked, and tell " +
  "the user what to do instead. Be concise: this panel is narrow.";

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
  usage: document.getElementById("usage"),
  pinBar: document.getElementById("pin-bar"),
  pinTitle: document.getElementById("pin-title"),
  rePin: document.getElementById("re-pin"),
  composer: document.getElementById("composer"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
  clear: document.getElementById("clear"),
};

/** Conversation history in Messages API shape, replayed on every request. */
let history = [];
let busy = false;
/**
 * Aborts the in-flight turn. Null while idle. A stop cannot kill a native
 * shell command already running -- it takes effect the moment that returns.
 */
let stopController = null;
/** Tokens billed across this conversation's requests, from message_delta. */
let usageTotals = { input: 0, output: 0 };

/**
 * The tab this conversation's page tools target, pinned from the first message.
 *
 * Without a pin, tools resolve "the active tab" per call, so switching tabs
 * mid-chat silently retargeted the model's view of "this page". The pin is
 * per-conversation: Clear drops it, Change tab re-pins to whatever is now
 * active, and if the tab closes the pin is dropped and the bar says so --
 * tools fall back to the active tab only from that point, never silently
 * beside a stale claim.
 */
let pinnedTab = null;

function renderPinBar() {
  ui.pinBar.hidden = !pinnedTab;
  if (pinnedTab) ui.pinTitle.textContent = `Watching ${pinnedTab.title || "a tab"}`;
}

async function pinActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  pinnedTab = { id: tab.id, title: tab.title ?? tab.url ?? "" };
  renderPinBar();
}

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
  pinnedTab = null;
  renderPinBar();
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
 * become valid JSON once concatenated, thinking as `thinking_delta` fragments
 * with a final `signature_delta`. `onDelta` fires per fragment with the kind
 * that grew, so the panel can render while the response is still arriving.
 */
async function streamAssistantTurn(response, onDelta, signal) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const blocks = [];
  let buffer = "";
  let stopReason = null;
  let usage = { input: 0, output: 0 };

  try {
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
          if (block.type === "tool_use") {
            blocks[event.index] = { type: "tool_use", id: block.id, name: block.name, json: "" };
          } else if (block.type === "thinking") {
            blocks[event.index] = { type: "thinking", text: "", signature: "" };
          } else if (block.type === "redacted_thinking") {
            blocks[event.index] = { type: "redacted_thinking", data: block.data ?? "" };
          } else {
            blocks[event.index] = { type: "text", text: "" };
          }
        } else if (event.type === "content_block_delta") {
          const block = blocks[event.index];
          if (!block) continue;
          if (event.delta.type === "text_delta") {
            block.text += event.delta.text;
            onDelta({ text: event.delta.text });
          } else if (event.delta.type === "input_json_delta") {
            block.json += event.delta.partial_json;
          } else if (event.delta.type === "thinking_delta") {
            block.text += event.delta.thinking;
            onDelta({ thinking: event.delta.thinking });
          } else if (event.delta.type === "signature_delta") {
            block.signature += event.delta.signature;
          }
        } else if (event.type === "message_delta") {
          stopReason = event.delta?.stop_reason ?? stopReason;
          if (Number.isFinite(event.usage?.input_tokens)) usage.input = event.usage.input_tokens;
          if (Number.isFinite(event.usage?.output_tokens)) usage.output = event.usage.output_tokens;
        } else if (event.type === "error") {
          throw new Error(event.error?.message ?? "The proxy reported a stream error.");
        }
      }
    }
  } catch (error) {
    // An aborted stream still finalizes what arrived: the partial reply is
    // context the conversation should keep. Anything else is a real failure.
    if (!signal?.aborted) throw error;
  }

  const content = blocks
    .filter(Boolean)
    .map((block) => {
      if (block.type === "tool_use") {
        let input = {};
        try {
          // An empty-object tool call streams as "" rather than "{}".
          input = block.json ? JSON.parse(block.json) : {};
        } catch {
          input = {};
        }
        return { type: "tool_use", id: block.id, name: block.name, input };
      }
      if (block.type === "thinking") {
        // The signature is replayed with the thinking block so reasoning-based
        // providers can verify the turn chain; omit it when none arrived.
        const thinking = { type: "thinking", thinking: block.text };
        if (block.signature) thinking.signature = block.signature;
        return thinking;
      }
      if (block.type === "redacted_thinking") {
        return { type: "redacted_thinking", data: block.data };
      }
      return { type: "text", text: block.text };
    })
    // A thinking-only or text-only stream used to leave empty text blocks in
    // history, which every later request replayed for nothing.
    .filter((block) => block.type !== "text" || block.text !== "");

  return { content, stopReason, usage };
}

async function requestTurn(settings, signal) {
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
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Proxy returned HTTP ${response.status}. ${detail.slice(0, 400)}`);
  }
  return response;
}

// ---------- assistant rendering ----------

/**
 * One in-flight assistant turn: the transcript node plus the raw text buffers
 * it renders from. Deltas accumulate into the buffers; rendering is scheduled
 * per animation frame, so a burst of fragments costs one repaint.
 */
function openAssistantTurn() {
  const node = addTurn("assistant", "");
  const body = document.createElement("div");
  body.className = "markdown";
  node.append(body);
  return { node, text: "", thinking: "", frame: 0 };
}

function renderAssistant(turn) {
  turn.node.replaceChildren();
  if (turn.thinking) turn.node.append(thinkingSection(turn.thinking));
  const body = document.createElement("div");
  body.className = "markdown";
  body.append(markdownNodes(turn.text));
  turn.node.append(body);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function scheduleAssistantRender(turn) {
  if (turn.frame) return;
  turn.frame = requestAnimationFrame(() => {
    turn.frame = 0;
    renderAssistant(turn);
  });
}

/**
 * Thinking arrives ahead of the reply on reasoning models. It is real content
 * -- kept in history, replayed with its signature -- but it is not what the
 * user asked for, so it sits collapsed unless opened.
 */
function thinkingSection(text) {
  const details = document.createElement("details");
  details.className = "thinking";
  const summary = document.createElement("summary");
  summary.textContent = "Thinking";
  const body = document.createElement("div");
  body.className = "thinking-body";
  body.textContent = text;
  details.append(summary, body);
  return details;
}

/**
 * Show what a tool actually returned under its call line.
 *
 * Hiding outputs made every tool a black box: the model reacted to material
 * the user could not see, which is exactly where distrust in the panel came
 * from. Shown clipped by default because results are often whole pages; the
 * full text is one click away and the model always received all of it.
 */
function showToolOutput(node, outcome) {
  const box = document.createElement("div");
  box.className = "tool-output";
  if (outcome.is_error) box.dataset.error = "";

  if (outcome.image) {
    const shot = document.createElement("img");
    shot.className = "tool-shot";
    shot.alt = "Screenshot captured for the model";
    shot.src = `data:${outcome.image.source.media_type};base64,${outcome.image.source.data}`;
    box.append(shot);
  }

  const text = String(outcome.content ?? "");
  if (text.trim()) {
    const clipped = text.length > OUTPUT_PREVIEW_CHARS;
    const body = document.createElement("div");
    body.className = "tool-output-text";
    body.textContent = clipped ? `${text.slice(0, OUTPUT_PREVIEW_CHARS)}…` : text;
    box.append(body);

    if (clipped) {
      const toggle = document.createElement("button");
      toggle.className = "tool-output-toggle";
      toggle.type = "button";
      toggle.textContent = "Show all";
      toggle.addEventListener("click", () => {
        const open = box.dataset.open !== undefined;
        if (open) {
          delete box.dataset.open;
          body.textContent = `${text.slice(0, OUTPUT_PREVIEW_CHARS)}…`;
          toggle.textContent = "Show all";
        } else {
          box.dataset.open = "";
          body.textContent = text;
          toggle.textContent = "Show less";
        }
      });
      box.append(toggle);
    }
  }

  node.append(box);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

// ---------- usage ----------

function formatTokens(count) {
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
}

/**
 * What this conversation has spent, from each request's final usage frame.
 *
 * Input is summed across requests rather than read from the last one, because
 * every round replays the whole history: the sum is what actually billed
 * against the key pool, which is the number the user watches.
 */
function renderUsage() {
  if (!usageTotals.input && !usageTotals.output) {
    ui.usage.hidden = true;
    return;
  }
  ui.usage.hidden = false;
  ui.usage.textContent =
    `${formatTokens(usageTotals.input)} in · ${formatTokens(usageTotals.output)} out this conversation`;
}

// ---------- persistence ----------

let saveTimer = 0;

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => void saveConversation(), 500);
}

/**
 * Drop leading messages until the first survivor is a plain user turn.
 *
 * Replay has two invariants: the first message must be from the user, and a
 * tool_result may only follow the assistant tool_use it answers. Cutting at a
 * text-bearing user message satisfies both at once.
 */
function trimLeading(messages) {
  for (let cut = 1; cut < messages.length; cut += 1) {
    const candidate = messages[cut];
    if (candidate?.role === "user" && candidate.content?.[0]?.type === "text") {
      return messages.slice(cut);
    }
  }
  return [];
}

async function saveConversation() {
  if (!history.length) return;
  let stored = history;
  while (JSON.stringify(stored).length > MAX_STORED_BYTES && stored.length > 2) {
    stored = trimLeading(stored);
    if (!stored.length) return;
  }
  await chrome.storage.local.set({
    [CONVERSATION_KEY]: {
      history: stored,
      pinnedTabId: pinnedTab?.id ?? null,
      pinnedTabTitle: pinnedTab?.title ?? "",
      usage: usageTotals,
    },
  });
}

/** Rebuild the transcript from replayable history after a panel reopen. */
function renderHistoryToTranscript() {
  ui.transcript.replaceChildren();
  for (const message of history) {
    const blocks = Array.isArray(message.content) ? message.content : [];
    if (message.role === "user") {
      for (const block of blocks) {
        if (block.type === "text") {
          addTurn("user", block.text);
        } else if (block.type === "tool_result") {
          const node = addTurn("tool", "result");
          showToolOutput(node, { content: block.content, is_error: block.is_error });
        } else if (block.type === "image") {
          const node = addTurn("tool", "screenshot");
          showToolOutput(node, { content: "", image: block });
        }
      }
      continue;
    }
    // One assistant turn holds its thinking and prose; its tool calls become
    // chips, exactly as they appeared live. Settled approval cards are not
    // rebuilt -- the chips already say what ran.
    const thinking = blocks.filter((b) => b.type === "thinking").map((b) => b.thinking).join("\n");
    const text = blocks.filter((b) => b.type === "text").map((b) => b.text).join("\n\n");
    if (thinking || text) {
      const turn = openAssistantTurn();
      turn.thinking = thinking;
      turn.text = text;
      renderAssistant(turn);
    }
    for (const block of blocks) {
      if (block.type === "tool_use") addTurn("tool", describeToolCall(block.name, block.input));
    }
  }
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

async function restoreConversation() {
  const stored = (await chrome.storage.local.get(CONVERSATION_KEY))[CONVERSATION_KEY];
  if (!Array.isArray(stored?.history) || !stored.history.length) return;

  history = stored.history;
  usageTotals = stored.usage ?? { input: 0, output: 0 };
  renderUsage();

  if (Number.isInteger(stored.pinnedTabId)) {
    pinnedTab = { id: stored.pinnedTabId, title: stored.pinnedTabTitle ?? "" };
    renderPinBar();
    // The tab can have closed while the panel was away; a pin claiming a dead
    // id is worse than no pin.
    chrome.tabs.get(stored.pinnedTabId).catch(() => {
      pinnedTab = null;
      renderPinBar();
    });
  }

  renderHistoryToTranscript();
}

// ---------- turn loop ----------

function describeToolCall(name, input) {
  const detail = Object.entries(input ?? {})
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(" ");
  return detail ? `↳ ${name} ${detail}` : `↳ ${name}`;
}

/**
 * Show an action and wait for the user to allow or refuse it.
 *
 * This is the control both the shell bridge and the page actions rest on.
 * Everything else -- the allowed_origins pin, BROWSER_SHELL_ENABLED, the
 * directory confinement -- limits the blast radius; this is what decides
 * whether an action runs at all. So it renders the exact thing that will
 * happen, and defaults to nothing happening if the user simply closes the
 * panel.
 */
function requestApproval({ detail, label = "", allow = "Run" }) {
  return new Promise((resolve) => {
    const card = document.createElement("div");
    card.className = "turn approval";

    if (label) {
      const where = document.createElement("p");
      where.className = "approval-label";
      where.textContent = label;
      card.append(where);
    }

    const line = document.createElement("code");
    line.className = "approval-command";
    line.textContent = detail;

    const actions = document.createElement("div");
    actions.className = "approval-actions";
    const deny = document.createElement("button");
    deny.className = "ghost";
    deny.type = "button";
    deny.textContent = "Deny";
    const allowButton = document.createElement("button");
    allowButton.type = "button";
    allowButton.textContent = allow;

    // The card's heading reports the outcome, so nothing else has to change:
    // rewriting the body left the card saying "Denied" twice.
    const settle = (approved) => {
      actions.remove();
      card.dataset.outcome = approved ? "approved" : "denied";
      resolve(approved);
    };
    deny.addEventListener("click", () => settle(false));
    allowButton.addEventListener("click", () => settle(true));

    actions.append(deny, allowButton);
    card.append(line, actions);
    ui.transcript.append(card);
    ui.transcript.scrollTop = ui.transcript.scrollHeight;
    allowButton.focus();
  });
}

async function runTool(call) {
  const node = addTurn("tool", describeToolCall(call.name, call.input));

  let outcome;
  if (call.name === SHELL_TOOL.name) {
    outcome = await runShellCommand(call.input, {
      requestApproval: ({ command, cwd }) => requestApproval({ detail: command, label: cwd }),
    });
  } else {
    const approval = PAGE_APPROVALS[call.name]?.(call.input ?? {});
    if (approval === null) {
      outcome = {
        content: `${call.name} needs a selector${call.name === "type_text" ? " and text" : ""}.`,
        is_error: true,
      };
    } else if (approval) {
      const allowed = await requestApproval({
        ...approval,
        label: pinnedTab ? `on ${pinnedTab.title}` : "",
        allow: "Allow",
      });
      if (!allowed) {
        outcome = { content: "The user declined this action.", is_error: true };
      }
    }
    if (!outcome) {
      outcome = await runPageTool(call.name, call.input, { tabId: pinnedTab?.id });
    }
  }

  showToolOutput(node, outcome);
  return outcome;
}

async function runConversation(settings, signal) {
  for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
    if (signal?.aborted) return;
    const response = await requestTurn(settings, signal);

    const turn = openAssistantTurn();
    const { content, stopReason, usage } = await streamAssistantTurn(
      response,
      (chunk) => {
        if (chunk.text !== undefined) turn.text += chunk.text;
        if (chunk.thinking !== undefined) turn.thinking += chunk.thinking;
        scheduleAssistantRender(turn);
      },
      signal,
    );
    renderAssistant(turn);
    if (usage.input) usageTotals.input += usage.input;
    if (usage.output) usageTotals.output += usage.output;
    renderUsage();

    if (signal?.aborted) {
      // Keep whatever arrived, minus tool calls: their inputs are half
      // streamed and a dangling tool_use with no result breaks replay.
      const partial = content.filter((block) => block.type !== "tool_use");
      if (partial.length) history.push({ role: "assistant", content: partial });
      addTurn("note", "Stopped. The reply so far was kept.");
      return;
    }

    if (!content.length) {
      addTurn("error", "The model returned an empty response.");
      return;
    }
    history.push({ role: "assistant", content });

    const toolCalls = content.filter((block) => block.type === "tool_use");
    if (stopReason !== "tool_use" || !toolCalls.length) return;

    const results = [];
    const images = [];
    for (const call of toolCalls) {
      if (signal?.aborted) break;
      const { content: text, is_error, image } = await runTool(call);
      results.push({ type: "tool_result", tool_use_id: call.id, content: text, is_error });
      if (image) images.push(image);
    }
    if (!results.length) return;
    // Every tool_result must satisfy its tool_use before any other block
    // appears: the proxy emits one provider tool message per result and only
    // then a user turn for the images, which is the one ordering both APIs
    // accept.
    history.push({ role: "user", content: [...results, ...images] });
    if (signal?.aborted) {
      addTurn("note", "Stopped. No further steps will run.");
      return;
    }
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
  stopController = new AbortController();
  ui.send.disabled = true;
  ui.stop.hidden = false;
  ui.prompt.value = "";
  resizePrompt();
  addTurn("user", text);
  history.push({ role: "user", content: [{ type: "text", text }] });
  // The conversation targets the tab the user was looking at when they started
  // it; without this, tools would follow their later tab switches instead.
  if (!pinnedTab) await pinActiveTab();

  try {
    await runConversation(settings, stopController.signal);
  } catch (error) {
    if (stopController.signal.aborted) {
      addTurn("note", "Stopped before a reply arrived.");
    } else {
      addTurn("error", error instanceof Error ? error.message : String(error));
    }
  } finally {
    busy = false;
    stopController = null;
    ui.send.disabled = false;
    ui.stop.hidden = true;
    ui.prompt.focus();
    scheduleSave();
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

ui.clear.addEventListener("click", () => {
  showEmptyState();
  usageTotals = { input: 0, output: 0 };
  renderUsage();
  clearTimeout(saveTimer);
  void chrome.storage.local.remove(CONVERSATION_KEY);
});

ui.stop.addEventListener("click", () => stopController?.abort());

ui.rePin.addEventListener("click", () => void pinActiveTab());

// Dropping the pin on close has to be explicit: the bar claims a specific tab,
// and a closed tab would leave that claim pointing at whatever Chrome reuses
// the id for in queries that only check it is alive.
chrome.tabs.onRemoved.addListener((tabId) => {
  if (pinnedTab?.id === tabId) {
    pinnedTab = null;
    renderPinBar();
  }
});

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
  await restoreConversation();
  await probeBridge();

  // Auto-connect: the common case is a proxy already running at the saved URL,
  // and making the user press Connect every time the panel reopens is friction
  // for nothing. Failure just opens the settings pane with the reason.
  if (!(await connect(settings))) setSettingsOpen(true);
})();

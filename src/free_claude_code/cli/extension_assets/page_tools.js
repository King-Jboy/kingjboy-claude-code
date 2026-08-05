// Anthropic tool definitions for the active tab, and the code that runs them.
//
// Every executor injects through chrome.scripting rather than asking a content
// script to fetch anything: in MV3 a content script inherits the *page's* CORS
// posture, so a page could not reach the proxy even though the extension can.
// Injecting keeps the network boundary in the panel and the DOM boundary in
// the page, which is also the split the permissions model expects.

// Tool results are prompt text. An unbounded document would spend the whole
// context window on one call, so results are truncated and say that they were.
const DEFAULT_MAX_CHARS = 12000;
const HARD_MAX_CHARS = 120000;

export const PAGE_TOOLS = [
  {
    name: "page_info",
    description:
      "Get the URL, title, and viewport size of the tab the user is currently looking at. " +
      "Call this first when the user refers to 'this page' and you do not yet know what it is.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "read_page",
    description:
      "Read the content of the active tab. Use format 'text' for reading and understanding " +
      "content, and 'html' when the user asks about markup, structure, attributes, or styling. " +
      "Pass a CSS selector to read one region instead of the whole document -- prefer this, " +
      "because whole pages are large.",
    input_schema: {
      type: "object",
      properties: {
        selector: {
          type: "string",
          description: "CSS selector to read. Omit to read the whole document.",
        },
        format: {
          type: "string",
          enum: ["text", "html"],
          description: "Rendered text, or raw markup. Defaults to text.",
        },
        max_chars: {
          type: "integer",
          description: `Truncation limit. Defaults to ${DEFAULT_MAX_CHARS}.`,
        },
      },
    },
  },
  {
    name: "read_console",
    description:
      "Read console output and uncaught errors recorded on the active tab since it loaded. " +
      "Use this to debug a page the user says is broken. Entries logged before the extension " +
      "was installed, or on a tab opened before then, will not appear -- ask the user to " +
      "reload the tab if the log looks empty when it should not be.",
    input_schema: {
      type: "object",
      properties: {
        levels: {
          type: "array",
          items: { type: "string", enum: ["log", "info", "warn", "error", "debug"] },
          description: "Only return these levels. Omit for all levels.",
        },
        limit: {
          type: "integer",
          description: "Return at most this many of the most recent entries. Defaults to 100.",
        },
      },
    },
  },
];

// chrome:// pages, the Web Store, and other extensions are closed to scripting
// by policy. Failing here with the reason beats a bare "Cannot access contents".
const CLOSED_SCHEMES = ["chrome:", "chrome-extension:", "devtools:", "about:", "edge:"];
const CLOSED_HOSTS = ["chromewebstore.google.com", "chrome.google.com"];

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("No active tab.");

  let url;
  try {
    url = new URL(tab.url ?? "");
  } catch {
    throw new Error("The active tab has no readable URL yet. Wait for it to load.");
  }
  if (CLOSED_SCHEMES.includes(url.protocol) || CLOSED_HOSTS.includes(url.hostname)) {
    throw new Error(
      `Chrome does not allow extensions to read ${url.protocol}//${url.hostname} pages. ` +
        "Switch to an ordinary http or https tab.",
    );
  }
  return tab;
}

async function inject(tab, { world = "ISOLATED", func, args = [] }) {
  const [result] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    world,
    func,
    args,
  });
  if (!result) throw new Error("The page did not respond to injection.");
  return result.result;
}

function clamp(requested) {
  const value = Number.isInteger(requested) ? requested : DEFAULT_MAX_CHARS;
  return Math.max(200, Math.min(value, HARD_MAX_CHARS));
}

function truncated(text, limit) {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}\n\n[truncated ${text.length - limit} more characters]`;
}

const EXECUTORS = {
  async page_info() {
    const tab = await activeTab();
    const viewport = await inject(tab, {
      func: () => ({ width: window.innerWidth, height: window.innerHeight }),
    });
    return JSON.stringify({
      url: tab.url,
      title: tab.title ?? "",
      viewport: `${viewport.width}x${viewport.height}`,
    });
  },

  async read_page({ selector, format, max_chars }) {
    const tab = await activeTab();
    const wantsHtml = format === "html";
    const payload = await inject(tab, {
      args: [selector ?? null, wantsHtml],
      func: (target, asHtml) => {
        const node = target ? document.querySelector(target) : document.documentElement;
        if (!node) return { missing: true };
        return { content: asHtml ? node.outerHTML : (node.innerText ?? node.textContent ?? "") };
      },
    });
    if (payload.missing) return `No element matches the selector ${selector}.`;
    return truncated(payload.content, clamp(max_chars));
  },

  async read_console({ levels, limit }) {
    const tab = await activeTab();
    const entries = await inject(tab, {
      world: "MAIN",
      func: () => window.__fccConsoleBuffer ?? null,
    });
    if (entries === null) {
      return (
        "The console recorder is not installed on this tab. It attaches at page load, so " +
        "reload the tab and try again."
      );
    }

    const wanted = Array.isArray(levels) && levels.length ? new Set(levels) : null;
    const matching = wanted ? entries.filter((entry) => wanted.has(entry.level)) : entries;
    const count = Number.isInteger(limit) && limit > 0 ? limit : 100;
    const recent = matching.slice(-count);
    if (!recent.length) return "No console entries recorded on this tab.";

    return recent
      .map((entry) => `[${entry.level}] ${new Date(entry.at).toISOString()} ${entry.text}`)
      .join("\n");
  },
};

export async function runPageTool(name, input) {
  const executor = EXECUTORS[name];
  if (!executor) return { content: `Unknown tool ${name}.`, is_error: true };
  try {
    return { content: await executor(input ?? {}), is_error: false };
  } catch (error) {
    // Tool failure is information the model can act on -- a closed page, a
    // selector that matched nothing. Report it as a result, not an exception,
    // so the turn continues instead of collapsing.
    return { content: error instanceof Error ? error.message : String(error), is_error: true };
  }
}

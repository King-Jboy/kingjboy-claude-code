// Anthropic tool definitions for the conversation's pinned tab, and the code
// that runs them.
//
// Every executor injects through chrome.scripting rather than asking a content
// script to fetch anything: in MV3 a content script inherits the *page's* CORS
// posture, so a page could not reach the proxy even though the extension can.
// Injecting keeps the network boundary in the panel and the DOM boundary in
// the page, which is also the split the permissions model expects.
//
// The tab is resolved per call from the id the panel pins to the conversation
// (falling back to the active tab when nothing is pinned). Querying the active
// tab fresh on every call made tools silently follow the user's tab switches
// mid-conversation, which retargeted the model's view of "this page" without
// anyone asking for that.

// Tool results are prompt text. An unbounded document would spend the whole
// context window on one call, so results are truncated and say that they were.
const DEFAULT_MAX_CHARS = 12000;
const HARD_MAX_CHARS = 120000;

// Screenshots are downscaled to this width before they are base64'd: a retina
// full-height capture is several megabytes of JPEG, and it buys no accuracy a
// 1280px-wide image does not already give a vision model.
const SCREENSHOT_MAX_WIDTH = 1280;

export const PAGE_TOOLS = [
  {
    name: "page_info",
    description:
      "Get the URL, title, and viewport size of the tab the conversation is watching. " +
      "Call this first when the user refers to 'this page' and you do not yet know what it is.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "read_page",
    description:
      "Read the content of the watched tab. Use format 'text' for reading and understanding " +
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
    name: "read_selection",
    description:
      "Read whatever text the user has highlighted on the watched tab, verbatim. Use this " +
      "when they refer to 'this part' or 'what I selected' instead of guessing a selector.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "read_console",
    description:
      "Read console output and uncaught errors recorded on the watched tab since it loaded. " +
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
  {
    name: "screenshot",
    description:
      "Capture what is visible in the watched tab and send it as an image you can see. Only " +
      "useful with a vision-capable model -- if you cannot see images, say so and use read_page " +
      "instead. Shows the current visual layout, including things that are not in the DOM text " +
      "such as canvas content or broken renders.",
    input_schema: { type: "object", properties: {} },
  },
  {
    name: "click",
    description:
      "Click an element on the watched tab. The user approves each click before it happens, " +
      "so say what you are clicking and why. Prefer read_page first to confirm the selector " +
      "matches the element you mean. Cannot open a new page or navigate away.",
    input_schema: {
      type: "object",
      properties: {
        selector: {
          type: "string",
          description: "CSS selector for the element to click. The first match is clicked.",
        },
      },
      required: ["selector"],
    },
  },
  {
    name: "type_text",
    description:
      "Type text into a field on the watched tab, optionally submitting its form. The user " +
      "approves each use before it runs, so say what you are filling in and why. Confirm the " +
      "selector with read_page first. Works on inputs, textareas, and contenteditable fields.",
    input_schema: {
      type: "object",
      properties: {
        selector: {
          type: "string",
          description: "CSS selector for the field to type into. The first match is used.",
        },
        text: {
          type: "string",
          description: "The text to put in the field. Replaces what is there.",
        },
        submit: {
          type: "boolean",
          description: "Also submit the field's form, as pressing Enter would. Defaults to false.",
        },
      },
      required: ["selector", "text"],
    },
  },
];

/**
 * Tools that change the page rather than read it, and the line each one shows
 * on its approval card. The card renders `detail` as the thing that will
 * happen, so it must state the action in full -- the model's surrounding
 * explanation is chat text, not consent. `null` means the input is unusable
 * and the caller should fail the call without showing a card.
 */
export const PAGE_APPROVALS = {
  click: ({ selector }) =>
    typeof selector === "string" && selector.trim()
      ? { detail: `click ${selector.trim()}` }
      : null,
  type_text: ({ selector, text, submit }) =>
    typeof selector === "string" && selector.trim() && typeof text === "string"
      ? {
          detail:
            `type ${JSON.stringify(text)} into ${selector.trim()}` +
            (submit ? " and submit" : ""),
        }
      : null,
};

// chrome:// pages, the Web Store, and other extensions are closed to scripting
// by policy. Failing here with the reason beats a bare "Cannot access contents".
const CLOSED_SCHEMES = ["chrome:", "chrome-extension:", "devtools:", "about:", "edge:"];
const CLOSED_HOSTS = ["chromewebstore.google.com", "chrome.google.com"];

function checkReadable(tab) {
  let url;
  try {
    url = new URL(tab.url ?? "");
  } catch {
    throw new Error("The tab has no readable URL yet. Wait for it to load.");
  }
  if (CLOSED_SCHEMES.includes(url.protocol) || CLOSED_HOSTS.includes(url.hostname)) {
    throw new Error(
      `Chrome does not allow extensions to read ${url.protocol}//${url.hostname} pages. ` +
        "Switch to an ordinary http or https tab.",
    );
  }
  return tab;
}

/** Resolve the tab this conversation's tools target: the pinned one, else active. */
async function targetTab(pinnedId) {
  if (Number.isInteger(pinnedId)) {
    let tab;
    try {
      tab = await chrome.tabs.get(pinnedId);
    } catch {
      throw new Error(
        "The tab this conversation was watching has been closed. Ask the user to pick a " +
          "new tab with the Change tab button, or to start a new conversation.",
      );
    }
    return checkReadable(tab);
  }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("No active tab.");
  return checkReadable(tab);
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

/** Decode a capture, shrink it if it is wider than `maxWidth`, re-encode as JPEG. */
function downscale(dataUrl, maxWidth) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      if (image.width <= maxWidth) {
        resolve(dataUrl);
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = maxWidth;
      canvas.height = Math.round(image.height * (maxWidth / image.width));
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", 0.85));
    };
    image.onerror = () => reject(new Error("The screenshot could not be decoded."));
    image.src = dataUrl;
  });
}

const EXECUTORS = {
  async page_info(_input, { tabId }) {
    const tab = await targetTab(tabId);
    const viewport = await inject(tab, {
      func: () => ({ width: window.innerWidth, height: window.innerHeight }),
    });
    return JSON.stringify({
      url: tab.url,
      title: tab.title ?? "",
      viewport: `${viewport.width}x${viewport.height}`,
    });
  },

  async read_page({ selector, format, max_chars }, { tabId }) {
    const tab = await targetTab(tabId);
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

  async read_selection(_input, { tabId }) {
    const tab = await targetTab(tabId);
    const selection = await inject(tab, {
      func: () => String(window.getSelection?.() ?? ""),
    });
    // A selection can span the page's whitespace; trimming the ends is enough,
    // because interior newlines and indentation may be exactly what got asked
    // about (code, a table).
    const text = selection.trim();
    if (!text) {
      return "Nothing is selected on the page right now. Ask the user to highlight it.";
    }
    return truncated(text, clamp());
  },

  async read_console({ levels, limit }, { tabId }) {
    const tab = await targetTab(tabId);
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

  async screenshot(_input, { tabId }) {
    const tab = await targetTab(tabId);
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: "jpeg",
      quality: 85,
    });
    const scaled = await downscale(dataUrl, SCREENSHOT_MAX_WIDTH);
    return {
      content: `Captured the visible area of "${tab.title ?? tab.url}" as an image.`,
      // An image block cannot live inside tool_result content -- the proxy
      // serializes that to text -- so it rides beside the tool_result in the
      // same user message, which the proxy forwards as a vision input.
      image: {
        type: "image",
        source: {
          type: "base64",
          media_type: "image/jpeg",
          data: scaled.slice(scaled.indexOf(",") + 1),
        },
      },
    };
  },

  async click({ selector }, { tabId }) {
    const tab = await targetTab(tabId);
    const outcome = await inject(tab, {
      args: [selector],
      func: (target) => {
        const el = document.querySelector(target);
        if (!el) return { missing: true };
        // Centering first means a follow-up screenshot shows what was clicked,
        // and lets pages that lazy-load on scroll react before the click.
        el.scrollIntoView({ block: "center", inline: "center" });
        el.click();
        return { clicked: true, tag: el.tagName.toLowerCase() };
      },
    });
    if (outcome.missing) return `No element matches the selector ${selector}.`;
    return `Clicked the <${outcome.tag}> matching ${selector}.`;
  },

  async type_text({ selector, text, submit }, { tabId }) {
    const tab = await targetTab(tabId);
    const outcome = await inject(tab, {
      args: [selector, String(text ?? ""), Boolean(submit)],
      func: (target, value, withSubmit) => {
        const el = document.querySelector(target);
        if (!el) return { missing: true };
        el.focus();
        if (el.isContentEditable) {
          el.textContent = value;
        } else if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
          // React and friends keep their own value model, and a plain
          // `el.value = x` is invisible to them: write through the prototype's
          // native setter, then announce it with the events they listen for.
          const proto =
            el instanceof HTMLTextAreaElement
              ? HTMLTextAreaElement.prototype
              : HTMLInputElement.prototype;
          const set = Object.getOwnPropertyDescriptor(proto, "value")?.set;
          if (set) set.call(el, value);
          else el.value = value;
        } else {
          return { unsupported: String(el.tagName).toLowerCase() };
        }
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        if (withSubmit) el.form?.requestSubmit();
        return { typed: true };
      },
    });
    if (outcome.missing) return `No element matches the selector ${selector}.`;
    if (outcome.unsupported) {
      return `<${outcome.unsupported}> is not a field that accepts typing.`;
    }
    return submit ? `Typed into ${selector} and submitted the form.` : `Typed into ${selector}.`;
  },
};

export async function runPageTool(name, input, { tabId } = {}) {
  const executor = EXECUTORS[name];
  if (!executor) return { content: `Unknown tool ${name}.`, is_error: true };
  try {
    // Executors return prompt text, or (screenshot only) an object that also
    // carries the image block beside it.
    const result = await executor(input ?? {}, { tabId });
    return typeof result === "string"
      ? { content: result, is_error: false }
      : { ...result, is_error: false };
  } catch (error) {
    // Tool failure is information the model can act on -- a closed page, a
    // selector that matched nothing. Report it as a result, not an exception,
    // so the turn continues instead of collapsing.
    return { content: error instanceof Error ? error.message : String(error), is_error: true };
  }
}

// Records console output and uncaught errors into a bounded per-page buffer.
//
// This runs in the page's MAIN world at document_start. The alternative,
// chrome.debugger, reports the same events without patching anything -- but it
// pins a permanent "extension is debugging this browser" banner to the window
// for as long as it is attached. Patching console costs us the entries a page
// logged from an inline script above our own injection point; the banner costs
// the user every tab they open. We take the missed entries.
//
// Nothing here contacts the extension. The buffer sits in the page until the
// side panel explicitly drains it for the tab the user is looking at.

(() => {
  const BUFFER_KEY = "__fccConsoleBuffer";
  const MAX_ENTRIES = 500;
  const MAX_TEXT = 4000;

  if (window[BUFFER_KEY]) return;

  const entries = [];
  const record = (level, parts) => {
    if (entries.length >= MAX_ENTRIES) entries.shift();
    entries.push({ level, at: Date.now(), text: parts.join(" ").slice(0, MAX_TEXT) });
  };

  // console arguments are arbitrary objects: circular, DOM nodes, proxies that
  // throw on property access. Anything that escapes here would break the very
  // page we are trying to observe, so every branch has to be total.
  const describe = (value) => {
    if (typeof value === "string") return value;
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (typeof value === "bigint") return `${value}n`;
    if (typeof value === "symbol" || typeof value === "function") {
      return String(value);
    }
    if (value instanceof Error) return `${value.name}: ${value.message}`;
    if (typeof value === "object") {
      const tag = value.nodeName ? `<${value.nodeName.toLowerCase()}>` : null;
      if (tag) return tag;
      try {
        const seen = new WeakSet();
        return JSON.stringify(value, (_key, nested) => {
          if (typeof nested !== "object" || nested === null) return nested;
          if (seen.has(nested)) return "[circular]";
          seen.add(nested);
          return nested;
        });
      } catch {
        return "[unserializable object]";
      }
    }
    return String(value);
  };

  const LEVELS = ["log", "info", "warn", "error", "debug"];
  for (const level of LEVELS) {
    const original = console[level];
    if (typeof original !== "function") continue;
    console[level] = function (...args) {
      try {
        record(level, args.map(describe));
      } catch {
        // Never let bookkeeping break the page's own logging.
      }
      return original.apply(this, args);
    };
  }

  window.addEventListener("error", (event) => {
    const where = event.filename ? ` (${event.filename}:${event.lineno})` : "";
    record("error", [`Uncaught ${event.message}${where}`]);
  });

  window.addEventListener("unhandledrejection", (event) => {
    record("error", [`Unhandled rejection: ${describe(event.reason)}`]);
  });

  Object.defineProperty(window, BUFFER_KEY, {
    value: entries,
    enumerable: false,
    configurable: false,
    writable: false,
  });
})();

// Markdown rendering for assistant replies, written by hand on purpose.
//
// The panel ships as static files with no build step, and the MV3 CSP blocks
// loading anything remote -- so there is no marked/remark to pull in. The
// subset models actually emit in a narrow chat panel is small: emphasis, code,
// fenced blocks, lists, headings, links.
//
// The reply being rendered is untrusted text shown in an extension page, so
// the renderer is escape-first: every character of input is HTML-escaped
// before any markup is introduced, and the only tags in the output are the
// ones this file writes. Attributes it does not control never exist -- link
// targets are the one attribute derived from input, and only http(s) URLs
// become hrefs at all.

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Bold, italics, and http(s) links over already-escaped text. */
function styleInline(escaped) {
  return escaped
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer noopener">$1</a>',
    );
}

/**
 * Inline markup for one already-escaped line: code spans first, because their
 * content must not gain emphasis or links.
 */
function renderInline(escaped) {
  // Splitting on the code-span pattern leaves code between odd indices, plain
  // text between even ones, so each half is treated by its own rules.
  return escaped
    .split(/`([^`\n]*)`/)
    .map((part, index) => (index % 2 ? `<code>${part}</code>` : styleInline(part)))
    .join("");
}

/**
 * Render a reply to HTML using only the tags this module emits.
 *
 * Returns a string rather than nodes so it stays runnable under plain node,
 * where the packaging tests exercise it; the panel instantiates it through a
 * <template>, which parses without executing anything.
 */
export function markdownToHtml(text) {
  const lines = String(text ?? "")
    .replace(/\r\n?/g, "\n")
    .split("\n");
  const html = [];
  let paragraph = [];
  let list = null; // { tag: "ul" | "ol", items: [] }
  let code = null; // lines inside an unclosed fence

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html.push(`<p>${paragraph.map(renderInline).join("<br>")}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (!list) return;
    html.push(
      `<${list.tag}>${list.items.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${list.tag}>`,
    );
    list = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (code) {
      if (trimmed.startsWith("```")) {
        html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = null;
      } else {
        code.push(line);
      }
      continue;
    }
    if (trimmed.startsWith("```")) {
      flushParagraph();
      flushList();
      code = [];
      continue;
    }
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(escapeHtml(heading[2]))}</h${level}>`);
      continue;
    }

    const bullet = /^[-*]\s+(.*)$/.exec(trimmed);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(trimmed);
    if (bullet || numbered) {
      flushParagraph();
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tag !== tag) {
        flushList();
        list = { tag, items: [] };
      }
      list.items.push(escapeHtml((bullet ?? numbered)[1]));
      continue;
    }

    flushList();
    paragraph.push(escapeHtml(trimmed));
  }

  // A stream cut off mid-fence still renders what arrived, as code.
  if (code) html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  flushParagraph();
  flushList();
  return html.join("");
}

/**
 * Parse renderer output into inert nodes for the transcript.
 *
 * The string contains only this module's tags over escaped text, and parsing
 * in a template executes nothing, so attaching the result is as safe as the
 * textContent the panel used before rendering existed.
 */
export function markdownNodes(text) {
  const template = document.createElement("template");
  template.innerHTML = markdownToHtml(text);
  return template.content;
}

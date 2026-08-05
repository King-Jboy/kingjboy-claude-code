// The run_command tool, and the native messaging call behind it.
//
// Manifest V3 cannot spawn a process, so this goes out to the fcc-bridge host
// over Chrome's native messaging channel. Chrome will only connect if the host
// manifest names this exact extension ID in allowed_origins, which is what
// `fcc-extension install --extension-id ID` writes.
//
// sendNativeMessage is one-shot: Chrome starts the host, delivers the message,
// takes the reply, and lets the process exit. A persistent port would be faster
// but would carry state between commands; one process per approved command is
// easier to reason about and leaves nothing running between turns.

const HOST_NAME = "com.free_claude_code.bridge";

export const SHELL_TOOL = {
  name: "run_command",
  description:
    "Run a shell command on the user's machine and return its output. The user is shown " +
    "the exact command and must approve it before it runs, so prefer one clear command " +
    "over a chain of speculative ones, and say what you expect it to show. Commands are " +
    "confined to a configured directory. There is no interactive stdin: a command that " +
    "prompts will hang until it times out, so pass non-interactive flags.",
  input_schema: {
    type: "object",
    properties: {
      command: { type: "string", description: "The command line to run." },
      cwd: {
        type: "string",
        description:
          "Directory to run in, absolute or relative to the configured root. " +
          "Omit to use the root itself.",
      },
      timeout: {
        type: "integer",
        description: "Seconds to wait before giving up. Defaults to 120, capped at 600.",
      },
    },
    required: ["command"],
  },
};

function sendNative(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendNativeMessage(HOST_NAME, message, (response) => {
      // Chrome reports a missing or misregistered host through lastError
      // rather than by throwing, and reading it clears the warning.
      const failure = chrome.runtime.lastError;
      if (failure) {
        reject(new Error(failure.message ?? "The native host could not be reached."));
        return;
      }
      resolve(response);
    });
  });
}

/** Probe the bridge so the panel only advertises run_command when it works. */
export async function bridgeStatus() {
  try {
    const pong = await sendNative({ type: "ping" });
    return {
      available: true,
      enabled: Boolean(pong?.enabled),
      root: pong?.root ?? "",
      shell: pong?.shell ?? "",
    };
  } catch (error) {
    return { available: false, enabled: false, reason: error.message };
  }
}

function formatResult(response) {
  const lines = [`exit ${response.exit_code} in ${response.cwd}`];
  if (response.stdout) lines.push("", "stdout:", response.stdout);
  if (response.stderr) lines.push("", "stderr:", response.stderr);
  if (!response.stdout && !response.stderr) lines.push("", "(no output)");
  if (response.truncated) lines.push("", "[output was truncated]");
  return lines.join("\n");
}

export async function runShellCommand(input, { requestApproval }) {
  const command = typeof input?.command === "string" ? input.command.trim() : "";
  if (!command) return { content: "No command supplied.", is_error: true };

  const cwd = typeof input?.cwd === "string" ? input.cwd.trim() : "";
  if (!(await requestApproval({ command, cwd }))) {
    // Refusal is an ordinary outcome, not a failure of the tool. Reporting it
    // as a result lets the model adapt instead of the turn collapsing.
    return { content: "The user declined to run this command.", is_error: true };
  }

  let response;
  try {
    response = await sendNative({
      type: "run",
      command,
      cwd: cwd || null,
      timeout: Number.isInteger(input?.timeout) ? input.timeout : null,
    });
  } catch (error) {
    return { content: `The command bridge is unreachable: ${error.message}`, is_error: true };
  }

  if (!response?.ok) {
    return { content: response?.error ?? "The bridge returned no result.", is_error: true };
  }
  return { content: formatResult(response), is_error: response.exit_code !== 0 };
}

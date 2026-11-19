"""Command parsing utilities for API optimizations."""

import re
import shlex

_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def _is_env_assignment(part: str) -> bool:
    """Return True when a token is a shell-style env assignment."""
    return bool(_ENV_ASSIGNMENT_RE.match(part))


def _strip_env_assignments(parts: list[str]) -> list[str]:
    """Return command parts after leading shell-style env assignments."""
    cmd_start = 0
    for i, part in enumerate(parts):
        if _is_env_assignment(part):
            cmd_start = i + 1
        else:
            break
    return parts[cmd_start:]


_SHELL_CONTROL_OPERATOR_CHARS = frozenset({";", "|", "&", ">", "<", "(", ")", "\n", "\r"})


def has_unquoted_shell_operator(command: str) -> bool:
    """Return True if ``command`` contains unquoted chaining, piping, or redirection.

    Quote-aware inspection keeps harmless quoted arguments (e.g.
    ``git commit -m "fix A && B"``) on the fast path while ensuring compound
    commands, subshells, newlines, and redirections cannot bypass permission
    checks. Unterminated quotes fail closed by returning True.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()\n\r")
    lexer.whitespace = " \t"
    lexer.whitespace_split = False
    lexer.commenters = ""
    try:
        for token in lexer:
            if token and all(ch in _SHELL_CONTROL_OPERATOR_CHARS for ch in token):
                return True
    except ValueError:
        return True
    return False


def extract_safe_command_prefix(command: str) -> str | None:
    """Extract a command prefix only when ``command`` is a single simple invocation.

    Returns ``"command_injection_detected"`` for backtick/``$(`` substitutions,
    ``None`` when unquoted shell operators, redirections, or newlines require
    upstream model classification, or the parsed prefix for simple commands.
    """
    if "`" in command or "$(" in command:
        return "command_injection_detected"
    if has_unquoted_shell_operator(command):
        return None
    return extract_command_prefix(command)


def extract_command_prefix(command: str) -> str:
    """Extract the command prefix for fast prefix detection.

    Parses a shell command safely, handling environment variables and
    command injection attempts. Returns the command prefix suitable
    for quick identification.

    Returns:
        Command prefix (e.g., "git", "git commit", "npm install")
        or "none" if no valid command found
    """
    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command, posix=False)
        if not parts:
            return "none"

        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if _is_env_assignment(part):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {
            "git",
            "npm",
            "docker",
            "kubectl",
            "cargo",
            "go",
            "pip",
            "yarn",
        }

        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            prefix = (
                f"{first_word} {second_word}"
                if not second_word.startswith("-")
                else first_word
            )
        else:
            prefix = first_word
        return prefix if not env_prefix else " ".join(env_prefix) + " " + prefix

    except ValueError:
        parts = command.split()
        if not parts:
            return "none"
        cmd_parts = _strip_env_assignments(parts)
        return cmd_parts[0] if cmd_parts else "none"


def extract_filepaths_from_command(command: str, output: str) -> str:
    """Extract file paths from a command locally without API call.

    Determines if the command reads file contents and extracts paths accordingly.
    Commands like ls/dir/find just list files, so return empty.
    Commands like cat/head/tail actually read contents, so extract the file path.

    Returns:
        Filepath extraction result in <filepaths> format
    """
    listing_commands = {
        "ls",
        "dir",
        "find",
        "tree",
        "pwd",
        "cd",
        "mkdir",
        "rmdir",
        "rm",
    }

    reading_commands = {"cat", "head", "tail", "less", "more", "bat", "type"}

    try:
        parts = shlex.split(command, posix=False)
        if not parts:
            return "<filepaths>\n</filepaths>"

        cmd_parts = _strip_env_assignments(parts)
        if not cmd_parts:
            return "<filepaths>\n</filepaths>"

        base_cmd = cmd_parts[0].split("/")[-1].split("\\")[-1].lower()

        if base_cmd in listing_commands:
            return "<filepaths>\n</filepaths>"

        if base_cmd in reading_commands:
            filepaths = []
            for part in cmd_parts[1:]:
                if part.startswith("-"):
                    continue
                filepaths.append(part)

            if filepaths:
                paths_str = "\n".join(filepaths)
                return f"<filepaths>\n{paths_str}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        if base_cmd == "grep":
            flags_with_args = {"-e", "-f", "-m", "-A", "-B", "-C"}
            pattern_provided_via_flag = False
            positional = []

            skip_next = False
            for part in cmd_parts[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if part.startswith("-"):
                    if part in flags_with_args:
                        if part in {"-e", "-f"}:
                            pattern_provided_via_flag = True
                        skip_next = True
                    continue
                positional.append(part)

            filepaths = positional if pattern_provided_via_flag else positional[1:]
            if filepaths:
                paths_str = "\n".join(filepaths)
                return f"<filepaths>\n{paths_str}\n</filepaths>"
            return "<filepaths>\n</filepaths>"

        return "<filepaths>\n</filepaths>"

    except ValueError:
        return "<filepaths>\n</filepaths>"

"""Request detection utilities for API optimizations.

Detects quota checks, title generation, prefix detection, safety classifier,
suggestion mode, and filepath extraction requests to enable targeted handling.
"""

from free_claude_code.core.anthropic import (
    Message,
    MessagesRequest,
    extract_text_from_content,
)


class _RequestInspectionView:
    """Lazy, single-pass text extraction view for one MessagesRequest."""

    __slots__ = (
        "_extractor",
        "_request",
        "_sole_user_resolved",
        "_sole_user_text",
        "_sole_user_turn",
        "_system_text",
        "_system_text_lower",
        "_top_system_text",
    )

    def __init__(self, request_data: MessagesRequest, extractor: object) -> None:
        self._request = request_data
        self._extractor = extractor
        self._sole_user_resolved = False
        self._sole_user_turn: Message | None = None
        self._sole_user_text: str | None = None
        self._system_text: str | None = None
        self._system_text_lower: str | None = None
        self._top_system_text: str | None = None

    @property
    def sole_user_turn(self) -> Message | None:
        if not self._sole_user_resolved:
            user_turn: Message | None = None
            for message in self._request.messages:
                if message.role == "system":
                    continue
                if message.role != "user" or user_turn is not None:
                    user_turn = None
                    break
                user_turn = message
            self._sole_user_turn = user_turn
            self._sole_user_resolved = True
        return self._sole_user_turn

    @property
    def sole_user_text(self) -> str | None:
        turn = self.sole_user_turn
        if turn is None:
            return None
        if self._sole_user_text is None:
            self._sole_user_text = extract_text_from_content(turn.content)
        return self._sole_user_text

    @property
    def top_system_text(self) -> str:
        if self._top_system_text is None:
            system = getattr(self._request, "system", None)
            self._top_system_text = extract_text_from_content(system) if system else ""
        return self._top_system_text

    @property
    def system_text(self) -> str:
        if self._system_text is None:
            parts: list[str] = []
            top = self.top_system_text
            if top:
                parts.append(top)
            for message in self._request.messages:
                if message.role != "system":
                    continue
                text = extract_text_from_content(message.content)
                if text:
                    parts.append(text)
            self._system_text = "\n".join(parts)
        return self._system_text

    @property
    def system_text_lower(self) -> str:
        if self._system_text_lower is None:
            self._system_text_lower = self.system_text.lower()
        return self._system_text_lower


_last_inspection: (
    tuple[
        MessagesRequest,
        object,
        int,
        int,
        bool,
        _RequestInspectionView,
    ]
    | None
) = None


def _inspect_request(request_data: MessagesRequest) -> _RequestInspectionView:
    """Return a cached inspection view when inspecting the same request in sequence."""
    global _last_inspection
    extractor = extract_text_from_content
    messages = getattr(request_data, "messages", ())
    msg_len = len(messages) if messages is not None else 0
    sys_id = id(getattr(request_data, "system", None))
    has_tools = bool(getattr(request_data, "tools", None))
    if (
        _last_inspection is not None
        and _last_inspection[0] is request_data
        and _last_inspection[1] is extractor
        and _last_inspection[2] == msg_len
        and _last_inspection[3] == sys_id
        and _last_inspection[4] == has_tools
    ):
        return _last_inspection[5]
    view = _RequestInspectionView(request_data, extractor)
    _last_inspection = (request_data, extractor, msg_len, sys_id, has_tools, view)
    return view


def _single_user_turn(request_data: MessagesRequest) -> Message | None:
    """Return the sole conversational user turn, ignoring system context."""
    return _inspect_request(request_data).sole_user_turn


def _request_system_text(request_data: MessagesRequest) -> str:
    """Return top-level and inline system text for request-shape detection."""
    return _inspect_request(request_data).system_text


def is_quota_check_request(request_data: MessagesRequest) -> bool:
    """Check if this is a quota probe request.

    Quota checks are typically simple requests with max_tokens=1
    and a single message containing the word "quota".
    """
    if getattr(request_data, "max_tokens", None) != 1:
        return False
    text = _inspect_request(request_data).sole_user_text
    if text is None:
        return False
    return "quota" in text.lower()


def is_title_generation_request(request_data: MessagesRequest) -> bool:
    """Check if this is a conversation title generation request.

    Title generation requests are detected by a system prompt containing
    title extraction instructions, no tools, and a single user message.

    Matches Claude Code session title prompts (sentence-case title, JSON
    \"title\" field, etc.).
    """
    if getattr(request_data, "tools", None):
        return False
    system_text = _inspect_request(request_data).system_text_lower
    if "title" not in system_text:
        return False
    return "sentence-case title" in system_text or (
        "return json" in system_text
        and "field" in system_text
        and ("coding session" in system_text or "this session" in system_text)
    )


def is_prefix_detection_request(request_data: MessagesRequest) -> tuple[bool, str]:
    """Check if this is a fast prefix detection request.

    Prefix detection requests contain a policy_spec block and
    a Command: section for extracting shell command prefixes.

    Returns:
        Tuple of (is_prefix_request, command_string)
    """
    content = _inspect_request(request_data).sole_user_text
    if content is None:
        return False, ""

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except TypeError:
            return False, ""

    return False, ""


def is_safety_classifier_request(request_data: MessagesRequest) -> bool:
    """Return whether this is Claude Code's auto-mode safety classifier prompt."""
    if getattr(request_data, "tools", None):
        return False

    view = _inspect_request(request_data)
    system_text = view.top_system_text
    if view.sole_user_turn is not None and len(request_data.messages) == 1:
        messages_text = view.sole_user_text or ""
    else:
        messages_text = "".join(
            extract_text_from_content(message.content)
            for message in request_data.messages
        )
    combined = f"{system_text}\n{messages_text}"
    has_verdict_instruction = "yes</block>" in combined or "no</block>" in combined
    return "<transcript>" in combined and has_verdict_instruction


def is_suggestion_mode_request(request_data: MessagesRequest) -> bool:
    """Check if this is a suggestion mode request.

    Suggestion mode requests contain "[SUGGESTION MODE:" in the user's message,
    used for auto-suggesting what the user might type next.
    """
    view = _inspect_request(request_data)
    if view.sole_user_turn is not None:
        sole_text = view.sole_user_text
        return sole_text is not None and "[SUGGESTION MODE:" in sole_text
    for msg in request_data.messages:
        if msg.role == "user":
            text = extract_text_from_content(msg.content)
            if "[SUGGESTION MODE:" in text:
                return True
    return False


def is_filepath_extraction_request(
    request_data: MessagesRequest,
) -> tuple[bool, str, str]:
    """Check if this is a filepath extraction request.

    Filepath extraction requests have a single user message with
    "Command:" and "Output:" sections, asking to extract file paths
    from command output.

    Returns:
        Tuple of (is_filepath_request, command, output)
    """
    if getattr(request_data, "tools", None):
        return False, "", ""
    view = _inspect_request(request_data)
    content = view.sole_user_text
    if content is None:
        return False, "", ""

    if "Command:" not in content or "Output:" not in content:
        return False, "", ""

    # Match if user content OR system block indicates filepath extraction
    content_lower = content.lower()
    user_has_filepaths = "filepaths" in content_lower or "<filepaths>" in content_lower
    system_text_lower = view.system_text_lower
    system_has_extract = (
        "extract any file paths" in system_text_lower
        or "file paths that this command" in system_text_lower
    )
    if not user_has_filepaths and not system_has_extract:
        return False, "", ""

    cmd_start = content.find("Command:") + len("Command:")
    output_marker = content.find("Output:", cmd_start)
    if output_marker == -1:
        return False, "", ""

    command = content[cmd_start:output_marker].strip()
    output = content[output_marker + len("Output:") :].strip()

    for marker in ["<", "\n\n"]:
        if marker in output:
            output = output.split(marker)[0].strip()

    return True, command, output

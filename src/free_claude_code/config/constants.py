"""Shared defaults used by config models and provider adapters."""

# HTTP client connect timeout (seconds). Keep aligned with README.md and .env.example.
HTTP_CONNECT_TIMEOUT_DEFAULT = 10.0

# Anthropic Messages API default when the client omits max_tokens.
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 81920

# Context window advertised to launched client CLIs for a model with no measured
# entry in ~/.fcc/context.md. 256k is the common ceiling across current
# large-context models, so it is the least-wrong guess when nothing is known.
#
# It is a guess, and an optimistic one: a route pointing at a 128k model will
# overrun it and fail upstream. Run `fcc-context` so the routed model resolves
# from a real measurement instead of falling back here.
DEFAULT_CLIENT_CONTEXT_WINDOW = 262144

"""Provider-prefixed model reference helpers."""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

PINNED_MODELS_EXAMPLE = (
    '["nvidia_nim/deepseek-ai/deepseek-v4-flash", "open_router/z-ai/glm-5.2:free"]'
)


class ModelCatalogScope(StrEnum):
    """How much of the provider catalog client and admin model lists expose."""

    # Every discovered provider model, plus the configured routes.
    ALL = "all"
    # Only the configured routes and the pinned list, so a picker lists what
    # you chose rather than several hundred ids of unknown health.
    CONFIGURED = "configured"


def parse_model_ref_list(raw: str, *, env_name: str) -> tuple[str, ...]:
    """Parse a JSON array of ``provider/model`` refs, ordered and de-duplicated.

    Malformed input raises rather than degrading to an empty list: a pinned
    list that silently vanishes looks like the setting never worked.
    """
    text = raw.strip()
    if not text:
        return ()
    try:
        decoded = json.loads(text)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} is not valid JSON: expected a list of provider/model "
            f"strings, for example {PINNED_MODELS_EXAMPLE}"
        ) from exc
    if not isinstance(decoded, list):
        raise ValueError(
            f"{env_name} must be a JSON list of provider/model strings, for "
            f"example {PINNED_MODELS_EXAMPLE}"
        )

    refs: list[str] = []
    seen: set[str] = set()
    for entry in decoded:
        if not isinstance(entry, str):
            raise ValueError(
                f"{env_name} must contain only provider/model strings, for "
                f"example {PINNED_MODELS_EXAMPLE}"
            )
        ref = entry.strip()
        if not ref or ref in seen:
            continue
        if ref.count("/") < 1 or not all(part for part in ref.split("/")):
            raise ValueError(
                f"{env_name} entry {entry!r} is not a provider/model ref, for "
                f"example {PINNED_MODELS_EXAMPLE}"
            )
        seen.add(ref)
        refs.append(ref)
    return tuple(refs)


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference."""

    model_ref: str
    provider_id: str
    model_id: str


class ChatModelConfig(Protocol):
    model: str
    model_fable: str | None
    model_opus: str | None
    model_sonnet: str | None
    model_haiku: str | None
    pinned_models: str
    model_fallbacks: str


def pinned_model_refs(settings: ChatModelConfig) -> tuple[str, ...]:
    """Return the user's pinned provider/model refs."""

    return parse_model_ref_list(settings.pinned_models, env_name="PINNED_MODELS")


def model_fallback_refs(settings: ChatModelConfig) -> tuple[str, ...]:
    """Return the ordered provider/model refs to try when capacity runs out."""

    return parse_model_ref_list(settings.model_fallbacks, env_name="MODEL_FALLBACKS")


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return model_ref.split("/", 1)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return model_ref.split("/", 1)[1]


def configured_chat_model_refs(
    settings: ChatModelConfig,
) -> tuple[ConfiguredChatModelRef, ...]:
    """Return unique configured chat provider/model refs."""

    model_refs = dict.fromkeys(
        model_ref
        for model_ref in (
            settings.model,
            settings.model_fable,
            settings.model_opus,
            settings.model_sonnet,
            settings.model_haiku,
        )
        if model_ref is not None
    )

    return tuple(
        ConfiguredChatModelRef(
            model_ref=model_ref,
            provider_id=parse_provider_type(model_ref),
            model_id=parse_model_name(model_ref),
        )
        for model_ref in model_refs
    )

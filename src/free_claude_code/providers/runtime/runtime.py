"""One closable generation of lazily constructed provider clients."""

import asyncio
from collections.abc import Callable, MutableMapping

from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.key_pool import ApiKeyPool, KeyStateSnapshot

from .factory import create_provider

ProviderConstructor = Callable[[str, Settings], BaseProvider]


class ProviderRuntime:
    """Own provider instances for one immutable settings snapshot."""

    def __init__(
        self,
        settings: Settings,
        providers: MutableMapping[str, BaseProvider] | None = None,
        *,
        provider_constructor: ProviderConstructor = create_provider,
    ) -> None:
        self.settings = settings
        self._providers = providers if providers is not None else {}
        self._provider_constructor = provider_constructor
        self._inherited_key_state: dict[str, dict[str, KeyStateSnapshot]] = {}

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already cached."""
        return provider_id in self._providers

    def inherit_credential_state(self, previous: ProviderRuntime) -> None:
        """Snapshot a predecessor's key-pool state for providers built later.

        Providers are built lazily, so the snapshot is taken now, while the
        previous generation still holds its providers, and applied on first use.
        """
        # getattr: runtime subclasses (test doubles among them) may not run
        # this class's __init__.
        previous_providers = getattr(previous, "_providers", None) or {}
        inherited_state = getattr(self, "_inherited_key_state", None)
        if inherited_state is None:
            return
        for provider_id, provider in previous_providers.items():
            pool = getattr(provider, "api_key_pool", None)
            if isinstance(pool, ApiKeyPool):
                inherited_state[provider_id] = pool.export_state()

    def resolve_provider(self, provider_id: str) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        if provider_id not in self._providers:
            provider = self._provider_constructor(provider_id, self.settings)
            inherited = self._inherited_key_state.pop(provider_id, None)
            pool = getattr(provider, "api_key_pool", None)
            if inherited is not None and isinstance(pool, ApiKeyPool):
                pool.import_state(inherited)
            self._providers[provider_id] = provider
        return self._providers[provider_id]

    async def cleanup(self) -> None:
        """Release every provider client constructed by this generation."""
        errors: list[Exception] = []
        for provider_id, provider in list(self._providers.items()):
            try:
                await provider.cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(exc)
            else:
                self._providers.pop(provider_id, None)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("One or more provider cleanups failed", errors)

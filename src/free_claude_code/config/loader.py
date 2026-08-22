"""Settings loading entry point shared by installed client launchers.

The upstream loader grew provenance tracking around a managed-config stack
this fork does not carry; launchers here only need the cached, validated
Settings the server itself uses.
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]

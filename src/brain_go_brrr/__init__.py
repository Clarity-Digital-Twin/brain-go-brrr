"""Brain Go Brrr: A digital twin brain-computer interface project.

This package provides tools for EEG signal processing and neural representation
learning based on the EEGPT transformer architecture.
"""

# Single source of truth for version from pyproject.toml
try:
    from importlib.metadata import version

    __version__ = version("brain-go-brrr")
except Exception:
    __version__ = "1.0.0"  # Fallback for development

__author__ = "CLARITY-DIGITAL-TWIN"
__email__ = "contact@clarity-digital-twin.org"

from typing import Any

# Lazy import Config to avoid circular dependencies at module import time
__all__ = ["Config"]


def __getattr__(name: str) -> Any:
    if name == "Config":
        from .application.config.base import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

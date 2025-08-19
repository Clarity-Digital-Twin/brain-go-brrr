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

# Package imports
from .application.config.base import Config

# Don't import logger here to avoid circular imports
# Users should import directly: from brain_go_brrr.infra.logger import get_logger

__all__ = ["Config"]

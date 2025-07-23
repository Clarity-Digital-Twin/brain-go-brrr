"""Brain Go Brrr: A digital twin brain-computer interface project.

This package provides tools for EEG signal processing and neural representation
learning based on the EEGPT transformer architecture.
"""

__version__ = "0.1.0"
__author__ = "CLARITY-DIGITAL-TWIN"
__email__ = "contact@clarity-digital-twin.org"

# Standard library imports
import importlib
import sys

# Import shim for backwards compatibility with tests
sys.modules.setdefault("core", importlib.import_module("brain_go_brrr.core"))

# Package imports
from .core.config import Config  # noqa: E402
from .core.logger import get_logger  # noqa: E402

__all__ = ["Config", "get_logger"]

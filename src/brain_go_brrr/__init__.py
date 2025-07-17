"""
Brain Go Brrr: A digital twin brain-computer interface project.

This package provides tools for EEG signal processing and neural representation 
learning based on the EEGPT transformer architecture.
"""

__version__ = "0.1.0"
__author__ = "CLARITY-DIGITAL-TWIN"
__email__ = "contact@clarity-digital-twin.org"

from brain_go_brrr.core.config import Config
from brain_go_brrr.core.logger import get_logger

__all__ = ["Config", "get_logger"]
"""Compatibility module for tests that import from eegpt_model.

This module exists solely to maintain backward compatibility with tests
that were written before the refactoring. It re-exports the EEGPTModel
from eegpt_compat.py.
"""

from .eegpt_compat import EEGPTConfig, EEGPTModel

__all__ = ["EEGPTConfig", "EEGPTModel"]

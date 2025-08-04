"""EEG preprocessing module for Brain-Go-Brrr.

This module provides preprocessing pipelines for various EEG analysis tasks.
"""

from .autoreject_adapter import SyntheticPositionGenerator, WindowEpochAdapter
from .chunked_autoreject import ChunkedAutoRejectProcessor
from .eeg_preprocessor import EEGPreprocessor

__all__ = [
    "ChunkedAutoRejectProcessor",
    "EEGPreprocessor",
    "SyntheticPositionGenerator",
    "WindowEpochAdapter",
]

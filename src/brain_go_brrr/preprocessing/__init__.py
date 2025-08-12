"""EEG preprocessing module for Brain-Go-Brrr.

This module provides preprocessing pipelines for various EEG analysis tasks.
"""

from .autoreject_adapter import SyntheticPositionGenerator, WindowEpochAdapter
from .basic import (
    BandpassFilter,
    Normalizer,
    NotchFilter,
    PreprocessingConfig,
    PreprocessingPipeline,
    Resampler,
)
from .chunked_autoreject import ChunkedAutoRejectProcessor
from .eeg_preprocessor import EEGPreprocessor

__all__ = [
    "BandpassFilter",
    # Advanced preprocessing
    "ChunkedAutoRejectProcessor",
    "EEGPreprocessor",
    "Normalizer",
    "NotchFilter",
    # Basic preprocessing
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "Resampler",
    "SyntheticPositionGenerator",
    "WindowEpochAdapter",
]

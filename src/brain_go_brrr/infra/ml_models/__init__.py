"""Brain-Go-Brrr Models Module.

Contains model implementations for EEG analysis.
"""

from .eegpt_model import EEGPTConfig, EEGPTModel, extract_features_from_raw, preprocess_for_eegpt
from .linear_probe import LinearProbeHead, TwoLayerProbe

__all__ = [
    "EEGPTConfig",
    "EEGPTModel",
    "LinearProbeHead",
    "TwoLayerProbe",
    "extract_features_from_raw",
    "preprocess_for_eegpt",
]

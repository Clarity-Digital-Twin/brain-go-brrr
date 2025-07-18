"""Brain-Go-Brrr Models Module.

Contains model implementations for EEG analysis.
"""

from .eegpt_model import EEGPTConfig, EEGPTModel, extract_features_from_raw, preprocess_for_eegpt

__all__ = ["EEGPTConfig", "EEGPTModel", "extract_features_from_raw", "preprocess_for_eegpt"]

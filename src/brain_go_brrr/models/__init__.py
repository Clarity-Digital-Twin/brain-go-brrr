"""Brain-Go-Brrr Models Module

Contains model implementations for EEG analysis.
"""

from .eegpt_model import (
    EEGPTModel,
    EEGPTConfig,
    preprocess_for_eegpt,
    extract_features_from_raw
)

__all__ = [
    'EEGPTModel',
    'EEGPTConfig', 
    'preprocess_for_eegpt',
    'extract_features_from_raw'
]

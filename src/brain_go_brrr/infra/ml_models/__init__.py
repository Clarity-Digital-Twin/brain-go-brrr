"""Brain-Go-Brrr Models Module.

Contains model implementations for EEG analysis.
"""

from .eegpt_compat import EEGPTConfig, EEGPTModel, extract_features_from_raw, preprocess_for_eegpt
from .linear_probe import LinearProbeHead, TwoLayerProbe
from .probe_factory import ProbeFactory, UnifiedProbe, migrate_eegpt_probe_to_factory

__all__ = [
    "EEGPTConfig",
    "EEGPTModel",
    "LinearProbeHead",
    "ProbeFactory",
    "TwoLayerProbe",
    "UnifiedProbe",
    "extract_features_from_raw",
    "migrate_eegpt_probe_to_factory",
    "preprocess_for_eegpt",
]

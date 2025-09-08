"""Brain-Go-Brrr Models Module.

Contains model implementations for EEG analysis.

NOTE: eegpt_compat module is deprecated and not re-exported.
Import directly from brain_go_brrr.infra.ml_models.eegpt_compat if needed.
Prefer using eegpt_wrapper.create_normalized_eegpt() instead.
"""

from .linear_probe import LinearProbeHead, TwoLayerProbe
from .probe_factory import ProbeFactory, UnifiedProbe, migrate_eegpt_probe_to_factory

__all__ = [
    "LinearProbeHead",
    "ProbeFactory",
    "TwoLayerProbe",
    "UnifiedProbe",
    "migrate_eegpt_probe_to_factory",
]

#!/usr/bin/env python
"""Domain preprocessing re-exports from core utilities.

This maintains backward compatibility while keeping proper architecture.
Core utilities are leaf nodes - domain can import from core.
"""

# Re-export from core (domain can depend on core utilities)
from brain_go_brrr.domain.preprocessing.core_logic import (
    BandpassFilter,
    Normalizer,
    NotchFilter,
    PreprocessingConfig,
    PreprocessingPipeline,
    Resampler,
)

__all__ = [
    "BandpassFilter",
    "Normalizer",
    "NotchFilter",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "Resampler",
]

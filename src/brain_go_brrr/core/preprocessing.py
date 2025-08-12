#!/usr/bin/env python
"""DEPRECATED: Use brain_go_brrr.preprocessing.basic instead.

This module is kept for backwards compatibility only.
All functionality has been moved to brain_go_brrr.preprocessing.basic
"""

import warnings

warnings.warn(
    "brain_go_brrr.core.preprocessing is deprecated. "
    "Use brain_go_brrr.preprocessing.basic instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from brain_go_brrr.preprocessing.basic import (
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


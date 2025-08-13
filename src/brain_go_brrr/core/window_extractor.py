"""Compatibility shim for moved window_extractor module.

DEPRECATED: Use brain_go_brrr.preprocessing.window_extractor instead.
This shim will be removed in a future release.
"""

import warnings

warnings.warn(
    "brain_go_brrr.core.window_extractor has moved to brain_go_brrr.preprocessing.window_extractor",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from brain_go_brrr.preprocessing.window_extractor import *  # noqa: F403, E402

try:
    from brain_go_brrr.preprocessing.window_extractor import (
        __all__,  # type: ignore
    )
except ImportError:
    __all__ = []

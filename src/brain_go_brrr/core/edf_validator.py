"""Compatibility shim for moved edf_validator module.

DEPRECATED: Use brain_go_brrr.data.edf_validator instead.
This shim will be removed in a future release.
"""

import warnings

warnings.warn(
    "brain_go_brrr.core.edf_validator has moved to brain_go_brrr.data.edf_validator",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from brain_go_brrr.data.edf_validator import *  # noqa: F403, E402

try:
    from brain_go_brrr.data.edf_validator import __all__  # type: ignore
except ImportError:
    __all__ = []

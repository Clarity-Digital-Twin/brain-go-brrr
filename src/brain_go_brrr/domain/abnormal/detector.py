"""Deprecated: use domain.abnormal.detector_pure."""

from warnings import warn

from .detector_pure import *  # noqa: F403, F401  # Re-export clean implementation

warn(
    "brain_go_brrr.domain.abnormal.detector is deprecated; use detector_pure",
    DeprecationWarning,
    stacklevel=2,
)

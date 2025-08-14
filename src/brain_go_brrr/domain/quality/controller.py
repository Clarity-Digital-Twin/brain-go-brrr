"""Deprecated: use domain.quality.controller_clean."""

from warnings import warn

from .controller_clean import *  # noqa: F403, F401  # Re-export clean implementation

warn(
    "brain_go_brrr.domain.quality.controller is deprecated; use controller_clean",
    DeprecationWarning,
    stacklevel=2,
)

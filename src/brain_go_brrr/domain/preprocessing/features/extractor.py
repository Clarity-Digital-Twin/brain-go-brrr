"""Deprecated: use domain.preprocessing.features.extractor_clean."""

from warnings import warn

from .extractor_clean import *  # noqa: F403  # Re-export clean implementation

warn(
    "brain_go_brrr.domain.preprocessing.features.extractor is deprecated; use extractor_clean",
    DeprecationWarning,
    stacklevel=2,
)

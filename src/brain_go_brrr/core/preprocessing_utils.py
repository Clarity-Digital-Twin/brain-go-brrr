"""Legacy import shim for preprocessing utilities.

DEPRECATED: This module will be removed in v2.0.0 (Q2 2025).
Import directly from brain_go_brrr.domain.preprocessing.core_logic instead.
"""

import warnings

warnings.warn(
    "Importing from brain_go_brrr.core.preprocessing_utils is deprecated. "
    "Import from brain_go_brrr.domain.preprocessing.core_logic instead. "
    "This shim will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# TODO: Remove this shim in v2.0.0 (Q2 2025)
from brain_go_brrr.domain.preprocessing.core_logic import *  # noqa: F401,F403

"""DEPRECATED: Use tuab_dataset.TUABDataset instead.

This module is kept for backward compatibility only.
"""

import warnings

from .tuab_dataset import TUABDataset

warnings.warn(
    "tuab_cached_dataset is deprecated. Use brain_go_brrr.infra.data.tuab_dataset instead",
    DeprecationWarning,
    stacklevel=2
)

# Alias for backward compatibility
TUABCachedDataset = TUABDataset

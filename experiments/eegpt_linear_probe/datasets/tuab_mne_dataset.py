"""
TUAB dataset shim - imports from src.
DEPRECATED: Use brain_go_brrr.infra.data.tuab_dataset directly.
"""

import warnings

from brain_go_brrr.infra.data.tuab_dataset import TUABDataset

warnings.warn(
    "experiments/eegpt_linear_probe/datasets/tuab_mne_dataset.py is deprecated. "
    "Import directly from brain_go_brrr.infra.data.tuab_dataset",
    DeprecationWarning,
    stacklevel=2,
)

# Alias for compatibility
TUABMNEDataset = TUABDataset

__all__ = ["TUABMNEDataset"]

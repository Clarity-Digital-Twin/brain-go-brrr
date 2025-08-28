"""
TUEV dataset shim - imports from src.
DEPRECATED: Use brain_go_brrr.infra.data.tuev_dataset directly.
"""

import warnings
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset, CLASS_MAPPING

warnings.warn(
    "experiments/eegpt_linear_probe/datasets/tuev_mne_dataset.py is deprecated. "
    "Import directly from brain_go_brrr.infra.data.tuev_dataset",
    DeprecationWarning,
    stacklevel=2
)

# Export for compatibility
__all__ = ["TUEVMNEDataset", "CLASS_MAPPING"]
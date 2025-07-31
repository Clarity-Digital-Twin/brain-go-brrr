#!/usr/bin/env python
"""Check validation dataset distribution."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402

print("Checking validation dataset distribution...")
ds = TUABDataset(
    root_dir=Path(os.environ["BGB_DATA_ROOT"]) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="eval",
    window_duration=8.0,
    window_stride=8.0,
)

# Sample first 1000 to check quickly
sample_size = min(1000, len(ds))
labels = [ds[i][1] for i in range(sample_size)]
print(f"Val sample size: {sample_size}, normals: {labels.count(0)}, abnormals: {labels.count(1)}")
print(
    f"Ratio: {labels.count(0) / len(labels):.2%} normal, {labels.count(1) / len(labels):.2%} abnormal"
)

# Check full dataset stats
print(f"\nFull validation set: {len(ds.samples)} windows")
print(f"Class distribution: {ds.class_counts}")

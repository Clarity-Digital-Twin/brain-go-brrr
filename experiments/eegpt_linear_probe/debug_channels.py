#!/usr/bin/env python
"""Debug script to check channel counts throughout the pipeline."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402

# Create dataset
dataset = TUABDataset(
    root_dir=Path(os.environ["BGB_DATA_ROOT"]) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    sampling_rate=256,
    window_duration=8.0,
    window_stride=8.0,
    normalize=True,
)

print(f"Dataset STANDARD_CHANNELS: {len(TUABDataset.STANDARD_CHANNELS)} channels")
print(f"Channels: {TUABDataset.STANDARD_CHANNELS}")

# Get one sample
if len(dataset) > 0:
    data, label = dataset[0]
    print(f"\nSample shape: {data.shape}")
    print("Expected: [20, 2048]")
    print(f"Actual channels: {data.shape[0]}")

#!/usr/bin/env python3
"""Test window sizes from cached dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
import os

# Create dataset
dataset = TUABCachedDataset(
    root_dir=Path(os.environ.get("BGB_DATA_ROOT", "data")) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=5.12,
    window_stride=2.56,
    sampling_rate=200,
    cache_index_path=Path("tuab_index.json"),
    max_files=20  # Small test
)

print(f"Expected window samples: {dataset.window_samples}")
print(f"Window duration: {dataset.window_duration}s @ {dataset.sampling_rate}Hz")

# Test several samples
sizes = set()
for i in range(min(100, len(dataset))):
    x, y = dataset[i]
    sizes.add(x.shape[1])
    if len(sizes) > 1:
        print(f"INCONSISTENT! Sample {i}: shape={x.shape}")

print(f"\nUnique window sizes found: {sizes}")
if len(sizes) == 1:
    print("✅ All windows have consistent size!")
else:
    print("❌ WINDOW SIZE MISMATCH DETECTED!")
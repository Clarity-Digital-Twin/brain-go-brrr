#!/usr/bin/env python3
"""Quick test of fixed dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from torch.utils.data import DataLoader
import os

# Test dataset
print("Creating cached dataset...")
dataset = TUABCachedDataset(
    root_dir=Path(os.environ.get("BGB_DATA_ROOT", "data")) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=5.12,
    window_stride=5.12,
    sampling_rate=200,
    cache_index_path=Path("tuab_index.json"),
    max_files=20  # Small test
)

print(f"Dataset size: {len(dataset)}")
print(f"Window samples: {dataset.window_samples}")

# Test a few samples
print("\nTesting samples...")
for i in range(5):
    x, y = dataset[i]
    print(f"Sample {i}: shape={x.shape}, label={y}")

# Test DataLoader
print("\nTesting DataLoader...")
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

for i, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {i}: shape={batch_x.shape}, labels={batch_y}")
    if i >= 2:
        break

print("\n✅ All tests passed!")
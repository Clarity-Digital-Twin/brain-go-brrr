#!/usr/bin/env python3
"""Debug dataloader to find window size issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
os.environ["BGB_DATA_ROOT"] = str(Path(__file__).parent.parent.parent / "data")

from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from torch.utils.data import DataLoader
from experiments.eegpt_linear_probe.train_enhanced import create_weighted_sampler

# Create dataset exactly as in training
dataset = TUABCachedDataset(
    root_dir=Path("data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"),
    split="train",
    window_duration=5.12,
    window_stride=2.56,
    sampling_rate=200,
    preload=False,
    normalize=True,
    cache_dir=Path("data/cache/tuab_enhanced"),
    cache_index_path=Path("data/cache/tuab_index.json"),
)

print(f"Dataset window_samples: {dataset.window_samples}")
print(f"Dataset size: {len(dataset)}")

# Create sampler
sampler = create_weighted_sampler(dataset)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=0,
    drop_last=True,
)

# Check first few batches
print("\nChecking batches...")
for i, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {i}: shape={batch_x.shape}")
    if i >= 5:
        break
    
    # Check individual items in batch
    for j in range(batch_x.shape[0]):
        item_shape = batch_x[j].shape
        if item_shape[1] != 1024:
            print(f"  ❌ Item {j} has wrong size: {item_shape}")
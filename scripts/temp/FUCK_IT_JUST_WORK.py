#!/usr/bin/env python
"""JUST FUCKING WORK WITH THE CACHE"""

import os
import sys
from pathlib import Path

# Setup
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

print("Loading TUABCachedDataset...")
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

# Paths
data_root = Path(os.environ["BGB_DATA_ROOT"])
cache_index = data_root / "cache/tuab_index.json"
cache_dir = data_root / "cache/tuab_enhanced"

print(f"\nCache index: {cache_index}")
print(f"Cache dir: {cache_dir}")
print(f"Index exists: {cache_index.exists()}")

# Create dataset
print("\nCreating cached dataset...")
dataset = TUABCachedDataset(
    root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    cache_dir=cache_dir,
    cache_index_path=cache_index,
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
    normalize=True,
)

print(f"\nDataset created with {len(dataset)} windows")
print("\nLoading first sample...")
x, y = dataset[0]
print(f"Sample shape: {x.shape}, label: {y}")
print("\n✅ CACHE WORKS!")
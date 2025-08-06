#!/usr/bin/env python
"""Convert 157GB cache to memory-mapped arrays for FAST training."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

print("="*60)
print("CONVERTING CACHE TO MEMORY-MAPPED ARRAYS")
print("This will take ~30 minutes but then training will be FAST")
print("="*60)

data_root = os.environ.get('BGB_DATA_ROOT', '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data')
cache_dir = Path(data_root) / "cache" / "tuab_4s_final"

# Load index
index_path = cache_dir / "index.json"
with open(index_path) as f:
    index = json.load(f)

# Process each split
for split in ['train', 'eval']:
    print(f"\nProcessing {split} split...")
    
    # Count windows first
    total_windows = 0
    for file_path, info in index["files"].items():
        if split in file_path:
            total_windows += info["n_windows"]
    
    print(f"Total {split} windows: {total_windows}")
    
    # Create memory-mapped arrays
    X_path = cache_dir / f"{split}_data.npy"
    y_path = cache_dir / f"{split}_labels.npy"
    
    # Pre-allocate arrays
    X_shape = (total_windows, 20, 1024)
    y_shape = (total_windows,)
    
    print(f"Creating memory-mapped arrays...")
    print(f"  X: {X_path} - shape {X_shape} - {np.prod(X_shape) * 4 / 1e9:.1f} GB")
    print(f"  y: {y_path} - shape {y_shape}")
    
    # Create arrays
    X_mmap = np.memmap(X_path, dtype='float32', mode='w+', shape=X_shape)
    y_mmap = np.memmap(y_path, dtype='int64', mode='w+', shape=y_shape)
    
    # Fill arrays
    idx = 0
    for file_path, info in tqdm(index["files"].items(), desc=f"Loading {split}"):
        if split in file_path:
            cache_file = cache_dir / info["cache_file"]
            data = torch.load(cache_file, weights_only=True)
            
            n_windows = data["x"].shape[0]
            X_mmap[idx:idx+n_windows] = data["x"].numpy()
            y_mmap[idx:idx+n_windows] = data["y"].numpy()
            idx += n_windows
    
    # Flush to disk
    del X_mmap
    del y_mmap
    
    print(f"✓ {split} arrays saved!")

print("\n" + "="*60)
print("CONVERSION COMPLETE!")
print("Arrays ready for fast memory-mapped training")
print("="*60)
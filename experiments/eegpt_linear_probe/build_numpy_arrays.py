#!/usr/bin/env python
"""Build numpy arrays for ultra-fast training."""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Set data root
data_root = os.environ.get('BGB_DATA_ROOT', '/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data')
cache_dir = Path(data_root) / "cache" / "tuab_4s_final"

# Load index
index_path = cache_dir / "index.json"
with open(index_path) as f:
    index = json.load(f)

# Process each split
for split in ['train', 'eval']:
    print(f"\nBuilding {split} arrays...")
    
    X_list = []
    y_list = []
    
    for file_path, info in tqdm(index["files"].items(), desc=f"Loading {split}"):
        if split in file_path:
            cache_file = cache_dir / info["cache_file"]
            data = torch.load(cache_file, weights_only=True)
            
            # Convert to numpy
            X_list.append(data["x"].numpy())
            y_list.append(data["y"].numpy())
    
    # Concatenate
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    print(f"{split}: X shape = {X.shape}, y shape = {y.shape}")
    
    # Save
    array_file = cache_dir / f"{split}_data.npy"
    label_file = cache_dir / f"{split}_labels.npy"
    
    print(f"Saving to {array_file}...")
    np.save(array_file, X)
    np.save(label_file, y)
    
    print(f"Saved {len(X)} windows for {split}")
    
    # Count classes
    n_normal = (y == 0).sum()
    n_abnormal = (y == 1).sum()
    print(f"Class distribution: normal={n_normal}, abnormal={n_abnormal}")

print("\nDone! Arrays ready for ultra-fast training.")
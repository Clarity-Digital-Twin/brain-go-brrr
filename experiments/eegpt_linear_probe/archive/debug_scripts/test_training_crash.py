#!/usr/bin/env python
"""Quick test to see where training crashes."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add to path
sys.path.insert(0, "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/src")
sys.path.insert(0, "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe")

from src.tuab_dataset import TUABMemoryMappedDataset

def test_dataset():
    """Test if we can load batches around where it crashes."""
    print("Testing dataset loading...")
    
    dataset = TUABMemoryMappedDataset(
        split="train",
        cache_dir="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuab_4s_final"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading around batch 154-205 (with batch_size=64)
    batch_size = 64
    crash_batch = 180  # Middle of crash range
    
    start_idx = crash_batch * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))
    
    print(f"\nTesting samples {start_idx} to {end_idx}...")
    
    for i in range(start_idx, end_idx):
        try:
            x, y = dataset[i]
            if i == start_idx:
                print(f"Sample {i}: shape={x.shape}, label={y}")
        except Exception as e:
            print(f"ERROR at sample {i}: {e}")
            return False
    
    print("All samples loaded successfully!")
    return True

if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""Build cache for TUAB dataset with 4-second windows."""

import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.brain_go_brrr.data.tuab_dataset import TUABDataset

def build_cache():
    """Build cache for 4-second window training."""
    # Configuration for 4s windows
    config = {
        "window_size": 4.0,  # 4 second windows (EEGPT pretrained)
        "window_stride": 0.25,  # 75% overlap
        "sampling_rate": 256,  # EEGPT requirement
        "preload": False,  # Don't preload to save memory during caching
        "cache_dir": "cache/tuab_4s_cache",
        "use_autoreject": False,  # Disable for speed
        "bandpass_low": 0.5,
        "bandpass_high": 50.0,
    }
    
    # Create cache directory
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building cache in {cache_dir}")
    print(f"Window size: {config['window_size']}s")
    print(f"Window stride: {config['window_stride']}s")
    print(f"Sampling rate: {config['sampling_rate']} Hz")
    
    # Build for train split
    print("\n=== Building TRAIN cache ===")
    train_dataset = TUABDataset(
        root_dir="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/TUAB",
        split="train",
        **config
    )
    
    print(f"Train dataset size: {len(train_dataset)} windows")
    
    # Access each item to trigger caching
    print("Caching train data...")
    for i in tqdm(range(min(100, len(train_dataset))), desc="Caching samples"):
        try:
            _ = train_dataset[i]
        except Exception as e:
            print(f"Error caching sample {i}: {e}")
            continue
    
    print("\n=== Building VAL cache ===")
    val_dataset = TUABDataset(
        root_dir="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/TUAB",
        split="val",
        **config
    )
    
    print(f"Val dataset size: {len(val_dataset)} windows")
    
    # Cache validation data
    print("Caching val data...")
    for i in tqdm(range(min(20, len(val_dataset))), desc="Caching samples"):
        try:
            _ = val_dataset[i]
        except Exception as e:
            print(f"Error caching sample {i}: {e}")
            continue
    
    print("\n✅ Cache building complete!")
    print(f"Cache location: {cache_dir.absolute()}")
    
if __name__ == "__main__":
    build_cache()
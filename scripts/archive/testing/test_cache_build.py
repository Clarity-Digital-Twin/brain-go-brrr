#!/usr/bin/env python
"""Quick test of cache building with just a few files."""

import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set up environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

import logging

logging.basicConfig(level=logging.INFO)

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset

# Test with just 10 files
print("Testing cache build with 10 files...")
dataset = TUABEnhancedDataset(
    root_dir=Path(os.environ["BGB_DATA_ROOT"]) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    split="train",
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
    bandpass_low=0.5,
    bandpass_high=50.0,
    preload=True,
    n_jobs=4,
    cache_mode="write",
    verbose=True,
)

print(f"Dataset created with {len(dataset)} windows")
print("Cache test successful!")

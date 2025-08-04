#!/usr/bin/env python3
"""Quick smoke test to verify training won't crash"""

import os
import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set environment BEFORE any imports that might use it
os.environ["BGB_DATA_ROOT"] = str(PROJECT_ROOT / "data")
os.environ["EEGPT_CONFIG"] = "configs/tuab_stable.yaml"

# Now safe to import
import torch
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

print("=" * 80)
print("SMOKE TEST: Verifying training won't crash")
print("=" * 80)

# Test dataset
print("\nTesting dataset loading...")
dataset = TUABCachedDataset(
    root_dir=PROJECT_ROOT / "data/datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
    cache_dir=PROJECT_ROOT / "data/cache/tuab_enhanced",
    split="train",
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
)

print(f"Dataset loaded: {len(dataset)} samples")

# Test first 5 samples
print("\nChecking first 5 samples...")
for i in range(min(5, len(dataset))):
    x, y = dataset[i]
    print(f"Sample {i}: shape={x.shape}, label={y}, min={x.min():.3f}, max={x.max():.3f}, std={x.std():.3f}")

print("\n✅ Dataset check passed")

# Quick test of model normalization
from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt

print("\nChecking EEGPT normalization...")
backbone = create_normalized_eegpt(
    checkpoint_path=PROJECT_ROOT / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
)
print(f"Normalization: mean={backbone.input_mean.item():.6f}, std={backbone.input_std.item():.6f}")
print(f"Source: {backbone._stats_source}")

if backbone.input_std.item() < 0.1:
    print("❌ WARNING: Normalization std too small!")
    sys.exit(1)

print("\n✅ All checks passed - safe to train")
print("\nTo start training:")
print("./experiments/eegpt_linear_probe/RUN_STABLE_TRAINING.sh")
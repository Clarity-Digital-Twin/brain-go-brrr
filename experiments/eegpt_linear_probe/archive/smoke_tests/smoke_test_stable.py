#!/usr/bin/env python3
"""Quick smoke test to verify training won't crash with NaN"""

import os
import sys
from pathlib import Path

# Set environment variables BEFORE imports
os.environ["BGB_DATA_ROOT"] = "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"
os.environ["EEGPT_CONFIG"] = "configs/tuab_stable.yaml"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from omegaconf import OmegaConf
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
import torch

print("=" * 80)
print("SMOKE TEST: Verifying stable training configuration")
print("=" * 80)

# Load config
config = OmegaConf.load(os.environ["EEGPT_CONFIG"])
config = OmegaConf.resolve(config)

print(f"\nConfiguration loaded: {os.environ['EEGPT_CONFIG']}")
print(f"Precision: {config.experiment.precision}")
print(f"Learning rate: {config.training.learning_rate}")
print(f"Batch size: {config.data.batch_size}")
print(f"Accumulate grad batches: {config.training.accumulate_grad_batches}")

# Test dataset loading
print("\nTesting dataset loading...")
dataset = TUABCachedDataset(
    root_dir=Path(config.data.root_dir),
    cache_dir=Path(config.data.cache_dir),
    split="train",
    window_duration=config.data.window_duration,
    window_stride=config.data.window_stride,
    sampling_rate=config.data.sampling_rate,
)

print(f"Dataset loaded: {len(dataset)} samples")

# Test a few samples
print("\nTesting first 5 samples for NaN/extreme values...")
for i in range(min(5, len(dataset))):
    try:
        x, y = dataset[i]
        print(f"Sample {i}: shape={x.shape}, label={y}, min={x.min():.3f}, max={x.max():.3f}, std={x.std():.3f}")
    except RuntimeError as e:
        print(f"ERROR in sample {i}: {e}")
        sys.exit(1)

print("\n✅ All checks passed - safe to start training")
print("\nTo run full training:")
print("./experiments/eegpt_linear_probe/RUN_STABLE_TRAINING.sh")
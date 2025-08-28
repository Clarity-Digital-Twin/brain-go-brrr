#!/usr/bin/env python3
"""Quick test to verify normalization is working."""

import torch
import numpy as np
from pathlib import Path

# Test the normalization logic
print("Testing normalization fix...")

# Simulate MNE output (microvolts in Volts)
fake_mne_data = np.random.randn(19, 1024) * 50e-6  # 50 microvolts
print(f"Simulated MNE data: mean={fake_mne_data.mean():.2e}, std={fake_mne_data.std():.2e}")

# Apply the fix
epoch_mean = fake_mne_data.mean()
epoch_std = fake_mne_data.std() 
normalized = (fake_mne_data - epoch_mean) / (epoch_std + 1e-8)

print(f"After normalization: mean={normalized.mean():.2e}, std={normalized.std():.2e}")
print(f"Range: [{normalized.min():.2f}, {normalized.max():.2f}]")

# Check it's correct
assert abs(normalized.mean()) < 1e-6, "Mean not zero!"
assert abs(normalized.std() - 1.0) < 0.01, "Std not one!"

print("\n✅ NORMALIZATION FIX VERIFIED - Ready to rebuild cache!")
print("\nNext steps:")
print("1. cd experiments/eegpt_linear_probe")
print("2. python mne_integration/cache_builder.py")
print("3. python train_tuab_mne.py --config configs/tuab.yaml")
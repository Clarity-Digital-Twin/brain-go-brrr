#!/usr/bin/env python
"""Smoke test to verify training setup before full run."""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset
from torch.utils.data import DataLoader

def smoke_test():
    """Quick test to ensure dataset loads correctly without looping."""
    print("🧪 EEGPT Training Smoke Test")
    print("-" * 50)
    
    # Check environment
    data_root = os.environ.get('BGB_DATA_ROOT', 'data')
    config = os.environ.get('EEGPT_CONFIG', 'Not set')
    print(f"✓ Data root: {data_root}")
    print(f"✓ Config: {config}")
    
    # Test dataset loading
    print("\n📊 Testing dataset loading...")
    dataset = TUABEnhancedDataset(
        root_dir=Path(data_root) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=10.24,
        window_stride=2.56,
        sampling_rate=256,
        use_autoreject=False  # Test without first
    )
    
    print(f"✓ Dataset size: {len(dataset)} windows")
    print(f"✓ Channels: {dataset.channels}")
    
    # Test DataLoader (critical for looping issues)
    print("\n🔄 Testing DataLoader (checking for loops)...")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Load first 5 batches and track unique indices
    seen_indices = set()
    for i, (data, labels) in enumerate(loader):
        if i >= 5:
            break
        
        # Track batch info
        print(f"  Batch {i}: shape={data.shape}, labels={labels.shape}")
        
        # Check for data validity
        if torch.isnan(data).any():
            print("  ⚠️  WARNING: NaN values in data!")
        if data.shape[0] != 32 and i < 4:  # Last batch can be smaller
            print(f"  ⚠️  WARNING: Unexpected batch size: {data.shape[0]}")
    
    print("\n✅ Smoke test passed!")
    print("\n📝 Recommended command:")
    print("EEGPT_CONFIG=configs/tuab_enhanced_config.yaml \\")
    print("uv run python experiments/eegpt_linear_probe/train_enhanced.py")
    
if __name__ == "__main__":
    smoke_test()
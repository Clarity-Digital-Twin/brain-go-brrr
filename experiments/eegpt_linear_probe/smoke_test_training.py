#!/usr/bin/env python
"""Smoke test for EEGPT training - catches errors BEFORE 6-hour runs."""

import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

def smoke_test():
    """Quick validation that training script will run."""
    print("🔍 Running smoke test...")
    
    # Test 1: Imports
    print("✓ Testing imports...")
    try:
        from brain_go_brrr.data.tuab_dataset import TUABDataset
        from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe
        from experiments.eegpt_linear_probe.train_tuab_probe import LinearProbeTrainer
        print("  ✓ All imports successful")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Test 2: Check data paths
    print("✓ Checking data paths...")
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    eegpt_checkpoint = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    tuab_path = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    
    if not eegpt_checkpoint.exists():
        print(f"  ✗ EEGPT checkpoint missing: {eegpt_checkpoint}")
        return False
    print(f"  ✓ EEGPT checkpoint found")
    
    if not tuab_path.exists():
        print(f"  ✗ TUAB dataset missing: {tuab_path}")
        return False
    print(f"  ✓ TUAB dataset found")
    
    # Test 3: Initialize components
    print("✓ Testing component initialization...")
    try:
        # Create probe
        probe = AbnormalityDetectionProbe(eegpt_checkpoint, n_input_channels=20)
        print("  ✓ Probe initialized")
        
        # Create trainer (WITHOUT warmup_epochs!)
        trainer = LinearProbeTrainer(
            model=probe,
            learning_rate=5e-4,
            weight_decay=0.05,
        )
        print("  ✓ Trainer initialized")
        
    except TypeError as e:
        print(f"  ✗ Initialization failed - bad arguments: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        return False
    
    # Test 4: Quick data load
    print("✓ Testing data loading (first file only)...")
    try:
        dataset = TUABDataset(
            root_dir=tuab_path,
            split="train",
            window_duration=8.0,
            window_stride=8.0,
            sampling_rate=256,
            preload=False,
            normalize=True,
        )
        
        if len(dataset) == 0:
            print("  ✗ Dataset is empty!")
            return False
            
        # Try loading one sample
        x, y = dataset[0]
        print(f"  ✓ Loaded sample: shape={x.shape}, label={y}")
        
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False
    
    print("\n✅ All smoke tests passed! Safe to start training.")
    return True


if __name__ == "__main__":
    if smoke_test():
        print("\n🚀 Run this to start training:")
        print("   bash launch_bulletproof_training.sh")
    else:
        print("\n❌ Fix the errors above before training!")
        sys.exit(1)
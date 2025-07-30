#!/usr/bin/env python
"""Verify EEGPT Linear Probe setup is ready for training."""

import os
import sys
from pathlib import Path

# Check Python version
print(f"✓ Python {sys.version.split()[0]}")

# Check critical imports
try:
    import torch

    print(f"✓ PyTorch {torch.__version__}")

    import pytorch_lightning as pl

    print(f"✓ PyTorch Lightning {pl.__version__}")

    from brain_go_brrr.tasks.abnormality_detection import AbnormalityDetectionProbe  # noqa: F401

    print("✓ Linear probe modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Check data paths
data_root = os.getenv("BGB_DATA_ROOT", "data")
print(f"\n📁 Data root: {data_root}")

# Check for EEGPT checkpoint
eegpt_path = Path(data_root) / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
if eegpt_path.exists():
    print(f"✓ EEGPT checkpoint found: {eegpt_path}")
    print(f"  Size: {eegpt_path.stat().st_size / 1e6:.1f} MB")
else:
    print(f"✗ EEGPT checkpoint not found at: {eegpt_path}")
    print(
        "  Download from: https://github.com/BINE022/EEGPT/releases/download/v1.0/eegpt_mcae_58chs_4s_large4E.ckpt"
    )

# Check for TUAB dataset
tuab_path = Path(data_root) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
if tuab_path.exists():
    print(f"\n✓ TUAB dataset directory found: {tuab_path}")

    # Check splits
    for split in ["train", "eval"]:
        split_path = tuab_path / split
        if split_path.exists():
            normal = (
                len(list((split_path / "normal").glob("**/*.edf")))
                if (split_path / "normal").exists()
                else 0
            )
            abnormal = (
                len(list((split_path / "abnormal").glob("**/*.edf")))
                if (split_path / "abnormal").exists()
                else 0
            )
            print(f"  {split}: {normal} normal, {abnormal} abnormal")
        else:
            print(f"  ✗ {split} split not found")
else:
    print(f"\n✗ TUAB dataset not found at: {tuab_path}")
    print("  Expected structure:")
    print("  datasets/external/tuh_eeg_abnormal/v3.0.1/edf/")
    print("    ├── train/")
    print("    │   ├── normal/*.edf")
    print("    │   └── abnormal/*.edf")
    print("    └── eval/")
    print("        ├── normal/*.edf")
    print("        └── abnormal/*.edf")

# Check GPU
if torch.cuda.is_available():
    print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\n⚠️  No GPU detected - training will be slow")

# Show training command
print("\n" + "=" * 60)
print("To start training (once data is ready):")
print("=" * 60)
print(f"cd {Path(__file__).parent}")
print("python train_tuab_probe.py")
print("\nOr with custom data root:")
print("export BGB_DATA_ROOT=/path/to/your/data")
print("python train_tuab_probe.py")
print("=" * 60)

#!/usr/bin/env python
"""Quick smoke test to verify training pipeline before full GPU run."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from omegaconf import OmegaConf  # noqa: E402

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from train_final_proper import main  # noqa: E402


def smoke_test():
    """Run minimal training to verify pipeline."""
    print("=" * 60)
    print("SMOKE TEST - Verify training pipeline")
    print("=" * 60)

    # Override config for smoke test
    cfg = OmegaConf.load("configs/tuab_config.yaml")

    # Minimal settings for quick test
    cfg.training.epochs = 1
    cfg.training.batch_size = 32  # Smaller for Mac
    cfg.training.num_workers = 0  # MPS compatible

    # Use only 1% of data for smoke test
    cfg.trainer = OmegaConf.create(
        {
            "limit_train_batches": 0.01,  # 1% of training data
            "limit_val_batches": 0.05,  # 5% of validation data
        }
    )

    print("Smoke test configuration:")
    print(f"  - Epochs: {cfg.training.epochs}")
    print(f"  - Batch size: {cfg.training.batch_size}")
    print("  - Train batches: 1% (~73 batches)")
    print("  - Val batches: 5% (~25 batches)")
    print("  - Estimated time: ~30 minutes on Mac")
    print("=" * 60)

    # Save overridden config
    OmegaConf.save(cfg, "configs/smoke_test_config.yaml")

    # Run with smoke test config
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Clear any CLI args

    try:
        main()
        print("\n✅ SMOKE TEST PASSED!")
        print("Pipeline is working correctly. Ready for GPU training.")
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        raise
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    smoke_test()

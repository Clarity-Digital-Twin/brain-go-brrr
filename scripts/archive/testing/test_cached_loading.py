#!/usr/bin/env python
"""Test cached dataset loading to diagnose hanging issue."""

import os
import sys
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONUNBUFFERED"] = "1"

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_cached_loading():
    """Test loading from cache."""
    from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset

    data_root = Path(os.environ["BGB_DATA_ROOT"])

    logger.info("Testing cached dataset loading...")
    logger.info(f"Data root: {data_root}")

    # Test parameters matching the cache
    params = {
        "root_dir": data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        "split": "train",
        "window_duration": 8.0,
        "window_stride": 4.0,
        "sampling_rate": 256,
        "channels": [
            "FP1",
            "FP2",
            "F7",
            "F3",
            "FZ",
            "F4",
            "F8",
            "T3",
            "C3",
            "CZ",
            "C4",
            "T4",
            "T5",
            "P3",
            "PZ",
            "P4",
            "T6",
            "O1",
            "O2",
        ],
        "preload": False,
        "normalize": True,
        "bandpass_low": 0.5,
        "bandpass_high": 50.0,
        "notch_freq": None,
        "cache_dir": data_root / "cache/tuab_enhanced",
        "use_old_naming": True,
        "n_jobs": 1,  # Single thread for debugging
        "use_autoreject": False,
        "cache_mode": "readonly",
        "verbose": True,
    }

    start_time = time.time()
    logger.info("Creating dataset with cache_mode='readonly'...")

    try:
        dataset = TUABEnhancedDataset(**params)
        logger.info(f"Dataset created in {time.time() - start_time:.2f}s")
        logger.info(f"Dataset length: {len(dataset)}")

        # Try to load one sample
        logger.info("Loading first sample...")
        sample_start = time.time()
        x, y = dataset[0]
        logger.info(f"Sample loaded in {time.time() - sample_start:.3f}s")
        logger.info(f"Sample shape: {x.shape}, label: {y}")

        return True

    except Exception as e:
        logger.error(f"Failed to load cached dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cached_loading()
    sys.exit(0 if success else 1)

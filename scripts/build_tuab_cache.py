#!/usr/bin/env python
"""Build complete TUAB cache for fast training startup.

This script pre-generates all windows for the TUAB dataset to avoid
the slow cache generation during training startup.
"""

import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set up environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset


def build_cache(split="train", max_files=None):
    """Build cache for a specific split."""
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    root_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    cache_dir = data_root / "cache/tuab_enhanced"

    logger.info(f"Building cache for {split} split...")
    logger.info(f"Root dir: {root_dir}")
    logger.info(f"Cache dir: {cache_dir}")

    # CRITICAL: Use consistent window parameters
    # Standardizing on 8s @ 256Hz as recommended
    dataset_params = {
        "root_dir": root_dir,
        "split": split,
        "window_duration": 8.0,  # 8 seconds
        "window_stride": 4.0,  # 50% overlap for train
        "sampling_rate": 256,  # 256 Hz
        "bandpass_low": 0.5,
        "bandpass_high": 50.0,
        "notch_freq": None,  # Use correct parameter name
        "preload": False,  # Don't load all files into memory!
        "normalize": True,
        "use_autoreject": False,  # Skip AR for cache building
        "cache_dir": cache_dir,
        "channels": [  # Use correct parameter name
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
        "n_jobs": 8,  # Use 8 cores for parallel processing
        "verbose": True,
        "cache_mode": "write",  # Ensure we're in write mode
    }

    if max_files:
        dataset_params["max_files"] = max_files

    # Adjust stride for eval split (no overlap)
    if split == "eval":
        dataset_params["window_stride"] = 8.0

    start_time = time.time()

    # Create dataset - this will trigger cache generation
    logger.info(f"Creating {split} dataset...")
    dataset = TUABEnhancedDataset(**dataset_params)

    # Access length to ensure all windows are cached
    n_windows = len(dataset)
    logger.info(f"Dataset has {n_windows} windows")

    # Force cache generation by iterating through ALL windows
    logger.info(f"Generating cache for {n_windows} windows...")
    logger.info("This will take a while, but only needs to be done once!")

    # Process in batches to show progress
    batch_size = 1000
    for start_idx in tqdm(range(0, n_windows, batch_size), desc=f"Building {split} cache"):
        end_idx = min(start_idx + batch_size, n_windows)
        for i in range(start_idx, end_idx):
            try:
                _ = dataset[i]  # This triggers cache writing
            except Exception as e:
                logger.warning(f"Failed to cache window {i}: {e}")
                continue

    elapsed = time.time() - start_time
    logger.info(f"Cache building for {split} completed in {elapsed:.1f} seconds")
    logger.info(f"Generated {n_windows} windows")

    return n_windows


def main():
    """Build complete TUAB cache."""
    logger.info("=" * 80)
    logger.info("TUAB CACHE BUILDER")
    logger.info("=" * 80)
    logger.info("This will pre-generate all windows for fast training startup")
    logger.info("Configuration: 8s windows @ 256Hz")
    logger.info("=" * 80)

    # Check if cache already exists
    cache_dir = Path(os.environ["BGB_DATA_ROOT"]) / "cache/tuab_enhanced"
    existing_files = list(cache_dir.glob("*.pkl"))
    if existing_files:
        logger.warning(f"Found {len(existing_files)} existing cache files")
        response = input("Clear existing cache? [y/N]: ")
        if response.lower() == "y":
            logger.info("Clearing existing cache...")
            for f in existing_files:
                f.unlink()
        else:
            logger.info("Keeping existing cache files")

    # Build cache for both splits
    total_windows = 0

    # Train split
    logger.info("\n" + "=" * 40)
    logger.info("Building TRAIN split cache...")
    logger.info("=" * 40)
    train_windows = build_cache("train")
    total_windows += train_windows

    # Eval split
    logger.info("\n" + "=" * 40)
    logger.info("Building EVAL split cache...")
    logger.info("=" * 40)
    eval_windows = build_cache("eval")
    total_windows += eval_windows

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("CACHE BUILDING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total windows cached: {total_windows}")
    logger.info(f"Train windows: {train_windows}")
    logger.info(f"Eval windows: {eval_windows}")
    logger.info(f"Cache location: {cache_dir}")
    logger.info("=" * 80)
    logger.info("You can now run training and it will load instantly!")


if __name__ == "__main__":
    main()

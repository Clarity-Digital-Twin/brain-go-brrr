#!/usr/bin/env python
"""Test dataset loading speed improvements."""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader  # noqa: E402

from brain_go_brrr.data.tuab_dataset import TUABDataset  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_speed():
    """Test dataset loading speed."""
    # Get data root
    data_root = os.getenv("BGB_DATA_ROOT", "data")
    dataset_root = Path(data_root) / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
    cache_dir = Path(data_root) / "cache/tuab_preprocessed"

    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Cache dir: {cache_dir}")

    # Create small dataset for testing
    logger.info("Creating dataset...")
    dataset = TUABDataset(
        root_dir=dataset_root,
        split="train",
        sampling_rate=256,
        window_duration=8.0,
        window_stride=8.0,
        normalize=True,
        cache_dir=cache_dir,
    )

    logger.info(f"Dataset has {len(dataset)} windows from {len(dataset.file_list)} files")

    # Test single sample loading
    logger.info("\nTesting single sample loading...")
    start = time.time()
    sample, label = dataset[0]
    elapsed = time.time() - start
    logger.info(f"First sample loaded in {elapsed:.3f}s")
    logger.info(f"Sample shape: {sample.shape}, label: {label}")

    # Test batch loading
    logger.info("\nTesting batch loading...")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    # Time first few batches
    logger.info("Loading first 5 batches...")
    batch_times = []
    for i, (batch, labels) in enumerate(loader):
        start = time.time()
        if i >= 5:
            break
        elapsed = time.time() - start
        batch_times.append(elapsed)
        logger.info(f"Batch {i + 1}: {batch.shape}, labels: {labels.shape} - {elapsed:.3f}s")

    avg_time = sum(batch_times) / len(batch_times) if batch_times else 0
    logger.info(f"\nAverage batch time: {avg_time:.3f}s")

    # Test with cache
    logger.info("\nTesting cached loading...")
    start = time.time()
    for i in range(100):
        idx = i % len(dataset)
        sample, label = dataset[idx]
    elapsed = time.time() - start
    logger.info(f"Loaded 100 samples (with cache) in {elapsed:.3f}s")
    logger.info(f"Average per sample: {elapsed / 100:.4f}s")

    logger.info("\nTest complete!")


if __name__ == "__main__":
    test_dataset_speed()

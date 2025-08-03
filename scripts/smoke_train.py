#!/usr/bin/env python
"""Smoke test for cached dataloader - verifies cache is being used properly."""

import os
import sys
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set environment
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import logging
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test cached dataset loading speed."""
    
    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    parser.add_argument("--n_batches", type=int, default=10, help="Number of batches to test")
    args = parser.parse_args()
    
    # Default config
    config = {
        "data": {
            "cache_mode": "write",
            "cache_dir": "${BGB_DATA_ROOT}/cache/tuab_enhanced",
            "window_duration": 8.0,
            "sampling_rate": 256,
            "window_stride": 4.0,
            "bandpass_low": 0.5,
            "bandpass_high": 50.0,
            "notch_filter": None,
            "batch_size": 32,
            "num_workers": 2,
            "channel_names": [
                'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                'T3', 'C3', 'CZ', 'C4', 'T4',
                'T5', 'P3', 'PZ', 'P4', 'T6',
                'O1', 'O2'
            ]
        }
    }
    
    # Apply overrides
    for override in args.overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        # Parse value
        try:
            value = eval(value)
        except:
            pass  # Keep as string
        d[keys[-1]] = value
    
    cfg = OmegaConf.create(config)
    # Resolve environment variables
    OmegaConf.resolve(cfg)
    
    logger.info("=" * 80)
    logger.info("CACHED DATALOADER SMOKE TEST")
    logger.info("=" * 80)
    logger.info(f"Config:")
    logger.info(f"  cache_mode: {cfg.data.cache_mode}")
    logger.info(f"  cache_dir: {cfg.data.cache_dir}")
    logger.info(f"  window: {cfg.data.window_duration}s @ {cfg.data.sampling_rate}Hz")
    logger.info(f"  batch_size: {cfg.data.batch_size}")
    logger.info("=" * 80)
    
    # Create dataset
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    
    logger.info("Creating dataset...")
    start_time = time.time()
    
    dataset = TUABEnhancedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        window_duration=cfg.data.window_duration,
        window_stride=cfg.data.window_stride,
        sampling_rate=cfg.data.sampling_rate,
        channels=cfg.data.channel_names,
        preload=False,
        normalize=True,
        bandpass_low=cfg.data.bandpass_low,
        bandpass_high=cfg.data.bandpass_high,
        notch_freq=cfg.data.notch_filter,
        cache_dir=Path(cfg.data.cache_dir),
        use_old_naming=True,
        n_jobs=4,
        use_autoreject=False,
        cache_mode=cfg.data.cache_mode,
    )
    
    dataset_time = time.time() - start_time
    logger.info(f"Dataset created in {dataset_time:.2f} seconds")
    logger.info(f"Dataset has {len(dataset)} windows")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Test loading batches
    logger.info(f"\nLoading {args.n_batches} batches...")
    cache_hits = 0
    cache_misses = 0
    
    batch_times = []
    for i, (data, labels) in enumerate(dataloader):
        if i >= args.n_batches:
            break
        
        batch_start = time.time()
        
        # Access data to trigger loading
        assert data.shape[0] == cfg.data.batch_size
        assert data.shape[1] == len(cfg.data.channel_names)
        expected_samples = int(cfg.data.window_duration * cfg.data.sampling_rate)
        assert data.shape[2] == expected_samples
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Check if cache was used (heuristic: very fast loading)
        if batch_time < 0.5:  # Less than 0.5s suggests cache hit
            cache_hits += 1
        else:
            cache_misses += 1
        
        logger.info(f"  Batch {i+1}: {batch_time:.3f}s - shape {data.shape}")
    
    # Summary
    avg_batch_time = sum(batch_times) / len(batch_times)
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Dataset creation: {dataset_time:.2f}s")
    logger.info(f"Average batch time: {avg_batch_time:.3f}s")
    logger.info(f"Cache performance: {cache_hits}/{args.n_batches} hits")
    
    if cfg.data.cache_mode == "readonly" and cache_misses > 0:
        logger.error(f"❌ CACHE MISSES in readonly mode! Check configuration")
        sys.exit(1)
    elif avg_batch_time > 2.0:
        logger.warning(f"⚠️  Slow loading detected - cache may not be working")
        sys.exit(1)
    else:
        logger.info(f"✅ Cache working properly!")
        sys.exit(0)


if __name__ == "__main__":
    main()
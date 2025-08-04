#!/usr/bin/env python
"""Smoke test for cached training - MUST PASS before training"""

import os
import sys
import time
from pathlib import Path

# Setup
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["BGB_DATA_ROOT"] = str(project_root / "data")

def test_cache():
    """Test that cache works properly"""
    print("=" * 60)
    print("CACHE SMOKE TEST")
    print("=" * 60)
    
    # 1. Check paths
    data_root = Path(os.environ["BGB_DATA_ROOT"])
    cache_dir = data_root / "cache/tuab_enhanced"
    cache_index = data_root / "cache/tuab_index.json"
    
    print(f"Cache dir: {cache_dir}")
    print(f"Cache index: {cache_index}")
    print(f"Cache dir exists: {cache_dir.exists()}")
    print(f"Cache index exists: {cache_index.exists()}")
    
    if not cache_index.exists():
        print("ERROR: Cache index not found!")
        return False
    
    # 2. Count cache files
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"\nCache files: {len(cache_files)}")
    if len(cache_files) == 0:
        print("ERROR: No cache files found!")
        return False
    
    # 3. Test dataset loading
    print("\n" + "-" * 40)
    print("Testing TUABCachedDataset...")
    print("-" * 40)
    
    from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
    
    start_time = time.time()
    dataset = TUABCachedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        sampling_rate=256,
        window_duration=8.0,
        window_stride=4.0,
        normalize=True,
        cache_dir=cache_dir,
        cache_index_path=cache_index,
    )
    load_time = time.time() - start_time
    
    print(f"Dataset loaded in {load_time:.2f} seconds")
    print(f"Dataset length: {len(dataset)}")
    
    if load_time > 10:
        print("WARNING: Dataset loading too slow!")
    
    # 4. Test sample loading
    print("\nTesting sample loading...")
    sample_start = time.time()
    x, y = dataset[0]
    sample_time = time.time() - sample_start
    
    print(f"First sample loaded in {sample_time:.3f} seconds")
    print(f"Sample shape: {x.shape}")
    print(f"Label: {y}")
    
    if sample_time > 1:
        print("WARNING: Sample loading too slow!")
    
    # 5. Test batch loading
    print("\nTesting batch loading...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    batch_start = time.time()
    batch = next(iter(loader))
    batch_time = time.time() - batch_start
    
    print(f"First batch loaded in {batch_time:.3f} seconds")
    print(f"Batch shapes: X={batch[0].shape}, y={batch[1].shape}")
    
    print("\n" + "=" * 60)
    if load_time < 10 and sample_time < 1 and batch_time < 2:
        print("✅ CACHE TEST PASSED - Ready for training!")
        return True
    else:
        print("❌ CACHE TEST FAILED - Fix issues before training!")
        return False


if __name__ == "__main__":
    success = test_cache()
    sys.exit(0 if success else 1)
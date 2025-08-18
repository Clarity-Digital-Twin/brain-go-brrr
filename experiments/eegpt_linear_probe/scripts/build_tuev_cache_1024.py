#!/usr/bin/env python
"""Rebuild TUEV cache with 1024 samples (4.0s windows) for EEGPT compatibility."""

import shutil
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tuev_dataset import TUEVDataset, build_tuev_cache


def main():
    """Rebuild TUEV cache with correct window size."""
    
    print("=" * 60)
    print("🔧 REBUILDING TUEV CACHE WITH 1024 SAMPLES")
    print("=" * 60)
    print()
    
    # Cache directory
    cache_dir = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13_1024")
    
    # Remove old cache if exists
    old_cache = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/tuev_table13")
    if old_cache.exists():
        print(f"⚠️  Found old cache with 1000 samples at: {old_cache}")
        print("   This cache is incompatible with EEGPT (not divisible by 64)")
        response = input("   Delete old cache? [y/N]: ")
        if response.lower() == 'y':
            shutil.rmtree(old_cache)
            print("   ✅ Old cache deleted")
        else:
            print("   ⏩ Keeping old cache, using new directory")
    
    # Build new cache
    print()
    print(f"📦 Building new cache at: {cache_dir}")
    print("   Window size: 1024 samples (4.0s @ 256Hz)")
    print("   This is compatible with EEGPT patch size of 64")
    print()
    
    # Build for both splits
    for split in ['train', 'eval']:
        print(f"\n🔨 Building {split.upper()} cache...")
        build_tuev_cache(
            root_dir=Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/external/tuh_eeg/TUEV/v2.0.1"),
            cache_dir=cache_dir,
            split=split,
            max_workers=8
        )
    
    print()
    print("=" * 60)
    print("✅ CACHE REBUILD COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Update config to use new cache:")
    print("   cache_dir: .../data/cache/tuev_table13_1024")
    print()
    print("2. Launch training with:")
    print("   python train_tuev_aligned.py \\")
    print("     --config configs/tuev_table13_aligned.yaml \\")
    print("     --device cuda \\")
    print("     --seed 42 \\")
    print("     --use-cache")


if __name__ == "__main__":
    main()
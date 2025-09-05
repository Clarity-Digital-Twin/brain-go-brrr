#!/usr/bin/env python
"""Test TUEV cache building with single file."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset

def test_cache_build():
    """Test building cache for TUEV dataset."""
    
    root_dir = Path('data/datasets/tuev')
    
    # Check if TUEV data exists
    if not root_dir.exists():
        print(f"ERROR: TUEV dataset not found at {root_dir}")
        return False
        
    # Check for EDF files
    edf_files = list(root_dir.glob('**/train/**/*.edf'))
    if not edf_files:
        print(f"ERROR: No EDF files found in {root_dir}/edf/train/")
        return False
    
    print(f"Found {len(edf_files)} EDF files in train split")
    
    try:
        # Try to build cache with force_rebuild
        print("\n[1] Testing cache build...")
        dataset = TUEVMNEDataset(
            root_dir=root_dir,
            split='train',
            cache_dir=Path('/tmp/tuev_cache_test'),
            force_rebuild=True
        )
        
        print(f"✅ Cache built successfully!")
        print(f"   - Total windows: {len(dataset)}")
        
        # Test loading a sample
        print("\n[2] Testing sample loading...")
        if len(dataset) > 0:
            x, y = dataset[0]
            print(f"✅ Sample loaded successfully!")
            print(f"   - Shape: {x.shape}")
            print(f"   - Label: {y}")
            print(f"   - Data range: [{x.min():.3f}, {x.max():.3f}] mV")
            
            # Verify shape
            assert x.shape == (20, 1024), f"Wrong shape: {x.shape}"
            assert isinstance(y, int), f"Label not int: {type(y)}"
            
        return True
        
    except Exception as e:
        print(f"❌ Cache build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cache_build()
    sys.exit(0 if success else 1)
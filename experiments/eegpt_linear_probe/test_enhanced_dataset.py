#!/usr/bin/env python
"""Quick test of enhanced dataset instantiation."""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset

def test_dataset_instantiation():
    """Test that enhanced dataset can be instantiated cleanly."""
    print("Testing TUABEnhancedDataset instantiation...")
    
    data_root = project_root / "data"
    
    # Minimal instantiation
    dataset = TUABEnhancedDataset(
        root_dir=data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf",
        split="train",
        preload=False
    )
    
    print(f"✓ Dataset created successfully!")
    print(f"  Total windows: {len(dataset)}")
    print(f"  Classes: {dataset.class_counts}")
    
    # Test getting one sample
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"✓ Sample loaded: shape={x.shape}, label={y}")
    
    return True


if __name__ == "__main__":
    test_dataset_instantiation()
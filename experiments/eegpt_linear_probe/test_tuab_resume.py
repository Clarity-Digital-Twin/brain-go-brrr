"""Test TUAB resume maintains sample order."""

import torch
from torch.utils.data import DataLoader, Subset
from datasets.tuab_dataset import TUABMemoryMappedDataset
from utils.custom_collate_fixed import collate_eeg_batch_fixed
from pathlib import Path
import os


def test_subset_resume():
    """Test that Subset-based resume gives the same sample sequence."""
    # Setup
    data_root = os.environ.get(
        "BGB_DATA_ROOT", "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"
    )
    cache_dir = Path(data_root) / "cache" / "tuab_4s_final"
    
    if not cache_dir.exists():
        print(f"⚠️  Cache not found at {cache_dir}, skipping test")
        return
    
    # Create dataset
    dataset = TUABMemoryMappedDataset(cache_dir=cache_dir, split="train")
    batch_size = 64
    
    # Create normal loader
    full_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_eeg_batch_fixed,
    )
    
    # Collect first 10 batches of sample indices (using data shape as proxy for identity)
    full_batch_shapes = []
    for i, (data, labels) in enumerate(full_loader):
        if i >= 10:
            break
        # Store batch info (use mean as fingerprint since we can't access indices directly)
        full_batch_shapes.append({
            'batch_idx': i,
            'shape': data.shape,
            'mean': data.mean().item(),
            'std': data.std().item(),
            'label_sum': labels.sum().item()
        })
    
    # Now simulate resume at batch 5
    start_batch = 5
    sample_offset = start_batch * batch_size
    remaining_indices = list(range(sample_offset, len(dataset)))
    
    # Create subset loader
    subset = Subset(dataset, remaining_indices)
    subset_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_eeg_batch_fixed,
    )
    
    # Collect batches from subset (should match batches 5-9 from full)
    subset_batch_shapes = []
    for i, (data, labels) in enumerate(subset_loader):
        if i >= 5:  # Get 5 batches
            break
        subset_batch_shapes.append({
            'batch_idx': start_batch + i,
            'shape': data.shape,
            'mean': data.mean().item(),
            'std': data.std().item(),
            'label_sum': labels.sum().item()
        })
    
    # Compare: subset batches should match full batches 5-9
    matches = 0
    for i in range(5):
        full_batch = full_batch_shapes[start_batch + i]
        subset_batch = subset_batch_shapes[i]
        
        # Check fingerprints match (within floating point tolerance)
        shape_match = full_batch['shape'] == subset_batch['shape']
        mean_match = abs(full_batch['mean'] - subset_batch['mean']) < 1e-5
        std_match = abs(full_batch['std'] - subset_batch['std']) < 1e-5
        label_match = full_batch['label_sum'] == subset_batch['label_sum']
        
        if shape_match and mean_match and std_match and label_match:
            matches += 1
            print(f"✓ Batch {start_batch + i}: Full and subset match")
        else:
            print(f"✗ Batch {start_batch + i}: Mismatch!")
            print(f"  Full:   mean={full_batch['mean']:.6f}, std={full_batch['std']:.6f}")
            print(f"  Subset: mean={subset_batch['mean']:.6f}, std={subset_batch['std']:.6f}")
    
    assert matches == 5, f"Only {matches}/5 batches matched after resume"
    print(f"\n✅ Resume test passed: All {matches} batches match perfectly")


if __name__ == "__main__":
    test_subset_resume()
    print("\n✅ TUAB resume test completed successfully!")
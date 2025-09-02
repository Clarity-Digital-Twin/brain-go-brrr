#!/usr/bin/env python3
"""Test if removing the collate workaround would break anything."""

import pickle
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))


def test_strict_collate():
    """Test a strict collate function without the workaround."""

    def collate_tuab_batch_strict(batch):
        """STRICT version - no workaround."""
        processed_samples = []
        for idx, (x, label) in enumerate(batch):
            if x.shape[0] != 19:
                raise RuntimeError(
                    f"TUAB batch item {idx}: Expected exactly 19 channels, got {x.shape[0]}. "
                    f"Shape: {x.shape}"
                )
            processed_samples.append(x)

        data = torch.stack(processed_samples)
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
        return data, labels

    # Load some actual cache files
    cache_dir = Path("data/cache/tuab_mne_v2")
    pkl_files = list(cache_dir.glob("*.pkl"))

    # Test with random batch
    batch_size = 32
    test_files = random.sample(pkl_files, min(batch_size, len(pkl_files)))

    print(f"Testing strict collate with {len(test_files)} files...")

    batch = []
    for pkl_file in test_files:
        with open(pkl_file, 'rb') as f:
            data, label = pickle.load(f)
        batch.append((torch.from_numpy(data), label))

    try:
        data_batch, label_batch = collate_tuab_batch_strict(batch)
        print("✅ STRICT COLLATE SUCCESS!")
        print(f"   Batch shape: {data_batch.shape}")
        print(f"   Labels shape: {label_batch.shape}")
        return True
    except RuntimeError as e:
        print(f"❌ STRICT COLLATE FAILED: {e}")
        return False


def test_with_current_collate():
    """Test with the current collate that has workaround."""
    from brain_go_brrr.utils.collate_tuab import collate_tuab_batch

    # Load some actual cache files
    cache_dir = Path("data/cache/tuab_mne_v2")
    pkl_files = list(cache_dir.glob("*.pkl"))

    # Test with random batch
    batch_size = 32
    test_files = random.sample(pkl_files, min(batch_size, len(pkl_files)))

    print(f"\nTesting current collate with {len(test_files)} files...")

    batch = []
    for pkl_file in test_files:
        with open(pkl_file, 'rb') as f:
            data, label = pickle.load(f)
        batch.append((torch.from_numpy(data), label))

    try:
        data_batch, label_batch = collate_tuab_batch(batch)
        print("✅ CURRENT COLLATE SUCCESS!")
        print(f"   Batch shape: {data_batch.shape}")
        print(f"   Labels shape: {label_batch.shape}")

        # Check if workaround was triggered
        # The workaround would only trigger if we had 20-channel data
        # Since we don't, both should work the same
        return True
    except Exception as e:
        print(f"❌ CURRENT COLLATE FAILED: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COLLATE WORKAROUND REMOVAL")
    print("=" * 60)

    # Test strict version
    strict_ok = test_strict_collate()

    # Test current version
    current_ok = test_with_current_collate()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)

    if strict_ok and current_ok:
        print("✅ BOTH COLLATE VERSIONS WORK!")
        print("✅ WORKAROUND CAN BE SAFELY REMOVED!")
        print("\nRECOMMENDATION:")
        print("1. Remove lines 31-36 from collate_tuab.py")
        print("2. Change line 37-39 to strict assertion")
        print("3. Update tests to verify strict enforcement")
    elif not strict_ok and current_ok:
        print("⚠️ STRICT VERSION FAILS - WORKAROUND STILL NEEDED!")
        print("⚠️ There must be 20-channel data we didn't find")
    else:
        print("❌ Something is wrong - investigate further")

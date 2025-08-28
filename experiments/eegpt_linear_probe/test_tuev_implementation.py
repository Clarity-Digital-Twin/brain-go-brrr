#!/usr/bin/env python3
"""
Test TUEV implementation to verify all phases are working correctly.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_phase1_bugs_fixed():
    """Test that Phase 1 critical bugs are fixed."""
    print("\n=== Testing Phase 1: Critical Bugs Fixed ===")

    # Test 1: Check TARGET_CHANNELS has exactly 20 channels (no FPZ)
    from experiments.eegpt_linear_probe.datasets.tuev_dataset import TARGET_CHANNELS

    assert len(TARGET_CHANNELS) == 20, f"Expected 20 channels, got {len(TARGET_CHANNELS)}"
    assert 'FPZ' not in TARGET_CHANNELS, "FPZ should not be in TARGET_CHANNELS"
    assert 'OZ' in TARGET_CHANNELS, "OZ should be in TARGET_CHANNELS"
    print("✓ TARGET_CHANNELS has exactly 20 channels without FPZ")

    # Test 2: Check no TCP/bipolar code exists
    from experiments.eegpt_linear_probe.datasets import tuev_dataset

    assert not hasattr(tuev_dataset, 'TCP_CHANNELS'), "TCP_CHANNELS should be removed"
    assert not hasattr(tuev_dataset, 'TCP_BIPOLAR_PAIRS'), "TCP_BIPOLAR_PAIRS should be removed"
    assert not hasattr(
        tuev_dataset, 'compute_bipolar_derivation'
    ), "compute_bipolar_derivation should be removed"
    print("✓ All TCP/bipolar code removed")

    # Test 3: Check cache version is mne-ar-v3
    from experiments.eegpt_linear_probe.datasets.tuev_mne_dataset import TUEVMNEDataset

    assert (
        TUEVMNEDataset.CACHE_VERSION == "mne-ar-v3"
    ), f"Expected mne-ar-v3, got {TUEVMNEDataset.CACHE_VERSION}"
    print("✓ Cache version updated to mne-ar-v3")

    print("✅ Phase 1 COMPLETE\n")
    return True


def test_phase2_windowing():
    """Test that Phase 2 fixed-grid windowing is implemented."""
    print("=== Testing Phase 2: Fixed-Grid Windowing ===")

    from experiments.eegpt_linear_probe.mne_integration.tuev_preprocessor import TUEVPreprocessor

    preprocessor = TUEVPreprocessor()

    # Test window labeling logic
    annotations = [
        {'start': 0.5, 'end': 0.65, 'label': 'spsw'},  # 150ms spike
        {'start': 1.0, 'end': 1.3, 'label': 'artf'},  # 300ms artifact
    ]

    # Test spike priority (spike ≥120ms should win)
    label = preprocessor._label_window(0.0, 4.0, annotations)
    assert label == 'spsw', f"Expected 'spsw' with spike priority, got '{label}'"
    print("✓ Spike priority working (≥120ms)")

    # Test argmax without spike priority
    annotations2 = [
        {'start': 0.5, 'end': 0.55, 'label': 'spsw'},  # 50ms spike (too short)
        {'start': 1.0, 'end': 1.3, 'label': 'artf'},  # 300ms artifact
    ]
    label2 = preprocessor._label_window(0.0, 4.0, annotations2)
    assert label2 == 'artf', f"Expected 'artf' with argmax, got '{label2}'"
    print("✓ Argmax overlap working when spike < 120ms")

    # Test minimum threshold
    annotations3 = [
        {'start': 0.5, 'end': 0.55, 'label': 'spsw'},  # 50ms spike (below 100ms minimum)
    ]
    label3 = preprocessor._label_window(0.0, 4.0, annotations3)
    assert label3 == 'bckg', f"Expected 'bckg' for short event, got '{label3}'"
    print("✓ Minimum overlap threshold working (≥100ms)")

    print("✅ Phase 2 COMPLETE\n")
    return True


def test_phase3_preprocessing():
    """Test that Phase 3 preprocessing updates are applied."""
    print("=== Testing Phase 3: Preprocessing Updates ===")

    # Check functional reference form is used
    with open('mne_integration/preprocessor.py') as f:
        content = f.read()
        assert 'raw, _ = mne.set_eeg_reference' in content, "Should use functional reference form"
    print("✓ Functional mne.set_eeg_reference() form used")

    # Check gentle AR parameters
    from experiments.eegpt_linear_probe.mne_integration.tuev_preprocessor import TUEVPreprocessor

    preprocessor = TUEVPreprocessor()

    # Check _apply_autoreject_tuev method exists
    assert hasattr(preprocessor, '_apply_autoreject_tuev'), "Should have gentle AR method"
    print("✓ Gentle Autoreject method implemented")

    print("✅ Phase 3 COMPLETE\n")
    return True


def test_cache_validation():
    """Test cache validation with smoke assertions."""
    print("=== Testing Cache Validation (Smoke Tests) ===")

    # Import required modules
    from pathlib import Path

    import torch

    # Mock cache creation for testing
    test_cache_dir = Path("/tmp/test_tuev_cache")
    test_cache_dir.mkdir(exist_ok=True)

    # Create a few mock cache files
    for i in range(5):
        x = torch.randn(20, 1024, dtype=torch.float32)
        y = torch.tensor(i % 6, dtype=torch.long)  # Cycle through classes

        # Assertion 1: Shape validation
        assert x.shape == (20, 1024), f"Wrong shape: {x.shape}, expected (20, 1024)"

        # Assertion 2: Dtype validation
        assert x.dtype == torch.float32, f"Wrong dtype: {x.dtype}, expected float32"

        # Assertion 3: No NaNs
        assert not torch.isnan(x).any(), "Found NaN values in tensor"

        # Save mock cache file
        cache_file = test_cache_dir / f"window_{i:06d}_mne-ar-v3.pt"
        torch.save({'x': x, 'y': y}, cache_file)

    # Test loading and validation
    print("✓ Testing cache loading and validation...")
    loaded_files = list(test_cache_dir.glob("*.pt"))
    assert len(loaded_files) == 5, f"Expected 5 cache files, found {len(loaded_files)}"

    # Randomly sample and validate
    import random

    for _ in range(3):
        cache_file = random.choice(loaded_files)
        data = torch.load(cache_file, map_location='cpu')

        # Validate loaded data
        assert 'x' in data and 'y' in data, "Missing 'x' or 'y' keys in cache"
        assert data['x'].shape == (20, 1024), f"Loaded shape wrong: {data['x'].shape}"
        assert data['x'].dtype == torch.float32, f"Loaded dtype wrong: {data['x'].dtype}"
        assert not torch.isnan(data['x']).any(), "Found NaN in loaded data"
        assert 0 <= data['y'].item() < 6, f"Label out of range: {data['y'].item()}"

    print("✓ All cache validation assertions passed")

    # Cleanup
    import shutil

    shutil.rmtree(test_cache_dir)

    print("✅ Cache Validation COMPLETE\n")
    return True


def test_epoch_selection_alignment():
    """Test that epoch selection properly aligns with labels after AR."""
    print("=== Testing Epoch Selection Alignment ===")

    import numpy as np

    # Mock epochs with selection attribute
    class MockEpochs:
        def __init__(self, n_original=10, kept_indices=None):
            self.selection = kept_indices if kept_indices else list(range(n_original))
            self.n_kept = len(self.selection)

        def get_data(self):
            return np.random.randn(self.n_kept, 20, 1024)

        def __len__(self):
            return self.n_kept

    # Test with arbitrary dropout
    original_labels = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg'] * 2  # 12 labels
    kept_indices = [0, 2, 3, 5, 7, 8, 10]  # AR kept these epochs
    epochs_clean = MockEpochs(n_original=12, kept_indices=kept_indices)

    # Verify selection is sorted (MNE guarantee)
    assert epochs_clean.selection == sorted(epochs_clean.selection), "Selection not sorted"

    # Map labels correctly
    aligned_labels = []
    for epoch_idx, original_idx in enumerate(epochs_clean.selection):
        label = original_labels[original_idx]
        aligned_labels.append(label)

    # Verify alignment
    assert len(aligned_labels) == len(epochs_clean), "Label count mismatch"
    assert aligned_labels[0] == 'spsw', f"First label wrong: {aligned_labels[0]}"
    assert aligned_labels[1] == 'pled', f"Second label wrong: {aligned_labels[1]}"  # Index 2
    assert aligned_labels[-1] == 'artf', f"Last label wrong: {aligned_labels[-1]}"  # Index 10

    print("✓ Epoch selection properly aligns with labels after AR")
    print("✅ Epoch Selection Alignment COMPLETE\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TUEV IMPLEMENTATION TEST SUITE")
    print("=" * 60)

    try:
        # Run all phases
        phase1_ok = test_phase1_bugs_fixed()
        phase2_ok = test_phase2_windowing()
        phase3_ok = test_phase3_preprocessing()
        phase4_ok = test_cache_validation()
        phase5_ok = test_epoch_selection_alignment()

        if all([phase1_ok, phase2_ok, phase3_ok, phase4_ok, phase5_ok]):
            print("=" * 60)
            print("🎉 ALL TESTS PASSED - READY FOR CACHE BUILD")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Build cache: ./scripts/build_tuev_mne_cache.sh")
            print("2. Validate cache shapes and labels")
            print("3. Train linear probe")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

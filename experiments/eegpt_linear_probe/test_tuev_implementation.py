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
    assert not hasattr(tuev_dataset, 'compute_bipolar_derivation'), "compute_bipolar_derivation should be removed"
    print("✓ All TCP/bipolar code removed")
    
    # Test 3: Check cache version is mne-ar-v3
    from experiments.eegpt_linear_probe.datasets.tuev_mne_dataset import TUEVMNEDataset
    assert TUEVMNEDataset.CACHE_VERSION == "mne-ar-v3", f"Expected mne-ar-v3, got {TUEVMNEDataset.CACHE_VERSION}"
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
        {'start': 1.0, 'end': 1.3, 'label': 'artf'},   # 300ms artifact
    ]
    
    # Test spike priority (spike ≥120ms should win)
    label = preprocessor._label_window(0.0, 4.0, annotations)
    assert label == 'spsw', f"Expected 'spsw' with spike priority, got '{label}'"
    print("✓ Spike priority working (≥120ms)")
    
    # Test argmax without spike priority
    annotations2 = [
        {'start': 0.5, 'end': 0.55, 'label': 'spsw'},  # 50ms spike (too short)
        {'start': 1.0, 'end': 1.3, 'label': 'artf'},   # 300ms artifact
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
    with open('experiments/eegpt_linear_probe/mne_integration/preprocessor.py') as f:
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

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TUEV IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    try:
        # Run all phases
        phase1_ok = test_phase1_bugs_fixed()
        phase2_ok = test_phase2_windowing()
        phase3_ok = test_phase3_preprocessing()
        
        if all([phase1_ok, phase2_ok, phase3_ok]):
            print("="*60)
            print("🎉 ALL TESTS PASSED - READY FOR CACHE BUILD")
            print("="*60)
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
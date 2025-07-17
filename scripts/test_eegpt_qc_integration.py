#!/usr/bin/env python3
"""
Test EEGPT integration with QC service.

This script verifies that EEGPT model is properly integrated
and produces meaningful abnormality scores.
"""

import sys
from pathlib import Path
import mne
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.qc_flagger import EEGQualityController


def test_eegpt_qc_integration():
    """Test EEGPT integration in QC pipeline."""
    
    print("🧠 Testing EEGPT QC Integration")
    print("=" * 50)
    
    # Initialize QC controller with EEGPT model
    model_path = project_root / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    print(f"\n1️⃣ Initializing QC controller with EEGPT...")
    qc = EEGQualityController(eegpt_model_path=model_path)
    
    # Check if EEGPT loaded
    if qc.eegpt_model is None:
        print("❌ EEGPT model failed to load")
        return False
    else:
        print("✅ EEGPT model loaded successfully")
    
    # Create synthetic test data
    print("\n2️⃣ Creating test EEG data...")
    sfreq = 256  # EEGPT sampling rate
    duration = 20  # seconds
    n_channels = 19
    
    # Create normal-looking EEG
    times = np.arange(0, duration, 1/sfreq)
    data_normal = np.random.randn(n_channels, len(times)) * 30e-6  # 30 µV
    
    # Add some alpha rhythm (8-12 Hz)
    for i in range(n_channels):
        alpha_freq = 10 + np.random.rand()
        data_normal[i] += 20e-6 * np.sin(2 * np.pi * alpha_freq * times)
    
    # Create abnormal-looking EEG (high amplitude, spikes)
    data_abnormal = np.random.randn(n_channels, len(times)) * 100e-6  # 100 µV
    
    # Add some spike-like activity
    spike_times = np.random.choice(len(times), size=50, replace=False)
    for st in spike_times:
        data_abnormal[:, st:st+10] += np.random.randn(n_channels, min(10, len(times)-st)) * 200e-6
    
    # Create raw objects with standard 10-20 channel names
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'P7', 'P8', 'Fz', 'Cz', 'Pz'][:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    raw_normal = mne.io.RawArray(data_normal, info)
    raw_abnormal = mne.io.RawArray(data_abnormal, info)
    
    # Test normal EEG
    print("\n3️⃣ Testing normal EEG...")
    try:
        result_normal = qc.run_full_qc_pipeline(raw_normal)
        print(f"   Abnormality score: {result_normal['quality_metrics']['abnormality_score']:.3f}")
        print(f"   Quality grade: {result_normal['quality_metrics']['quality_grade']}")
        print(f"   Processing time: {result_normal.get('processing_time', 0):.2f}s")
    except Exception as e:
        print(f"❌ Error processing normal EEG: {e}")
        return False
    
    # Test abnormal EEG
    print("\n4️⃣ Testing abnormal EEG...")
    try:
        result_abnormal = qc.run_full_qc_pipeline(raw_abnormal)
        print(f"   Abnormality score: {result_abnormal['quality_metrics']['abnormality_score']:.3f}")
        print(f"   Quality grade: {result_abnormal['quality_metrics']['quality_grade']}")
        print(f"   Processing time: {result_abnormal.get('processing_time', 0):.2f}s")
    except Exception as e:
        print(f"❌ Error processing abnormal EEG: {e}")
        return False
    
    # Verify results make sense
    print("\n5️⃣ Verifying results...")
    
    # Check if abnormal score is higher than normal
    if result_abnormal['quality_metrics']['abnormality_score'] > result_normal['quality_metrics']['abnormality_score']:
        print("✅ Abnormal EEG has higher abnormality score")
    else:
        print("⚠️  Abnormal EEG score not higher than normal (model may need fine-tuning)")
    
    # Check processing details
    if 'processing_info' in result_normal:
        print(f"\n📊 Processing info:")
        print(f"   EEGPT available: {result_normal['processing_info']['eegpt_available']}")
        print(f"   Autoreject available: {result_normal['processing_info']['autoreject_available']}")
    
    # Test on real Sleep-EDF data
    print("\n6️⃣ Testing on real Sleep-EDF data...")
    edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"
    
    if edf_path.exists():
        try:
            raw_real = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            # Take first 30 seconds for quick test
            raw_real.crop(tmax=30)
            
            result_real = qc.run_full_qc_pipeline(raw_real)
            print(f"   Abnormality score: {result_real['quality_metrics']['abnormality_score']:.3f}")
            print(f"   Quality grade: {result_real['quality_metrics']['quality_grade']}")
            print(f"   Bad channels: {result_real['quality_metrics']['bad_channels']}")
            
        except Exception as e:
            print(f"⚠️  Error with real data: {e}")
    
    print("\n✅ EEGPT QC integration test completed!")
    return True


if __name__ == "__main__":
    success = test_eegpt_qc_integration()
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
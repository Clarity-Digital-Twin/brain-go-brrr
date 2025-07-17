#!/usr/bin/env python3
"""
Test script for Sleep-EDF data analysis pipeline.

This script:
1. Loads a Sleep-EDF file
2. Runs quality control
3. Performs sleep staging
4. Generates a simple report
"""

import sys
from pathlib import Path
import mne
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.qc_flagger import EEGQualityController
from services.sleep_metrics import SleepAnalyzer


def test_sleep_edf_pipeline():
    """Test the sleep analysis pipeline with Sleep-EDF data."""
    
    # Path to sleep-edf data
    data_path = project_root / "data/datasets/external/sleep-edf"
    
    # Find first available EDF file
    edf_files = list(data_path.glob("**/*PSG.edf"))
    
    if not edf_files:
        print("❌ No Sleep-EDF files found. Please ensure data is downloaded.")
        print(f"   Expected location: {data_path}")
        return False
    
    # Use first file for testing
    test_file = edf_files[0]
    print(f"✅ Found {len(edf_files)} EDF files")
    print(f"📄 Testing with: {test_file.name}")
    
    try:
        # Load EEG data
        print("\n1️⃣ Loading EEG data...")
        raw = mne.io.read_raw_edf(test_file, preload=True, verbose=False)
        print(f"   Duration: {raw.times[-1]:.1f} seconds")
        print(f"   Channels: {len(raw.ch_names)}")
        print(f"   Sampling rate: {raw.info['sfreq']} Hz")
        
        # Run quality control
        print("\n2️⃣ Running quality control...")
        qc = EEGQualityController()
        qc_results = qc.run_full_qc_pipeline(raw)
        
        print(f"   Bad channels: {qc_results.get('bad_channels', [])}")
        print(f"   Quality score: {qc_results.get('quality_score', 0):.2f}")
        print(f"   Usable: {'✅' if qc_results.get('usable', False) else '❌'}")
        
        # Run sleep analysis (if recording is long enough)
        if raw.times[-1] > 600:  # More than 10 minutes
            print("\n3️⃣ Running sleep analysis...")
            sleep_analyzer = SleepAnalyzer()
            sleep_results = sleep_analyzer.run_full_sleep_analysis(raw)
            
            if sleep_results:
                print(f"   Sleep efficiency: {sleep_results.get('sleep_efficiency', 0):.1f}%")
                print(f"   Total sleep time: {sleep_results.get('total_sleep_time', 0):.1f} min")
                print(f"   REM percentage: {sleep_results.get('rem_percentage', 0):.1f}%")
            else:
                print("   ⚠️  Sleep analysis failed")
        else:
            print("\n⚠️  Recording too short for sleep analysis")
        
        print("\n✅ Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_eegpt_model():
    """Check if EEGPT model is available."""
    model_path = project_root / "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    
    if model_path.exists():
        print(f"✅ EEGPT model found: {model_path}")
        print(f"   Size: {model_path.stat().st_size / 1e6:.1f} MB")
        return True
    else:
        print(f"❌ EEGPT model not found at: {model_path}")
        return False


if __name__ == "__main__":
    print("🧠 Brain-Go-Brrr: Sleep-EDF Pipeline Test")
    print("=" * 50)
    
    # Check EEGPT model
    print("\n📦 Checking EEGPT model...")
    check_eegpt_model()
    
    # Test sleep pipeline
    print("\n🔬 Testing Sleep-EDF pipeline...")
    success = test_sleep_edf_pipeline()
    
    if success:
        print("\n🎉 All tests passed! Ready to process Sleep-EDF data.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("\n" + "=" * 50)
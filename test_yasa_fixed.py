#!/usr/bin/env python
"""Test YASA sleep staging - FIXED version."""

import mne
import numpy as np
from pathlib import Path

print("=== TESTING YASA SLEEP STAGING (FIXED) ===\n")

# Create synthetic sleep-like EEG data
print("1. Creating synthetic EEG data...")
sfreq = 256  # Hz
duration = 300  # 5 minutes
n_channels = 19

# Standard 10-20 channels
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz']

# Create data with sleep-like characteristics
np.random.seed(42)
data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6  # 50 µV

# Add some slow waves for N3 sleep simulation
for i in range(n_channels):
    # Add delta waves (0.5-4 Hz) for deep sleep
    t = np.arange(0, duration, 1/sfreq)
    delta = 100e-6 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz delta wave
    data[i, :] += delta

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
print(f"Created {duration}s of EEG data with {n_channels} channels")

# Test YASA adapter with CORRECT method signature
print("\n2. Testing YASA adapter...")
try:
    from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager
    
    stager = YASASleepStager()
    print("✅ YASA adapter imported successfully")
    
    # Test staging with numpy array (as expected by the method)
    print("\n3. Running sleep staging with numpy array...")
    eeg_array = raw.get_data()  # Get numpy array from Raw
    results = stager.stage_sleep(
        eeg_data=eeg_array,
        sfreq=raw.info['sfreq'],
        ch_names=raw.ch_names
    )
    
    if results:
        print("✅ Sleep staging completed!")
        print(f"   - Hypnogram length: {len(results.get('hypnogram', []))} epochs")
        print(f"   - Sleep stages found: {set(results.get('hypnogram', []))}")
        print(f"   - Sleep efficiency: {results.get('sleep_efficiency', 'N/A')}%")
        
        # Check metrics
        if 'sleep_metrics' in results:
            metrics = results['sleep_metrics']
            print("\nSleep Metrics:")
            for key, value in metrics.items():
                print(f"   - {key}: {value}")
    else:
        print("❌ Sleep staging returned None!")
        
except ImportError as e:
    print(f"❌ Failed to import YASA adapter: {e}")
except Exception as e:
    print(f"❌ Sleep staging failed: {e}")
    import traceback
    traceback.print_exc()

# Test process_sleep_edf method if we have an EDF file
print("\n4. Testing process_sleep_edf method...")
# First, let's create a test EDF file from our data
test_edf_path = Path("test_sleep.edf")
try:
    # Export to EDF
    raw.export(test_edf_path, fmt='edf', overwrite=True)
    print(f"Created test EDF: {test_edf_path}")
    
    # Test the process_sleep_edf method
    results_edf = stager.process_sleep_edf(test_edf_path)
    
    if results_edf:
        print("✅ process_sleep_edf completed!")
        print(f"   - Keys in result: {list(results_edf.keys())}")
    else:
        print("❌ process_sleep_edf returned None!")
        
except Exception as e:
    print(f"❌ process_sleep_edf failed: {e}")
finally:
    # Clean up
    if test_edf_path.exists():
        test_edf_path.unlink()
        print(f"Cleaned up {test_edf_path}")

print("\n=== YASA BEHAVIOR TEST COMPLETE ===")
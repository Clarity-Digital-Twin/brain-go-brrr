#!/usr/bin/env python
"""Test YASA sleep staging actual behavior."""

import mne
import numpy as np
from pathlib import Path

# Test with synthetic data first
print("=== TESTING YASA SLEEP STAGING ===\n")

# Create synthetic sleep-like EEG data
print("1. Creating synthetic EEG data...")
sfreq = 256  # Hz
duration = 300  # 5 minutes
n_channels = 19

# Standard 10-20 channels
ch_names = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]

# Create data with sleep-like characteristics
np.random.seed(42)
data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6  # 50 µV

# Add some slow waves for N3 sleep simulation
for i in range(n_channels):
    # Add delta waves (0.5-4 Hz) for deep sleep
    t = np.arange(0, duration, 1 / sfreq)
    delta = 100e-6 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz delta wave
    data[i, :] += delta

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(data, info)
print(f"Created {duration}s of EEG data with {n_channels} channels")

# Test YASA adapter
print("\n2. Testing YASA adapter...")
try:
    from brain_go_brrr.infra.external.yasa_adapter import YASASleepStager

    stager = YASASleepStager()
    print("✅ YASA adapter imported successfully")

    # Test staging
    print("\n3. Running sleep staging...")
    results = stager.stage_sleep(raw)

    if results:
        print("✅ Sleep staging completed!")
        print(f"   - Hypnogram length: {len(results.get('hypnogram', []))} epochs")
        print(f"   - Sleep stages found: {set(results.get('hypnogram', []))}")
        print(f"   - Sleep efficiency: {results.get('sleep_efficiency', 'N/A')}%")

        # Check if we have all expected keys
        expected_keys = ["hypnogram", "sleep_efficiency", "sleep_stages", "total_sleep_time"]
        for key in expected_keys:
            if key in results:
                print(f"   ✅ {key}: Present")
            else:
                print(f"   ❌ {key}: Missing")
    else:
        print("❌ Sleep staging returned None!")

except ImportError as e:
    print(f"❌ Failed to import YASA adapter: {e}")
except Exception as e:
    print(f"❌ Sleep staging failed: {e}")
    import traceback

    traceback.print_exc()

# Test with real fixture data
print("\n4. Testing with fixture data...")
fixture_path = Path("tests/fixtures/eeg/tuab_001_norm_30s.fif")
if fixture_path.exists():
    try:
        raw_fixture = mne.io.read_raw_fif(fixture_path, preload=True)
        print(f"Loaded fixture: {fixture_path.name}")

        results_fixture = stager.stage_sleep(raw_fixture)
        if results_fixture:
            print("✅ Fixture sleep staging completed!")
            print(f"   - Hypnogram: {results_fixture.get('hypnogram', [])[:10]}...")
        else:
            print("❌ Fixture sleep staging returned None!")
    except Exception as e:
        print(f"❌ Fixture test failed: {e}")
else:
    print(f"⚠️ Fixture not found: {fixture_path}")

print("\n=== YASA BEHAVIOR TEST COMPLETE ===")

#!/usr/bin/env python3
"""Quick API test with mock data to verify pipeline behavior."""

import time
from pathlib import Path

import mne
import numpy as np


def create_test_edf():
    """Create a small test EDF file."""
    # Create mock EEG data - 30 seconds, 19 channels, 256 Hz
    sfreq = 256
    duration = 30  # seconds
    n_channels = 19
    n_times = int(sfreq * duration)

    # Standard 10-20 channel names
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

    # Create data with some artifacts
    data = np.random.randn(n_channels, n_times) * 20e-6  # ~20 μV

    # Add some bad channel behavior to T3 (high amplitude noise)
    data[ch_names.index("T3"), :] = np.random.randn(n_times) * 100e-6

    # Add eye blink artifact
    blink_time = int(10 * sfreq)
    data[:4, blink_time : blink_time + int(0.5 * sfreq)] += 50e-6

    # Create MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # Save as EDF
    output_path = Path(__file__).parent.parent / "test_data.edf"
    raw.export(output_path, fmt="edf", overwrite=True)

    return output_path


def test_api_with_mock_data():
    """Test the API with a small mock EDF file."""
    import requests

    # Create test file
    print("📝 Creating test EDF file...")
    edf_path = create_test_edf()
    print(f"   Created: {edf_path.name} ({edf_path.stat().st_size / 1024:.1f} KB)")

    # Test analyze endpoint
    print("\n🔍 Testing /api/v1/eeg/analyze endpoint...")
    start_time = time.time()

    with open(edf_path, "rb") as f:
        files = {"file": (edf_path.name, f, "application/octet-stream")}
        response = requests.post("http://localhost:8000/api/v1/eeg/analyze", files=files)

    elapsed = time.time() - start_time

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Processing took {elapsed:.2f} seconds")
        print("\n📋 Results:")
        for key, value in data.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")

    # Clean up
    edf_path.unlink()
    print("\n🧹 Cleaned up test file")


def main():
    """Run quick test."""
    # First, start the API server
    print("🚀 Starting API server...")
    import atexit
    import subprocess

    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "api.main:app"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Register cleanup
    def cleanup():
        proc.terminate()
        proc.wait()

    atexit.register(cleanup)

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)

    # Check if server is ready
    import requests

    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Server is ready!")
        else:
            print("❌ Server not responding properly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Run the test
    test_api_with_mock_data()

    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()

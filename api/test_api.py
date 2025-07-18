#!/usr/bin/env python3
"""Test script for Brain-Go-Brrr API.

Tests the QC flagger endpoint with sample EEG data.
"""

import json
import sys
from pathlib import Path

import requests

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_health_check():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get("http://localhost:8000/health")

    if response.status_code == 200:
        data = response.json()
        print("✅ Health check passed")
        print(f"   Status: {data['status']}")
        print(f"   EEGPT loaded: {data['eegpt_loaded']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


def test_qc_analysis():
    """Test QC analysis endpoint."""
    print("\nTesting QC analysis...")

    # Find a test EDF file
    edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"

    if not edf_path.exists():
        print(f"❌ Test file not found: {edf_path}")
        return

    # Upload and analyze
    with open(edf_path, "rb") as f:
        files = {"file": ("test.edf", f, "application/octet-stream")}
        response = requests.post("http://localhost:8000/api/v1/eeg/analyze", files=files)

    if response.status_code == 200:
        data = response.json()
        print("✅ QC analysis completed")
        print("\n📊 Results:")
        print(f"   Status: {data['status']}")
        print(f"   Bad channels: {data['bad_channels']}")
        print(f"   Bad channel %: {data['bad_pct']}%")
        print(f"   Abnormality probability: {data['abnormal_prob']}")
        print(f"   Triage flag: {data['flag']}")
        print(f"   Confidence: {data['confidence']}")
        print(f"   Quality grade: {data['quality_grade']}")
        print(f"   Processing time: {data['processing_time']}s")

        # Pretty print full response
        print("\n📄 Full response:")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ QC analysis failed: {response.status_code}")
        print(f"   Error: {response.text}")


def test_root_endpoint():
    """Test root endpoint."""
    print("\nTesting root endpoint...")
    response = requests.get("http://localhost:8000/")

    if response.status_code == 200:
        data = response.json()
        print("✅ Root endpoint accessible")
        print(f"   Message: {data['message']}")
        print(f"   Version: {data['version']}")
    else:
        print(f"❌ Root endpoint failed: {response.status_code}")


if __name__ == "__main__":
    print("🧠 Brain-Go-Brrr API Test")
    print("=" * 50)
    print("\nMake sure the API is running:")
    print("  uvicorn api.main:app --reload")
    print("\n" + "=" * 50)

    try:
        test_root_endpoint()
        test_health_check()
        test_qc_analysis()

        print("\n✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Is it running?")
        print("   Run: uvicorn api.main:app --reload")

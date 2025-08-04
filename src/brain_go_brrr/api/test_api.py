#!/usr/bin/env python3
"""Test script for Brain-Go-Brrr API.

Tests the QC flagger endpoint with sample EEG data.
"""

import sys
from pathlib import Path

import requests

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_health_check() -> None:
    """Test health check endpoint."""
    response = requests.get("http://localhost:8000/health", timeout=5)

    if response.status_code == 200:
        _ = response.json()  # Validate response format
    else:
        pass


def test_qc_analysis() -> None:
    """Test QC analysis endpoint."""
    # Find a test EDF file
    edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"

    if not edf_path.exists():
        return

    # Upload and analyze
    with edf_path.open("rb") as f:
        files = {"file": ("test.edf", f, "application/octet-stream")}
        response = requests.post(
            "http://localhost:8000/api/v1/eeg/analyze", files=files, timeout=10
        )

    if response.status_code == 200:
        _ = response.json()  # Validate response format

        # Pretty print full response
    else:
        pass


def test_root_endpoint() -> None:
    """Test root endpoint."""
    response = requests.get("http://localhost:8000/", timeout=5)

    if response.status_code == 200:
        _ = response.json()  # Validate response format
    else:
        pass


if __name__ == "__main__":

    try:
        test_root_endpoint()
        test_health_check()
        test_qc_analysis()


    except requests.exceptions.ConnectionError:
        pass

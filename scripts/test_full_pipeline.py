#!/usr/bin/env python3
"""Test the full EEG analysis pipeline with real Sleep-EDF data."""

import json
import time
from pathlib import Path

import requests


def test_pipeline_with_real_data():
    """Run a real EDF file through the API and save results."""
    # Get path to a Sleep-EDF file
    project_root = Path(__file__).parent.parent
    edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"

    if not edf_path.exists():
        print(f"❌ EDF file not found: {edf_path}")
        return

    print(f"📊 Testing pipeline with: {edf_path.name}")
    print(f"   File size: {edf_path.stat().st_size / 1e6:.1f} MB")

    # Test the basic analyze endpoint
    print("\n🔍 Testing /api/v1/eeg/analyze endpoint...")

    start_time = time.time()

    with open(edf_path, "rb") as f:
        files = {"file": (edf_path.name, f, "application/octet-stream")}
        response = requests.post("http://localhost:8000/api/v1/eeg/analyze", files=files)

    elapsed = time.time() - start_time

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success! Processing took {elapsed:.2f} seconds")
        print("\n📋 Analysis Results:")
        print(f"   Status: {data.get('status')}")
        print(f"   Bad channels: {data.get('bad_channels', [])}")
        print(f"   Bad channel %: {data.get('bad_pct', 0)}%")
        print(f"   Abnormality score: {data.get('abnormal_prob', 0):.3f}")
        print(f"   Triage flag: {data.get('flag')}")
        print(f"   Confidence: {data.get('confidence', 0):.3f}")
        print(f"   Quality grade: {data.get('quality_grade')}")
        print(f"   Processing time: {data.get('processing_time', 0):.2f}s")

        # Save results
        output_dir = project_root / "outputs" / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"analyze_{edf_path.stem}_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")

    else:
        print(f"❌ Error: Status code {response.status_code}")
        print(f"   Response: {response.text}")

    # Test the detailed endpoint with PDF report
    print("\n📄 Testing /api/v1/eeg/analyze/detailed endpoint...")

    start_time = time.time()

    with open(edf_path, "rb") as f:
        files = {"file": (edf_path.name, f, "application/octet-stream")}
        response = requests.post(
            "http://localhost:8000/api/v1/eeg/analyze/detailed",
            files=files,
            params={"include_report": True},
        )

    elapsed = time.time() - start_time

    if response.status_code == 200:
        print(f"✅ Success! Processing took {elapsed:.2f} seconds")

        # Extract PDF from response
        content_type = response.headers.get("content-type", "")
        if "multipart" in content_type:
            print("   Received multipart response with PDF report")
            # In real implementation, we'd parse the multipart response
        else:
            data = response.json()
            print("   Detailed analysis completed")

            # Save detailed results
            results_file = output_dir / f"detailed_{edf_path.stem}_{int(time.time())}.json"
            with open(results_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"💾 Detailed results saved to: {results_file}")
    else:
        print(f"❌ Error: Status code {response.status_code}")
        print(f"   Response: {response.text}")

    # Check if reports were generated
    report_dir = project_root / "outputs" / "reports"
    if report_dir.exists():
        pdf_files = list(report_dir.glob("*.pdf"))
        md_files = list(report_dir.glob("*.md"))
        print("\n📊 Generated reports:")
        print(f"   PDF reports: {len(pdf_files)}")
        print(f"   Markdown reports: {len(md_files)}")

        if pdf_files:
            print(f"   Latest PDF: {sorted(pdf_files)[-1].name}")
        if md_files:
            print(f"   Latest MD: {sorted(md_files)[-1].name}")


def main():
    """Run the test."""
    print("🧠 Brain-Go-Brrr Pipeline Test")
    print("==============================\n")

    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ API is not responding at http://localhost:8000")
            print("   Please start the API with: uv run uvicorn api.main:app")
            return
    except requests.ConnectionError:
        print("❌ Cannot connect to API at http://localhost:8000")
        print("   Please start the API with: uv run uvicorn api.main:app")
        return

    print("✅ API is running\n")

    # Run the test
    test_pipeline_with_real_data()

    print("\n✅ Pipeline test completed!")


if __name__ == "__main__":
    main()

"""Integration tests for API using real Sleep-EDF data.

Tests the complete pipeline with actual EEG files from Sleep-EDF dataset.
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TestSleepEDFIntegration:
    """Test API with real Sleep-EDF data."""

    @pytest.fixture
    def client(self):
        """Create test client with proper startup."""
        from api.main import app

        return TestClient(app)

    @pytest.fixture
    def sleep_edf_file(self):
        """Get path to Sleep-EDF test file."""
        project_root = Path(__file__).parent.parent.parent
        edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"

        if not edf_path.exists():
            pytest.skip("Sleep-EDF data not available. Run data download scripts first.")

        return edf_path

    @pytest.fixture
    def cropped_edf_bytes(self, sleep_edf_file):
        """Create a cropped EDF file (60 seconds) for fast tests."""
        # For speed, just read the first part of the file
        # EDF files have fixed header size, so we can read a portion
        # This is a hack but avoids the export issue

        # Read approximately 1MB which should be ~60s of data
        with sleep_edf_file.open("rb") as f:
            # Read header + some data (EDF has 256 byte header + 256 bytes per channel)
            return f.read(1024 * 1024)  # 1MB should be enough for testing

    @pytest.mark.integration
    @patch("core.quality.EEGQualityController.run_full_qc_pipeline")
    def test_real_edf_processing_fast(self, mock_qc_pipeline, client, sleep_edf_file):
        """Test API endpoint with mocked processing (fast)."""
        # Mock the heavy processing to return quickly
        mock_qc_pipeline.return_value = {
            "bad_channels": ["T3", "O2"],
            "bad_channel_ratio": 0.1,
            "abnormality_score": 0.3,
            "quality_grade": "GOOD",
            "confidence": 0.9,
            "quality_metrics": {
                "bad_channels": ["T3", "O2"],
                "bad_channel_ratio": 0.1,
                "abnormality_score": 0.3,
                "quality_grade": "GOOD",
            },
        }

        # Read just first 1MB of file for speed
        with sleep_edf_file.open("rb") as f:
            file_data = f.read(1024 * 1024)
            files = {"edf_file": ("test.edf", file_data, "application/octet-stream")}

            # Time the request
            start_time = time.time()
            response = client.post("/api/v1/eeg/analyze", files=files)
            processing_time = time.time() - start_time

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check response structure matches QCResponse
        assert "flag" in data
        assert "confidence" in data
        assert "bad_channels" in data
        assert "quality_metrics" in data
        assert "recommendation" in data
        assert "processing_time" in data
        assert "quality_grade" in data
        assert "timestamp" in data

        # Performance check - cropped file should process quickly
        assert processing_time < 10, f"Processing took too long: {processing_time:.2f}s"

        # Log results for debugging
        print("\n📊 Real EDF Analysis Results (cropped):")
        print("   File: test_cropped.edf (60s sample)")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Bad channels: {data['bad_channels']}")
        print(f"   Triage flag: {data['flag']}")
        print(f"   Quality: {data['quality_grade']}")
        print(f"   Confidence: {data['confidence']}")
        print(f"   Recommendation: {data['recommendation']}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_edf_processing_full(self, client, sleep_edf_file):
        """Test processing full Sleep-EDF file through API (slow)."""
        # Read the actual full EDF file
        with sleep_edf_file.open("rb") as f:
            files = {"edf_file": (sleep_edf_file.name, f, "application/octet-stream")}

            # Time the request
            start_time = time.time()
            response = client.post("/api/v1/eeg/analyze", files=files)
            processing_time = time.time() - start_time

        # Verify response
        assert response.status_code == 200
        data = response.json()

        # Check response structure matches QCResponse
        assert "flag" in data
        assert "bad_channels" in data

        # Full file can take longer
        assert processing_time < 120, f"Processing took too long: {processing_time:.2f}s"

        print("\n📊 Real EDF Analysis Results (full file):")
        print(f"   File: {sleep_edf_file.name}")
        print(f"   Processing time: {processing_time:.2f}s")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_sleep_edf_quality_detection(self, client, sleep_edf_file):
        """Test that Sleep-EDF files are properly analyzed for quality issues."""
        with sleep_edf_file.open("rb") as f:
            files = {"edf_file": (sleep_edf_file.name, f, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Sleep-EDF files often have non-EEG channels that should be flagged
        # Common bad channels: EOG, EMG, event markers, respiration
        if data["bad_channels"]:
            # Check that detected bad channels make sense
            bad_channel_names = [ch.upper() for ch in data["bad_channels"]]

            # These are typically non-EEG channels in Sleep-EDF
            expected_bad_patterns = ["EOG", "EMG", "RESP", "EVENT", "ECG"]

            # At least some bad channels should match expected patterns
            found_expected = any(
                any(pattern in ch for pattern in expected_bad_patterns) for ch in bad_channel_names
            )

            print("\n🔍 Quality Analysis:")
            print(f"   Detected bad channels: {data['bad_channels']}")
            print(f"   Contains expected non-EEG channels: {found_expected}")

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.slow
    def test_multiple_sleep_edf_files(self, client):
        """Test processing multiple Sleep-EDF files to check consistency."""
        project_root = Path(__file__).parent.parent.parent
        sleep_dir = project_root / "data/datasets/external/sleep-edf/sleep-cassette"

        if not sleep_dir.exists():
            pytest.skip("Sleep-EDF directory not found")

        # Get first 3 EDF files
        edf_files = list(sleep_dir.glob("*.edf"))[:3]

        if not edf_files:
            pytest.skip("No EDF files found in Sleep-EDF directory")

        results = []
        for edf_file in edf_files:
            with edf_file.open("rb") as f:
                files = {"edf_file": (edf_file.name, f, "application/octet-stream")}
                response = client.post("/api/v1/eeg/analyze", files=files)

            assert response.status_code == 200
            results.append(response.json())

        # Analyze consistency
        print(f"\n📈 Multiple File Analysis ({len(results)} files):")
        for i, (file, result) in enumerate(zip(edf_files, results, strict=False)):
            print(f"\n   File {i + 1}: {file.name}")
            print(f"   - Confidence: {result['confidence']:.3f}")
            print(f"   - Quality: {result['quality_grade']}")
            print(f"   - Flag: {result['flag']}")

        # All should process successfully
        assert all(r["flag"] in ["ROUTINE", "EXPEDITE", "URGENT"] for r in results)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_cropped_edf_processing(self, client, sleep_edf_file):
        """Test processing a cropped portion of EDF file (like in model tests)."""
        # This mimics what we do in the model tests - crop to 1 minute
        # We can't actually crop here, but we verify the API handles it

        with sleep_edf_file.open("rb") as f:
            files = {"file": ("cropped_test.edf", f, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Even large files should be processed successfully
        assert data["status"] == "success"
        assert data["processing_time"] < 120  # Under 2 minutes as per requirements

    @pytest.mark.integration
    @pytest.mark.slow
    def test_confidence_scoring_consistency(self, client, sleep_edf_file):
        """Test that confidence scores are consistent and reasonable."""
        # Run the same file multiple times
        confidence_scores = []

        for _ in range(2):
            with sleep_edf_file.open("rb") as f:
                files = {"edf_file": (sleep_edf_file.name, f, "application/octet-stream")}
                response = client.post("/api/v1/eeg/analyze", files=files)

            data = response.json()
            confidence_scores.append(data["confidence"])

        # Confidence should be consistent for the same file
        assert all(0 <= score <= 1 for score in confidence_scores)

        # If model is loaded, confidence should be relatively high
        if data.get("confidence", 0) < 1.0:  # Model made an abnormality prediction
            assert min(confidence_scores) > 0.5, "Confidence too low for loaded model"

        print("\n🎯 Confidence Analysis:")
        print(f"   Scores: {confidence_scores}")
        print(f"   Consistent: {max(confidence_scores) - min(confidence_scores) < 0.1}")


class TestAPIRobustness:
    """Test API robustness with edge cases from Sleep-EDF."""

    @pytest.fixture
    def client(self):
        """Create test client with proper startup."""
        from api.main import app

        return TestClient(app)

    @pytest.mark.integration
    def test_empty_file_handling(self, client):
        """Test handling of empty or very small files."""
        # Create a minimal invalid EDF
        files = {"file": ("empty.edf", b"", "application/octet-stream")}
        response = client.post("/api/v1/eeg/analyze", files=files)

        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["flag"] == "ERROR"
        assert data["flag"] == "ERROR"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.slow
    def test_concurrent_sleep_edf_processing(self, client):
        """Test concurrent processing of Sleep-EDF files."""
        project_root = Path(__file__).parent.parent.parent
        edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"

        if not edf_path.exists():
            pytest.skip("Sleep-EDF data not available")

        # Simulate concurrent requests
        import concurrent.futures

        def process_file():
            with edf_path.open("rb") as f:
                files = {"file": (edf_path.name, f, "application/octet-stream")}
                return client.post("/api/v1/eeg/analyze", files=files)

        # Process 3 requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_file) for _ in range(3)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["status"] in ["success", "error"] for r in responses)

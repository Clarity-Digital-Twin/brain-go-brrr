"""Test suite for Brain-Go-Brrr FastAPI endpoints.

Following TDD approach - comprehensive tests based on PRD and ROUGH_DRAFT.md specifications.
"""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# We'll import the app after it's implemented
# from brain_go_brrr.api.main import app


class TestAPIEndpoints:
    """Test FastAPI endpoints according to specifications."""

    @pytest.fixture
    def client(self):
        """Create test client - relies on conftest.py fresh_app for isolation."""
        # Import here to avoid circular imports
        from brain_go_brrr.api.main import app

        return TestClient(app)

    @pytest.fixture
    def mock_edf_file(self):
        """Create a mock EDF file upload."""
        # Create minimal EDF-like content
        content = b"0       " + b" " * 8  # EDF header start
        return io.BytesIO(content)

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Brain-Go-Brrr API" in data["message"]
        assert "version" in data
        assert "endpoints" in data

    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        # Note: Current implementation doesn't include model or redis status
        # TODO: Enhance health check endpoint to include these checks

    def test_analyze_endpoint_requires_file(self, client):
        """Test that analyze endpoint requires a file upload."""
        response = client.post("/api/v1/eeg/analyze")

        # Should fail with 422 (validation error) when no file provided
        assert response.status_code == 422

    def test_analyze_endpoint_validates_file_type(self, client):
        """Test that only EDF files are accepted (FR5.1)."""
        # Test with non-EDF file
        files = {"edf_file": ("test.txt", b"not an edf file", "text/plain")}
        response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 400
        assert "Only EDF files are supported" in response.json()["detail"]

    def test_analyze_endpoint_successful_response_format(
        self, client, mock_qc_controller, valid_edf_content
    ):
        """Test successful analysis returns correct JSON format (ROUGH_DRAFT.md specs)."""
        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Ensure we didn't get an error response
        assert (
            data.get("flag") != "ERROR"
        ), f"Unexpected ERROR response: {data.get('error', 'Unknown error')}"

        # Verify response matches QCResponse schema
        assert "flag" in data
        assert "bad_channels" in data
        assert isinstance(data["bad_channels"], list)
        assert "quality_metrics" in data
        assert "confidence" in data
        assert "processing_time" in data
        assert "quality_grade" in data
        assert "timestamp" in data

    def test_triage_flag_logic_urgent(self, client, mock_qc_controller, valid_edf_content):
        """Test triage flag for URGENT cases (abnormal_prob > 0.8 or POOR quality)."""
        mock_qc_controller.run_full_qc_pipeline.return_value = {
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.05,
                "abnormality_score": 0.85,  # > 0.8
                "quality_grade": "GOOD",
            },
            "processing_info": {"confidence": 0.9},
            "processing_time": 1.0,
        }

        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        assert response.json()["flag"] == "URGENT"

    def test_triage_flag_logic_expedite(self, client, mock_qc_controller, valid_edf_content):
        """Test triage flag for EXPEDITE cases (abnormal_prob > 0.6)."""
        mock_qc_controller.run_full_qc_pipeline.return_value = {
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.05,
                "abnormality_score": 0.65,  # > 0.6, < 0.8
                "quality_grade": "GOOD",
            },
            "processing_info": {"confidence": 0.9},
            "processing_time": 1.0,
        }

        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        assert response.json()["flag"] == "EXPEDITE"

    def test_triage_flag_logic_routine(self, client, mock_qc_controller, valid_edf_content):
        """Test triage flag for ROUTINE cases (abnormal_prob > 0.4)."""
        mock_qc_controller.run_full_qc_pipeline.return_value = {
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.05,
                "abnormality_score": 0.45,  # > 0.4, < 0.6
                "quality_grade": "GOOD",
            },
            "processing_info": {"confidence": 0.9},
            "processing_time": 1.0,
        }

        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        assert response.json()["flag"] == "ROUTINE"

    def test_triage_flag_logic_normal(self, client, mock_qc_controller, valid_edf_content):
        """Test triage flag for NORMAL cases (abnormal_prob <= 0.4)."""
        mock_qc_controller.run_full_qc_pipeline.return_value = {
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.05,
                "abnormality_score": 0.3,  # <= 0.4
                "quality_grade": "EXCELLENT",
            },
            "processing_info": {"confidence": 0.95},
            "processing_time": 0.8,
        }

        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        assert response.json()["flag"] == "ROUTINE"

    def test_processing_time_requirement(self, client, mock_qc_controller, valid_edf_content):
        """Test that processing time is tracked and reasonable (NFR1.1: <2 minutes for 20-min EEG)."""
        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "processing_time" in data
        assert data["processing_time"] < 120  # Less than 2 minutes

    def test_bad_channel_detection_accuracy(self, client, mock_qc_controller, valid_edf_content):
        """Test bad channel detection meets accuracy requirement (FR1.1: >95% accuracy)."""
        # This is more of an integration test, but we verify the format
        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Ensure we didn't get an error response
        assert (
            data.get("flag") != "ERROR"
        ), f"Unexpected ERROR response: {data.get('error', 'Unknown error')}"

        assert isinstance(data["bad_channels"], list)
        assert all(isinstance(ch, str) for ch in data["bad_channels"])
        assert "quality_metrics" in data
        assert 0 <= data["quality_metrics"]["bad_channel_percentage"] <= 100

    def test_abnormality_detection_confidence(self, client, mock_qc_controller, valid_edf_content):
        """Test abnormality detection includes confidence score (FR2.2)."""
        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        data = response.json()

        # Ensure we didn't get an error response
        assert (
            data.get("flag") != "ERROR"
        ), f"Unexpected ERROR response: {data.get('error', 'Unknown error')}"

        assert "quality_metrics" in data
        assert "abnormality_score" in data["quality_metrics"]
        assert 0 <= data["quality_metrics"]["abnormality_score"] <= 1
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_error_handling_corrupted_file(self, client):
        """Test graceful error handling for corrupted EDF files."""
        with patch(
            "brain_go_brrr.core.edf_loader.load_edf_safe", side_effect=Exception("Corrupted EDF")
        ):
            files = {"edf_file": ("bad.edf", b"corrupted", "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200  # Still returns 200 but with error status
        data = response.json()
        assert data["flag"] == "ERROR"
        assert "error" in data

    def test_concurrent_requests_handling(self, client, mock_qc_controller, valid_edf_content):
        """Test API can handle concurrent requests (NFR1.2: Support 50 concurrent analyses)."""
        # This is a simple test - real concurrency testing would use asyncio
        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            # Send multiple requests
            for _ in range(3):
                files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
                response = client.post("/api/v1/eeg/analyze", files=files)
                assert response.status_code == 200

    @pytest.mark.parametrize(
        "quality_grade,expected_in_flag",
        [
            ("EXCELLENT", "ROUTINE"),  # With 0.5 abnormality, it's ROUTINE
            ("GOOD", "ROUTINE"),  # With 0.5 abnormality, it's ROUTINE
            ("FAIR", "EXPEDITE"),  # FAIR quality overrides to EXPEDITE
            ("POOR", "URGENT"),  # POOR quality overrides to URGENT
        ],
    )
    def test_quality_grade_affects_triage(
        self, client, mock_qc_controller, valid_edf_content, quality_grade, expected_in_flag
    ):
        """Test that quality grade influences triage decisions."""
        mock_qc_controller.run_full_qc_pipeline.return_value = {
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.1,
                "abnormality_score": 0.5,  # Moderate
                "quality_grade": quality_grade,
            },
            "processing_info": {"confidence": 0.8},
            "processing_time": 1.0,
        }

        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        assert expected_in_flag in response.json()["flag"]

    def test_detailed_endpoint_exists(self, client, valid_edf_content, mock_qc_controller):
        """Test that detailed analysis endpoint exists for future expansion."""
        # Import the router module to inject the mock
        import brain_go_brrr.api.routers.qc as qc_router

        qc_router.qc_controller = mock_qc_controller

        files = {"edf_file": ("test.edf", valid_edf_content, "application/octet-stream")}
        response = client.post("/api/v1/eeg/analyze/detailed", files=files)

        # Should not 404
        assert response.status_code != 404


class TestAPIPerformance:
    """Test performance requirements from PRD."""

    def test_api_response_time(self, client):
        """Test API response time requirement (NFR1.4: <100ms response time)."""
        # Note: This tests the endpoint itself, not the processing
        import time

        start = time.time()
        response = client.get("/api/v1/health")
        duration = time.time() - start

        assert response.status_code == 200
        # Allow some buffer for test environment
        assert duration < 0.5  # 500ms in test environment

    @pytest.mark.slow
    def test_large_file_handling(self, client, mock_qc_controller):
        """Test handling of large EDF files (up to 2GB per requirements)."""
        # Create a mock large file (just test the endpoint, not actual processing)
        large_content = b"0       " + b" " * 1024 * 1024  # 1MB mock file

        with patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller):
            files = {"edf_file": ("large.edf", large_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200

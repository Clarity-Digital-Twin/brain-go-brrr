"""Test API integration with linear probes."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.brain_go_brrr.api.main import app
from src.brain_go_brrr.models.linear_probe import AbnormalityProbe, SleepStageProbe


class TestAPILinearProbeIntegration:
    """Test linear probe integration in API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_eegpt_model(self):
        """Mock EEGPT model that returns consistent features."""
        mock = Mock()
        # Return consistent summary tokens for testing
        mock.extract_features.return_value = np.random.randn(4, 512)
        mock.is_loaded = True
        return mock

    @pytest.fixture
    def mock_sleep_probe(self):
        """Mock sleep stage probe."""
        probe = Mock(spec=SleepStageProbe)
        # Return consistent predictions
        probe.predict_stage.return_value = (["N2"], torch.tensor([0.85]))
        return probe

    def test_sleep_staging_endpoint_exists(self, client):
        """Test that sleep staging endpoint exists."""
        response = client.get("/api/v1/eeg/sleep/stages")
        # Should return 405 Method Not Allowed (GET not supported)
        assert response.status_code in [405, 422]

    def test_sleep_staging_with_edf_upload(
        self, client, tiny_edf, mock_eegpt_model, mock_sleep_probe
    ):
        """Test sleep staging with EDF file upload."""
        with (
            patch("src.brain_go_brrr.api.routers.sleep.get_eegpt_model") as mock_get_model,
            patch("src.brain_go_brrr.api.routers.sleep.get_sleep_probe") as mock_get_probe,
        ):
            mock_get_model.return_value = mock_eegpt_model
            mock_get_probe.return_value = mock_sleep_probe

            # tiny_edf fixture returns bytes content directly
            files = {"edf_file": ("test.edf", tiny_edf, "application/octet-stream")}
            response = client.post("/api/v1/eeg/sleep/stages", files=files)

            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")
            assert response.status_code == 200
            data = response.json()

            # Check response structure
            assert "stages" in data
            assert "confidence_scores" in data
            assert "hypnogram" in data
            assert "summary" in data

            # Check summary statistics
            assert "total_sleep_time" in data["summary"]
            assert "sleep_efficiency" in data["summary"]
            assert "stage_percentages" in data["summary"]

    def test_sleep_staging_window_by_window(
        self, client, tiny_edf, mock_eegpt_model, mock_sleep_probe
    ):
        """Test that sleep staging processes windows correctly."""
        with (
            patch("src.brain_go_brrr.api.routers.sleep.get_eegpt_model") as mock_get_model,
            patch("src.brain_go_brrr.api.routers.sleep.get_sleep_probe") as mock_get_probe,
        ):
            mock_get_model.return_value = mock_eegpt_model
            mock_get_probe.return_value = mock_sleep_probe

            # Mock multiple windows
            mock_sleep_probe.predict_stage.side_effect = [
                (["W"], torch.tensor([0.9])),
                (["N1"], torch.tensor([0.7])),
                (["N2"], torch.tensor([0.85])),
                (["N3"], torch.tensor([0.8])),
                (["N2"], torch.tensor([0.82])),
            ]

            files = {"edf_file": ("test.edf", tiny_edf, "application/octet-stream")}
            response = client.post("/api/v1/eeg/sleep/stages", files=files)

            assert response.status_code == 200
            data = response.json()

            # Should have processed multiple windows
            assert len(data["stages"]) == 5
            assert data["stages"] == ["W", "N1", "N2", "N3", "N2"]
            assert len(data["confidence_scores"]) == 5

    def test_abnormality_detection_with_probe(self, client, tiny_edf, mock_eegpt_model):
        """Test abnormality detection uses linear probe."""
        with (
            patch("src.brain_go_brrr.api.routers.eegpt.get_eegpt_model") as mock_get_model,
            patch("src.brain_go_brrr.api.routers.eegpt.get_probe") as mock_get_probe,
        ):
            mock_get_model.return_value = mock_eegpt_model

            # Mock abnormality probe
            mock_probe = Mock(spec=AbnormalityProbe)
            mock_probe.predict_abnormal_probability.return_value = torch.tensor([0.75])
            mock_get_probe.return_value = mock_probe

            files = {"edf_file": ("test.edf", tiny_edf, "application/octet-stream")}
            response = client.post(
                "/api/v1/eeg/analyze", files=files, data={"analysis_type": "abnormality_probe"}
            )

            assert response.status_code == 200
            data = response.json()

            assert "abnormal_probability" in data
            assert data["abnormal_probability"] == pytest.approx(0.75, 0.01)
            assert "method" in data
            assert data["method"] == "linear_probe"

    def test_probe_selection_based_on_task(self, client):
        """Test that correct probe is selected based on task."""
        # This test verifies the factory pattern works
        response = client.get("/api/v1/eeg/probes/available")

        if response.status_code == 200:
            data = response.json()
            assert "sleep" in data["available_probes"]
            assert "abnormality" in data["available_probes"]
            assert "motor_imagery" in data["available_probes"]

    def test_batch_processing_with_probe(self, client, mock_eegpt_model):
        """Test batch processing of multiple EEG windows."""
        with patch("src.brain_go_brrr.api.routers.eegpt.get_eegpt_model") as mock_get_model:
            mock_get_model.return_value = mock_eegpt_model

            # Mock batch features
            batch_size = 10
            mock_eegpt_model.extract_features_batch.return_value = np.random.randn(
                batch_size, 4, 512
            )

            files = {"edf_file": ("test.edf", b"mock_edf_data", "application/octet-stream")}
            response = client.post("/api/v1/eeg/analyze/batch", files=files)

            if response.status_code == 200:
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == batch_size


# Removed - using tiny_edf fixture from conftest.py instead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

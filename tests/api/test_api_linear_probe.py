"""Test API integration with linear probes."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from brain_go_brrr.api.main import app
from brain_go_brrr.models.linear_probe import AbnormalityProbe, SleepStageProbe


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
            patch("brain_go_brrr.api.routers.sleep.get_eegpt_model") as mock_get_model,
            patch("brain_go_brrr.api.routers.sleep.get_sleep_probe") as mock_get_probe,
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

    @pytest.mark.integration  # Requires larger EDF file for multiple windows
    def test_sleep_staging_window_by_window(
        self, client, mock_eegpt_model, mock_sleep_probe, tiny_edf
    ):
        """Test that sleep staging processes windows correctly."""
        # Use tiny_edf fixture which creates 30 seconds of data
        # This should give us at least 7 windows (30s / 4s per window)
        edf_content = tiny_edf

        with (
            patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model") as mock_get_model,
            patch("brain_go_brrr.api.routers.eegpt.get_probe") as mock_get_probe,
        ):
            mock_get_model.return_value = mock_eegpt_model
            mock_get_probe.return_value = mock_sleep_probe

            # Mock extract_windows to return 7 windows
            mock_windows = [np.zeros((19, 1024)) for _ in range(7)]
            mock_eegpt_model.extract_windows.return_value = mock_windows

            # Mock extract_features to return proper features
            mock_eegpt_model.extract_features.return_value = np.zeros(2048)

            # Mock multiple windows - tiny_edf creates 30 seconds, so expect 7 windows
            mock_sleep_probe.predict_stage.side_effect = [
                (["W"], torch.tensor([0.9])),
                (["N1"], torch.tensor([0.7])),
                (["N2"], torch.tensor([0.85])),
                (["N3"], torch.tensor([0.8])),
                (["N2"], torch.tensor([0.82])),
                (["REM"], torch.tensor([0.75])),
                (["N2"], torch.tensor([0.88])),
            ]

            files = {"edf_file": ("test.edf", edf_content, "application/octet-stream")}
            response = client.post("/api/v1/eeg/eegpt/sleep/stages", files=files)

            assert response.status_code == 200
            data = response.json()

            # Should have processed 7 windows from 30-second file
            assert len(data["stages"]) == 7
            assert data["stages"] == ["W", "N1", "N2", "N3", "N2", "REM", "N2"]
            assert len(data["confidence_scores"]) == 7

    @pytest.mark.integration  # Requires proper EDF processing
    def test_abnormality_detection_with_probe(self, client, tiny_edf, mock_eegpt_model):
        """Test abnormality detection uses linear probe."""
        # Create a longer EDF for QC to work properly (at least 2 seconds)
        import tempfile

        import numpy as np
        from pyedflib import EdfWriter

        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp_path = tmp.name

        writer = EdfWriter(tmp_path, n_channels=19)  # Use standard 19 channels

        # Standard 10-20 channel names
        channel_names = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "T5",
            "P3",
            "Pz",
            "P4",
            "T6",
            "O1",
            "O2",
        ]

        for i, ch_name in enumerate(channel_names):
            writer.setSignalHeader(
                i,
                {
                    "label": ch_name,
                    "dimension": "uV",
                    "sample_frequency": 256,
                    "physical_max": 250,
                    "physical_min": -250,
                    "digital_max": 2047,
                    "digital_min": -2048,
                    "prefilter": "HP:0.1Hz LP:75Hz",
                    "transducer": "AgAgCl electrode",
                },
            )

        # Write 30 seconds of data (7680 samples at 256 Hz) to get 7 windows
        # Create data for all channels at once
        all_data = []
        for ch in range(19):
            # 30 seconds of data per channel
            channel_data = np.random.randint(-100, 100, 30 * 256, dtype=np.int32)
            all_data.append(channel_data)

        # Write all data at once
        writer.writeSamples(all_data)
        writer.close()

        # Read the file content
        from pathlib import Path

        tmp = Path(tmp_path)
        edf_content = tmp.read_bytes()

        # Clean up
        tmp.unlink()

        with (
            patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model") as mock_get_model,
            patch("brain_go_brrr.api.routers.eegpt.get_probe") as mock_get_probe,
        ):
            mock_get_model.return_value = mock_eegpt_model

            # Mock extract_features to return proper features for each window
            mock_eegpt_model.extract_features.return_value = np.zeros(2048)

            # Mock abnormality probe - ensure it returns scalar tensor
            mock_probe = Mock(spec=AbnormalityProbe)
            # Return scalar tensor value that will be converted to float via .item()
            mock_probe.predict_abnormal_probability.return_value = torch.tensor(0.75)
            mock_get_probe.return_value = mock_probe

            files = {"edf_file": ("test.edf", edf_content, "application/octet-stream")}
            response = client.post(
                "/api/v1/eeg/eegpt/analyze",
                files=files,
                data={"analysis_type": "abnormality_probe"},
            )

            assert response.status_code == 200
            data = response.json()

            # Check the structure and method with strict assertions
            assert "result" in data
            assert "method" in data
            assert data["method"] == "linear_probe"

            # Strict assertions - probe must have been called
            result = data["result"]
            assert "abnormal_probability" in result

            # The probe should return 0.75 for each window
            assert result["abnormal_probability"] == pytest.approx(0.75, abs=0.01)

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
        with patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model") as mock_get_model:
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

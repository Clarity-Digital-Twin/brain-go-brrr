"""CLEAN tests for EEGPT router - minimal mocking, test real logic."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from brain_go_brrr.api.routers.eegpt import (
    get_eegpt_model,
    get_probe,
    router,
)


class TestEEGPTRouterClean:
    """Test EEGPT router with minimal mocking - focus on REAL logic."""

    @pytest.fixture(autouse=True)
    def clear_globals(self):
        """Clear global state between tests."""
        from brain_go_brrr.api.routers.eegpt import _reset_state_for_tests

        _reset_state_for_tests()
        yield
        _reset_state_for_tests()

    @pytest.fixture
    def mock_eegpt_model(self):
        """Create minimal mock of EEGPT model with real-like behavior."""
        model = MagicMock()
        model.is_loaded = True

        # Extract features returns realistic feature vector
        def extract_features(data, channels):
            # Return fixed 2048-dim feature vector for deterministic tests
            return np.ones(2048).astype(np.float32) * 0.1

        def extract_windows(data, sfreq):
            # Return realistic windows (4-second windows)
            window_samples = int(4.0 * sfreq)
            n_windows = data.shape[1] // window_samples
            windows = []
            for i in range(n_windows):
                start = i * window_samples
                end = start + window_samples
                windows.append(data[:, start:end])
            return windows

        def extract_features_batch(batch_array, channels):
            # Return fixed batch of feature vectors for deterministic tests
            batch_size = batch_array.shape[0]
            return np.ones((batch_size, 2048)).astype(np.float32) * 0.1

        model.extract_features = extract_features
        model.extract_windows = extract_windows
        model.extract_features_batch = extract_features_batch

        return model

    @pytest.fixture
    def mock_abnormality_probe(self):
        """Create minimal mock of abnormality probe."""
        from brain_go_brrr.models.linear_probe import AbnormalityProbe

        probe = MagicMock(spec=AbnormalityProbe)

        def predict_abnormal_probability(features_tensor):
            # Return fixed probability for deterministic tests
            return torch.tensor(0.42)

        probe.predict_abnormal_probability = predict_abnormal_probability
        return probe

    @pytest.fixture
    def mock_sleep_probe(self):
        """Create minimal mock of sleep probe."""
        from brain_go_brrr.models.linear_probe import SleepStageProbe

        probe = MagicMock(spec=SleepStageProbe)

        def predict_stage(features_tensor):
            # Return fixed sleep stage and confidence for deterministic tests
            stages = [2]  # N2 sleep
            confidences = torch.tensor([0.8])
            return stages, confidences

        probe.predict_stage = predict_stage
        return probe

    @pytest.fixture
    def valid_edf_bytes(self):
        """Create valid EDF file bytes."""
        from pyedflib import EdfWriter

        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            writer = EdfWriter(str(tmp.name), n_channels=2)

            # Set up 2 EEG channels
            for i in range(2):
                writer.setSignalHeader(
                    i,
                    {
                        "label": f"EEG{i + 1}",
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

            # Write 20 seconds of deterministic sine wave data (5 windows of 4 seconds)
            t = np.arange(20 * 256) / 256.0
            # 10 Hz sine wave, realistic amplitude
            signal = 50 * np.sin(2 * np.pi * 10 * t)  # pyedflib needs float64
            data = np.vstack([signal, signal * 0.9]).astype(
                np.float64
            )  # 2 channels with slight difference
            for i in range(2):
                writer.writePhysicalSamples(data[i])

            writer.close()

            # Read the file bytes
            tmp_path = Path(tmp.name)
            content = tmp_path.read_bytes()
            tmp_path.unlink()

            return content

    @pytest.fixture
    def test_client(self):
        """Create test client with router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_get_eegpt_model_singleton_pattern(self):
        """Test get_eegpt_model returns singleton instance."""
        with patch("brain_go_brrr.api.routers.eegpt.EEGPTModel") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            # First call creates instance
            model1 = get_eegpt_model()
            assert model1 == mock_instance
            mock_cls.assert_called_once()

            # Second call returns same instance
            model2 = get_eegpt_model()
            assert model2 == model1
            mock_cls.assert_called_once()  # Still only called once

    def test_get_probe_caches_probes(self):
        """Test get_probe caches probe instances."""
        with patch("brain_go_brrr.api.routers.eegpt.create_probe_for_task") as mock_create:
            mock_probe = MagicMock()
            mock_create.return_value = mock_probe

            # First call creates probe
            probe1 = get_probe("abnormality")
            assert probe1 == mock_probe
            mock_create.assert_called_once_with("abnormality")

            # Second call returns cached probe
            probe2 = get_probe("abnormality")
            assert probe2 == probe1
            mock_create.assert_called_once()  # Still only called once

            # Different task creates new probe
            probe3 = get_probe("sleep")
            assert probe3 == mock_probe  # Same mock but different cache entry
            assert mock_create.call_count == 2

    @patch("brain_go_brrr.api.routers.eegpt.load_edf_safe")
    @patch("brain_go_brrr.api.routers.eegpt.get_probe")
    @patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model")
    def test_analyze_with_probe_abnormality(
        self,
        mock_get_model,
        mock_get_probe,
        mock_load_edf,
        mock_eegpt_model,
        mock_abnormality_probe,
        valid_edf_bytes,
        test_client,
    ):
        """Test analyze_with_probe endpoint for abnormality detection."""
        # Setup mocks
        mock_get_model.return_value = mock_eegpt_model
        mock_get_probe.return_value = mock_abnormality_probe

        # Create mock raw object with deterministic sine wave
        sfreq = 256
        duration = 20  # seconds
        t = np.arange(duration * sfreq) / sfreq
        # 12 Hz sine wave with small DC offset, realistic EEG amplitude
        signal = 50e-6 * (0.1 + 0.9 * np.sin(2 * np.pi * 12 * t))

        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.vstack([signal] * 19)  # 19 channels
        mock_raw.ch_names = [f"EEG{i + 1}" for i in range(19)]
        mock_raw.info = {"sfreq": sfreq}
        mock_load_edf.return_value = mock_raw

        # Make request
        response = test_client.post(
            "/eeg/eegpt/analyze",
            files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
            data={"analysis_type": "abnormality_probe"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["analysis_type"] == "abnormality_probe"
        assert "result" in data
        assert "confidence" in data
        assert data["method"] == "linear_probe"
        assert "metadata" in data

        # Verify result structure
        result = data["result"]
        assert "abnormal_probability" in result
        assert "window_scores" in result
        assert "n_windows" in result
        assert result["n_windows"] == 5  # 20 seconds / 4 seconds per window

        # Verify metadata
        metadata = data["metadata"]
        assert metadata["n_channels"] == 19
        assert metadata["sampling_rate"] == 256
        assert metadata["model"] == "eegpt_10m"

    @patch("brain_go_brrr.api.routers.eegpt.load_edf_safe")
    @patch("brain_go_brrr.api.routers.eegpt.get_probe")
    @patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model")
    def test_analyze_with_probe_sleep(
        self,
        mock_get_model,
        mock_get_probe,
        mock_load_edf,
        mock_eegpt_model,
        mock_sleep_probe,
        valid_edf_bytes,
        test_client,
    ):
        """Test analyze_with_probe endpoint for sleep staging."""
        # Setup mocks
        mock_get_model.return_value = mock_eegpt_model
        mock_get_probe.return_value = mock_sleep_probe

        # Create mock raw object with deterministic sine wave
        sfreq = 256
        duration = 20  # seconds
        t = np.arange(duration * sfreq) / sfreq
        # 12 Hz sine wave with small DC offset, realistic EEG amplitude
        signal = 50e-6 * (0.1 + 0.9 * np.sin(2 * np.pi * 12 * t))

        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.vstack([signal] * 19)  # 19 channels
        mock_raw.ch_names = [f"EEG{i + 1}" for i in range(19)]
        mock_raw.info = {"sfreq": sfreq}
        mock_load_edf.return_value = mock_raw

        # Make request - analysis_type as query param
        response = test_client.post(
            "/eeg/eegpt/analyze?analysis_type=sleep_probe",
            files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response
        assert data["analysis_type"] == "sleep_probe"
        result = data["result"]
        assert "stages" in result
        assert "confidence_scores" in result
        assert result["n_windows"] == 5

    def test_analyze_with_probe_invalid_file(self, test_client):
        """Test analyze_with_probe rejects non-EDF files."""
        response = test_client.post(
            "/eeg/eegpt/analyze",
            files={"edf_file": ("test.txt", b"not an edf", "text/plain")},
            data={"analysis_type": "abnormality_probe"},
        )

        assert response.status_code == 400
        assert "Only EDF files are supported" in response.json()["detail"]

    def test_analyze_with_probe_small_file(self, test_client):
        """Test analyze_with_probe rejects too small files."""
        response = test_client.post(
            "/eeg/eegpt/analyze",
            files={"edf_file": ("test.edf", b"tiny", "application/octet-stream")},
            data={"analysis_type": "abnormality_probe"},
        )

        assert response.status_code == 400
        assert "File too small" in response.json()["detail"]

    @patch("brain_go_brrr.api.routers.eegpt.load_edf_safe")
    def test_analyze_with_probe_edf_load_error(self, mock_load_edf, valid_edf_bytes, test_client):
        """Test analyze_with_probe handles EDF load errors."""
        from brain_go_brrr.core.exceptions import EdfLoadError

        mock_load_edf.side_effect = EdfLoadError("Corrupted EDF")

        response = test_client.post(
            "/eeg/eegpt/analyze",
            files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
            data={"analysis_type": "abnormality_probe"},
        )

        assert response.status_code == 400
        assert "Failed to load EDF" in response.json()["detail"]

    def test_get_available_probes(self, test_client):
        """Test get_available_probes endpoint."""
        response = test_client.get("/eeg/eegpt/probes/available")

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "available_probes" in data
        assert "probe_info" in data

        # Verify content
        probes = data["available_probes"]
        assert "sleep" in probes
        assert "abnormality" in probes
        assert "motor_imagery" in probes

        # Verify probe info
        probe_info = data["probe_info"]
        assert probe_info["sleep"]["num_classes"] == 5
        assert probe_info["abnormality"]["num_classes"] == 2
        assert probe_info["motor_imagery"]["num_classes"] == 4

    @patch("brain_go_brrr.api.routers.eegpt.load_edf_safe")
    @patch("brain_go_brrr.api.routers.eegpt.get_probe")
    @patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model")
    def test_analyze_sleep_stages(
        self,
        mock_get_model,
        mock_get_probe,
        mock_load_edf,
        mock_eegpt_model,
        mock_sleep_probe,
        valid_edf_bytes,
        test_client,
    ):
        """Test analyze_sleep_stages endpoint."""
        # Setup mocks
        mock_get_model.return_value = mock_eegpt_model
        mock_get_probe.return_value = mock_sleep_probe

        # Create mock raw object with deterministic sine wave
        sfreq = 256
        duration = 20  # seconds
        t = np.arange(duration * sfreq) / sfreq
        # 12 Hz sine wave with small DC offset, realistic EEG amplitude
        signal = 50e-6 * (0.1 + 0.9 * np.sin(2 * np.pi * 12 * t))

        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.vstack([signal] * 19)  # 19 channels
        mock_raw.ch_names = [f"EEG{i + 1}" for i in range(19)]
        mock_raw.info = {"sfreq": sfreq}
        mock_load_edf.return_value = mock_raw

        # Make request
        response = test_client.post(
            "/eeg/eegpt/sleep/stages",
            files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response
        assert "stages" in data
        assert "confidence_scores" in data
        assert "total_windows" in data
        assert data["total_windows"] == 5
        assert data["sampling_rate"] == 256

    @patch("brain_go_brrr.api.routers.eegpt.load_edf_safe")
    @patch("brain_go_brrr.api.routers.eegpt.get_probe")
    @patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model")
    def test_analyze_batch(
        self,
        mock_get_model,
        mock_get_probe,
        mock_load_edf,
        mock_eegpt_model,
        valid_edf_bytes,
        test_client,
    ):
        """Test analyze_batch endpoint."""
        # Setup mocks
        mock_get_model.return_value = mock_eegpt_model

        mock_probe = MagicMock()
        mock_probe.predict_proba = MagicMock(return_value=torch.tensor([[0.7, 0.3], [0.6, 0.4]]))
        mock_get_probe.return_value = mock_probe

        # Create mock raw object with more data for batching
        sfreq = 256
        duration = 40  # seconds for more windows
        t = np.arange(duration * sfreq) / sfreq
        # 12 Hz sine wave with small DC offset, realistic EEG amplitude
        signal = 50e-6 * (0.1 + 0.9 * np.sin(2 * np.pi * 12 * t))

        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.vstack([signal] * 19)  # 19 channels
        mock_raw.ch_names = [f"EEG{i + 1}" for i in range(19)]
        mock_raw.info = {"sfreq": sfreq}
        mock_load_edf.return_value = mock_raw

        # Make request - batch_size is a query param, not form data
        response = test_client.post(
            "/eeg/eegpt/analyze/batch?batch_size=2",
            files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
            data={"analysis_type": "abnormality"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response
        assert data["analysis_type"] == "abnormality"
        assert "results" in data
        assert data["total_windows"] == 10  # 40 seconds / 4 seconds per window
        assert data["batch_size"] == 2

    @patch("brain_go_brrr.api.routers.eegpt.load_edf_safe")
    @patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model")
    def test_analyze_batch_fallback_probe(
        self, mock_get_model, mock_load_edf, mock_eegpt_model, valid_edf_bytes, test_client
    ):
        """Test analyze_batch with probe that doesn't have predict_proba."""
        # Setup mocks
        mock_get_model.return_value = mock_eegpt_model

        # Create mock raw object with deterministic sine wave
        sfreq = 256
        duration = 20  # seconds
        t = np.arange(duration * sfreq) / sfreq
        # 12 Hz sine wave with small DC offset, realistic EEG amplitude
        signal = 50e-6 * (0.1 + 0.9 * np.sin(2 * np.pi * 12 * t))

        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.vstack([signal] * 19)  # 19 channels
        mock_raw.ch_names = [f"EEG{i + 1}" for i in range(19)]
        mock_raw.info = {"sfreq": sfreq}
        mock_load_edf.return_value = mock_raw

        with patch("brain_go_brrr.api.routers.eegpt.get_probe") as mock_get_probe:
            # Probe without predict_proba
            mock_probe = MagicMock(spec=[])  # No methods
            mock_get_probe.return_value = mock_probe

            # Make request - batch_size as query param
            response = test_client.post(
                "/eeg/eegpt/analyze/batch?batch_size=2",
                files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
                data={"analysis_type": "custom"},
            )

            assert response.status_code == 200
            data = response.json()

            # Should use fallback [0.5, 0.5] predictions
            assert data["results"][0] == [0.5, 0.5]

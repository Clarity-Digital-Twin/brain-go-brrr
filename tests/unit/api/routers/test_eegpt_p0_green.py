"""GREEN PHASE: Verify that router calls extract_features WITH summary=False (the fix).

After applying P0 fixes, these tests should PASS, confirming that routers now
correctly pass summary=False and flatten to 2048-d features.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pyedflib import EdfWriter


class TestP0GreenPhase:
    """Verify the fixes work correctly."""

    @pytest.fixture
    def test_client(self):
        """Create test client with router."""
        from fastapi import FastAPI

        from brain_go_brrr.api.routers.eegpt import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def valid_edf_bytes(self):
        """Create valid EDF file bytes."""
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            writer = EdfWriter(str(tmp.name), n_channels=19)

            for i in range(19):
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

            t = np.arange(20 * 256) / 256.0
            signal = 50 * np.sin(2 * np.pi * 10 * t)
            data = np.vstack([signal] * 19).astype(np.float64)
            for i in range(19):
                writer.writePhysicalSamples(data[i])

            writer.close()

            tmp_path = Path(tmp.name)
            content = tmp_path.read_bytes()
            tmp_path.unlink()

            return content

    def test_analyze_endpoint_passes_summary_false(self, test_client, valid_edf_bytes):
        """GREEN TEST: /analyze endpoint MUST pass summary=False (after fix).

        This test should PASS after the fix is applied.
        """

        # Mock the load_edf_safe to avoid EDF parsing
        class _StubRaw:
            def __init__(self):
                self.ch_names = [f"EEG{i}" for i in range(19)]
                self.info = {"sfreq": 256}

            def get_data(self):
                return np.random.randn(19, 5120).astype(np.float32)  # 20 seconds

        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", return_value=_StubRaw()):
            # Mock the model to track calls
            mock_model = MagicMock()

            # Track what parameters are passed to extract_features
            extract_features_calls = []

            def track_extract_features(*args, **kwargs):
                extract_features_calls.append((args, kwargs))
                # Return 2048-d features when summary=False
                summary = kwargs.get('summary', True)
                if summary:
                    return np.ones(512).astype(np.float32)
                else:
                    return np.ones((4, 512)).astype(np.float32)  # Will be flattened

            mock_model.extract_features = track_extract_features
            mock_model.extract_windows = lambda data, sfreq: [data[:, :1024] for _ in range(5)]

            # Mock probe that expects 2048-d features
            from brain_go_brrr.infra.ml_models.linear_probe import AbnormalityProbe

            mock_probe = MagicMock(spec=AbnormalityProbe)

            def check_2048_dims(features_tensor):
                assert features_tensor.shape[-1] == 2048, (
                    f"Expected 2048 dims, got {features_tensor.shape[-1]}"
                )
                return torch.tensor(0.5)

            mock_probe.predict_abnormal_probability = check_2048_dims

            with (
                patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model),
                patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe),
            ):
                response = test_client.post(
                    "/eeg/eegpt/analyze",
                    files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
                    data={"analysis_type": "abnormality_probe"},
                )

                assert response.status_code == 200, f"Request failed: {response.text}"

                # Check that extract_features was called with summary=False
                assert len(extract_features_calls) > 0, "extract_features was not called"

                for _args, kwargs in extract_features_calls:
                    assert 'summary' in kwargs and not kwargs['summary'], (
                        f"BUG NOT FIXED! extract_features should be called with summary=False. "
                        f"Got kwargs: {kwargs}"
                    )

    def test_sleep_stages_endpoint_passes_summary_false(self, test_client, valid_edf_bytes):
        """GREEN TEST: /sleep/stages endpoint MUST pass summary=False (after fix)."""

        class _StubRaw:
            def __init__(self):
                self.ch_names = [f"EEG{i}" for i in range(19)]
                self.info = {"sfreq": 256}

            def get_data(self):
                return np.random.randn(19, 5120).astype(np.float32)

        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", return_value=_StubRaw()):
            mock_model = MagicMock()
            extract_features_calls = []

            def track_extract_features(*args, **kwargs):
                extract_features_calls.append((args, kwargs))
                summary = kwargs.get('summary', True)
                if summary:
                    return np.ones(512).astype(np.float32)
                else:
                    return np.ones((4, 512)).astype(np.float32)

            mock_model.extract_features = track_extract_features
            mock_model.extract_windows = lambda data, sfreq: [data[:, :1024] for _ in range(5)]

            from brain_go_brrr.infra.ml_models.linear_probe import SleepStageProbe

            mock_probe = MagicMock(spec=SleepStageProbe)

            def check_2048_dims(features_tensor):
                assert features_tensor.shape[-1] == 2048, (
                    f"Expected 2048 dims, got {features_tensor.shape[-1]}"
                )
                return [2], torch.tensor([0.8])

            mock_probe.predict_stage = check_2048_dims

            with (
                patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model),
                patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe),
            ):
                response = test_client.post(
                    "/eeg/eegpt/sleep/stages",
                    files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
                )

                assert response.status_code == 200, f"Request failed: {response.text}"

                for _args, kwargs in extract_features_calls:
                    assert 'summary' in kwargs and not kwargs['summary'], (
                        f"BUG NOT FIXED! Got kwargs: {kwargs}"
                    )

    def test_analyze_batch_passes_summary_false(self, test_client, valid_edf_bytes):
        """GREEN TEST: /analyze/batch endpoint MUST pass summary=False to extract_features_batch."""

        class _StubRaw:
            def __init__(self):
                self.ch_names = [f"EEG{i}" for i in range(19)]
                self.info = {"sfreq": 256}

            def get_data(self):
                return np.random.randn(19, 10240).astype(np.float32)  # 40 seconds

        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", return_value=_StubRaw()):
            mock_model = MagicMock()
            extract_batch_calls = []

            def track_extract_batch(*args, **kwargs):
                extract_batch_calls.append((args, kwargs))
                batch_size = args[0].shape[0] if args else 2
                summary = kwargs.get('summary', True)
                if summary:
                    return np.ones((batch_size, 512)).astype(np.float32)
                else:
                    return np.ones((batch_size, 4, 512)).astype(np.float32)

            mock_model.extract_windows = lambda data, sfreq: [
                data[:, i : i + 1024] for i in range(0, 10240 - 1024, 1024)
            ]
            mock_model.extract_features_batch = track_extract_batch

            mock_probe = MagicMock()

            def check_2048_dims(features_tensor):
                assert features_tensor.shape[-1] == 2048, (
                    f"Expected 2048 dims, got {features_tensor.shape[-1]}"
                )
                return torch.tensor([[0.7, 0.3]] * features_tensor.shape[0])

            mock_probe.predict_proba = check_2048_dims

            with (
                patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model),
                patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe),
            ):
                response = test_client.post(
                    "/eeg/eegpt/analyze/batch?batch_size=2",
                    files={"edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")},
                    data={"analysis_type": "abnormality"},
                )

                assert response.status_code == 200, f"Request failed: {response.text}"

                assert len(extract_batch_calls) > 0, "extract_features_batch was not called"

                for _args, kwargs in extract_batch_calls:
                    assert 'summary' in kwargs and not kwargs['summary'], (
                        f"BUG NOT FIXED! extract_features_batch should be called with summary=False. "
                        f"Got kwargs: {kwargs}"
                    )


if __name__ == "__main__":
    # Run GREEN phase tests to confirm fixes work
    pytest.main([__file__, "-xvs"])

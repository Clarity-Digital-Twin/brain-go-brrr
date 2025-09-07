"""RED PHASE: Verify that router calls extract_features WITHOUT summary=False (the bug).

Per P0_CRITICAL_FIXES.md, the bug is that routers call extract_features without
passing summary=False, resulting in 512-d summaries instead of 2048-d features.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyedflib import EdfWriter


class TestP0ParameterCheck:
    """Verify the ACTUAL parameters passed to extract_features."""

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

            # Set up 19 EEG channels (minimum for EEGPT)
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

            # Write 20 seconds of data
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

    def test_analyze_endpoint_missing_summary_false(self, test_client, valid_edf_bytes):
        """RED TEST: /analyze endpoint MUST NOT pass summary=False (current bug).

        This test will PASS with the bug (summary not passed) and FAIL after fix.
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
                # Return appropriate shape based on summary parameter
                summary = kwargs.get('summary', True)  # Default is True (the bug!)
                if summary:
                    return np.ones(512).astype(np.float32)  # Summary features
                else:
                    return np.ones(2048).astype(np.float32)  # Full features

            mock_model.extract_features = track_extract_features
            mock_model.extract_windows = lambda data, sfreq: [data[:, :1024] for _ in range(5)]

            # Mock probe - make it look like AbnormalityProbe
            import torch

            from brain_go_brrr.infra.ml_models.linear_probe import AbnormalityProbe

            mock_probe = MagicMock(spec=AbnormalityProbe)
            mock_probe.predict_abnormal_probability = lambda x: torch.tensor(0.5)

            with (
                patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model),
                patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe)
            ):
                response = test_client.post(
                    "/eeg/eegpt/analyze",
                    files={
                        "edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")
                    },
                    data={"analysis_type": "abnormality_probe"},
                )

                assert response.status_code == 200

                # Check that extract_features was called
                assert len(extract_features_calls) > 0, "extract_features was not called"

                # THE BUG: summary=False is NOT passed
                for _args, kwargs in extract_features_calls:
                        # With the bug, 'summary' is not in kwargs or is True
                        # After fix, 'summary' should be False
                        assert 'summary' not in kwargs or kwargs['summary'], (
                            "BUG ALREADY FIXED! extract_features is being called with summary=False. "
                            f"Got kwargs: {kwargs}"
                        )

    def test_sleep_stages_endpoint_missing_summary_false(self, test_client, valid_edf_bytes):
        """RED TEST: /sleep/stages endpoint MUST NOT pass summary=False (current bug).

        This test will PASS with the bug (summary not passed) and FAIL after fix.
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
                # Return appropriate shape based on summary parameter
                summary = kwargs.get('summary', True)  # Default is True (the bug!)
                if summary:
                    return np.ones(512).astype(np.float32)  # Summary features
                else:
                    return np.ones(2048).astype(np.float32)  # Full features

            mock_model.extract_features = track_extract_features
            mock_model.extract_windows = lambda data, sfreq: [data[:, :1024] for _ in range(5)]

            # Mock probe - make it look like SleepStageProbe
            import torch

            from brain_go_brrr.infra.ml_models.linear_probe import SleepStageProbe

            mock_probe = MagicMock(spec=SleepStageProbe)
            mock_probe.predict_stage = lambda x: ([2], torch.tensor([0.8]))

            with (
                patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model),
                patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe)
            ):
                response = test_client.post(
                        "/eeg/eegpt/sleep/stages",
                        files={
                            "edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")
                        },
                    )

                    assert response.status_code == 200

                    # Check that extract_features was called
                    assert len(extract_features_calls) > 0, "extract_features was not called"

                    # THE BUG: summary=False is NOT passed
                    for _args, kwargs in extract_features_calls:
                        # With the bug, 'summary' is not in kwargs or is True
                        # After fix, 'summary' should be False
                        assert 'summary' not in kwargs or kwargs['summary'], (
                            "BUG ALREADY FIXED! extract_features is being called with summary=False. "
                            f"Got kwargs: {kwargs}"
                        )

    def test_analyze_batch_missing_summary_parameter(self, test_client, valid_edf_bytes):
        """RED TEST: extract_features_batch doesn't support summary parameter at all.

        This test verifies that extract_features_batch lacks the summary parameter.
        """

        # Mock the load_edf_safe to avoid EDF parsing
        class _StubRaw:
            def __init__(self):
                self.ch_names = [f"EEG{i}" for i in range(19)]
                self.info = {"sfreq": 256}

            def get_data(self):
                return np.random.randn(19, 10240).astype(np.float32)  # 40 seconds

        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", return_value=_StubRaw()):
            # Mock the model to track calls
            mock_model = MagicMock()

            # Track what parameters are passed to extract_features_batch
            extract_batch_calls = []

            def track_extract_batch(*args, **kwargs):
                extract_batch_calls.append((args, kwargs))
                # Return batch of 512-d features (the bug!)
                batch_size = args[0].shape[0] if args else 2
                return np.ones((batch_size, 512)).astype(np.float32)

            mock_model.extract_windows = lambda data, sfreq: [
                data[:, i : i + 1024] for i in range(0, 10240 - 1024, 1024)
            ]
            mock_model.extract_features_batch = track_extract_batch

            # Mock probe
            mock_probe = MagicMock()
            import torch

            mock_probe.predict_proba = lambda x: torch.tensor([[0.7, 0.3]] * x.shape[0])

            with (
                patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model),
                patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe)
            ):
                response = test_client.post(
                        "/eeg/eegpt/analyze/batch?batch_size=2",
                        files={
                            "edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")
                        },
                        data={"analysis_type": "abnormality"},
                    )

                    assert response.status_code == 200

                    # Check that extract_features_batch was called
                    assert len(extract_batch_calls) > 0, "extract_features_batch was not called"

                    # THE BUG: extract_features_batch doesn't accept summary parameter
                    for _args, kwargs in extract_batch_calls:
                        # With the bug, 'summary' cannot be in kwargs
                        # After fix, 'summary' should be False in kwargs
                        assert 'summary' not in kwargs, (
                            "BUG ALREADY FIXED! extract_features_batch now accepts summary parameter. "
                            f"Got kwargs: {kwargs}"
                        )


if __name__ == "__main__":
    # Run these tests to verify the bug exists
    pytest.main([__file__, "-xvs"])

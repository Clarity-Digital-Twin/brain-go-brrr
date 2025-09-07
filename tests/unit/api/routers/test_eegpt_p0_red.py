"""RED PHASE TESTS for P0 CRITICAL FIXES - These MUST FAIL until we fix the code!

These tests verify the dimension mismatch bugs described in P0_CRITICAL_FIXES.md.
The bug is subtle: code passes 512-d summaries (summary=True default) when it should
pass 2048-d flattened features (summary=False). The probes can handle both, but our
SSOT architecture requires 2048-d at boundaries.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from brain_go_brrr.api.routers.eegpt import router


class TestP0CriticalFixes:
    """RED PHASE: These tests MUST FAIL until we fix the dimension mismatch!"""

    @pytest.fixture
    def test_client(self):
        """Create test client with router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def valid_edf_bytes(self):
        """Create valid EDF file bytes."""
        from pyedflib import EdfWriter

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

            # Write 20 seconds of data (5 windows of 4 seconds)
            t = np.arange(20 * 256) / 256.0
            signal = 50 * np.sin(2 * np.pi * 10 * t)
            data = np.vstack([signal] * 19).astype(np.float64)
            for i in range(19):
                writer.writePhysicalSamples(data[i])

            writer.close()

            # Read the file bytes
            tmp_path = Path(tmp.name)
            content = tmp_path.read_bytes()
            tmp_path.unlink()

            return content

    @pytest.fixture
    def mock_raw_data(self):
        """Create mock raw EEG data."""
        sfreq = 256
        duration = 20  # seconds
        t = np.arange(duration * sfreq) / sfreq
        signal = 50e-6 * np.sin(2 * np.pi * 12 * t)

        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.vstack([signal] * 19)  # 19 channels
        mock_raw.ch_names = [f"EEG{i + 1}" for i in range(19)]
        mock_raw.info = {"sfreq": sfreq}
        return mock_raw

    @pytest.fixture
    def stub_edf_loader(self, mock_raw_data):
        """Stub EDF loader to avoid parsing real EDF."""

        class _StubRaw:
            def __init__(self):
                self.ch_names = mock_raw_data.ch_names
                self.info = mock_raw_data.info

            def get_data(self):
                return mock_raw_data.get_data()

        return lambda *_args, **_kw: _StubRaw()

    def test_analyze_endpoint_dimension_mismatch(
        self, test_client, valid_edf_bytes, stub_edf_loader
    ):
        """RED TEST #1: /analyze endpoint MUST FAIL with dimension mismatch.

        Current bug: extract_features uses summary=True (default) → 512 dims
        Probe expects: 2048 dims
        Expected: RuntimeError or similar dimension mismatch error
        """
        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", stub_edf_loader):
            # Create a mock model that simulates the ACTUAL bug behavior
            mock_model = MagicMock()

            def extract_windows(data, sfreq):
                # Return 5 windows for testing
                return [data[:, :1024] for _ in range(5)]

            def extract_features(data, channels, summary=True):  # DEFAULT IS TRUE (THE BUG!)
                # Return 512-d if summary=True (default), 2048-d if summary=False
                if summary:
                    return np.ones(512).astype(np.float32)  # BUG: Returns 512!
                else:
                    return np.ones(2048).astype(np.float32)  # Fixed: Returns 2048

            mock_model.extract_windows = extract_windows
            mock_model.extract_features = extract_features

            # Create a mock probe that strictly checks dimensions
            mock_probe = MagicMock()

            def strict_dimension_check(features_tensor):
                # This should receive (B, 2048) but will get (B, 512) with current bug
                if features_tensor.shape[-1] != 2048:
                    raise RuntimeError(
                        f"Dimension mismatch! Probe expects 2048 dims, got {features_tensor.shape[-1]}"
                    )
                return torch.tensor(0.5)

            mock_probe.predict_abnormal_probability = strict_dimension_check

            with patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model):
                with patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe):
                    # This SHOULD crash with dimension mismatch
                    response = test_client.post(
                        "/eeg/eegpt/analyze",
                        files={
                            "edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")
                        },
                        data={"analysis_type": "abnormality_probe"},
                    )

                    # With the bug, this should fail with 500 Internal Server Error
                    # After fix, it should succeed with 200
                    assert response.status_code == 500, (
                        "BUG NOT DETECTED! The endpoint should crash with dimension mismatch. "
                        "Current code passes 512-d features to probe expecting 2048-d."
                    )

    def test_sleep_stages_endpoint_dimension_mismatch(
        self, test_client, valid_edf_bytes, stub_edf_loader
    ):
        """RED TEST #2: /sleep/stages endpoint MUST FAIL with dimension mismatch.

        Current bug: extract_features uses summary=True (default) → 512 dims
        Probe expects: 2048 dims
        Expected: RuntimeError or similar dimension mismatch error
        """
        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", stub_edf_loader):
            # Create a mock model that simulates the ACTUAL bug behavior
            mock_model = MagicMock()

            def extract_windows(data, sfreq):
                # Return 5 windows for testing
                return [data[:, :1024] for _ in range(5)]

            def extract_features(data, channels, summary=True):  # DEFAULT IS TRUE (THE BUG!)
                # Return 512-d if summary=True (default), 2048-d if summary=False
                if summary:
                    return np.ones(512).astype(np.float32)  # BUG: Returns 512!
                else:
                    return np.ones(2048).astype(np.float32)  # Fixed: Returns 2048

            mock_model.extract_windows = extract_windows
            mock_model.extract_features = extract_features

            # Create a mock probe that strictly checks dimensions
            mock_probe = MagicMock()

            def strict_dimension_check(features_tensor):
                # This should receive (B, 2048) but will get (B, 512) with current bug
                if features_tensor.shape[-1] != 2048:
                    raise RuntimeError(
                        f"Dimension mismatch! Probe expects 2048 dims, got {features_tensor.shape[-1]}"
                    )
                return [2], torch.tensor([0.8])  # Return valid sleep stage

            mock_probe.predict_stage = strict_dimension_check

            with patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model):
                with patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe):
                    # This SHOULD crash with dimension mismatch
                    response = test_client.post(
                        "/eeg/eegpt/sleep/stages",
                        files={
                            "edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")
                        },
                    )

                    # With the bug, this should fail with 500 Internal Server Error
                    # After fix, it should succeed with 200
                    assert response.status_code == 500, (
                        "BUG NOT DETECTED! The endpoint should crash with dimension mismatch. "
                        "Current code passes 512-d features to probe expecting 2048-d."
                    )

    def test_analyze_batch_endpoint_dimension_mismatch(
        self, test_client, valid_edf_bytes, stub_edf_loader
    ):
        """RED TEST #3: /analyze/batch endpoint MUST FAIL with dimension mismatch.

        Current bug: extract_features_batch doesn't support summary parameter at all
        Even if we pass summary=False, it won't be forwarded
        Expected: RuntimeError or similar dimension mismatch error
        """
        with patch("brain_go_brrr.api.routers.eegpt.load_edf_safe", stub_edf_loader):
            # Create a mock model that returns 512-d features (the bug)
            mock_model = MagicMock()

            def extract_windows(data, sfreq):
                # Return 10 windows for batching
                return [data[:, :1024] for _ in range(10)]

            def extract_features_batch(batch_array, channels):
                # Current bug: always returns 512-d features
                batch_size = batch_array.shape[0]
                return np.ones((batch_size, 512)).astype(np.float32)  # BUG: 512 not 2048!

            mock_model.extract_windows = extract_windows
            mock_model.extract_features_batch = extract_features_batch

            # Create a mock probe that strictly checks dimensions
            mock_probe = MagicMock()

            def strict_dimension_check(features_tensor):
                # This should receive (B, 2048) but will get (B, 512) with current bug
                if features_tensor.shape[-1] != 2048:
                    raise RuntimeError(
                        f"Dimension mismatch! Probe expects 2048 dims, got {features_tensor.shape[-1]}"
                    )
                return torch.tensor([[0.7, 0.3]] * features_tensor.shape[0])

            mock_probe.predict_proba = strict_dimension_check

            with patch("brain_go_brrr.api.routers.eegpt.get_eegpt_model", return_value=mock_model):
                with patch("brain_go_brrr.api.routers.eegpt.get_probe", return_value=mock_probe):
                    # This SHOULD crash with dimension mismatch
                    response = test_client.post(
                        "/eeg/eegpt/analyze/batch?batch_size=2",
                        files={
                            "edf_file": ("test.edf", valid_edf_bytes, "application/octet-stream")
                        },
                        data={"analysis_type": "abnormality"},
                    )

                    # With the bug, this should fail with 500 Internal Server Error
                    # After fix, it should succeed with 200
                    assert response.status_code == 500, (
                        "BUG NOT DETECTED! The endpoint should crash with dimension mismatch. "
                        "Current code passes 512-d features to probe expecting 2048-d."
                    )

    def test_extract_features_batch_missing_summary_parameter(self):
        """RED TEST #4: extract_features_batch MUST NOT have summary parameter.

        This test verifies the bug exists - that extract_features_batch
        doesn't support the summary parameter we need.
        """
        from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel

        # Mock the model loading to avoid needing actual model file
        with patch("brain_go_brrr.infra.ml_models.eegpt_compat.EEGPTModel._load_model"):
            with patch("brain_go_brrr.infra.ml_models.eegpt_compat.EEGPTModel._validate_model"):
                model = EEGPTModel()

                # Check that extract_features_batch doesn't accept summary parameter
                import inspect

                sig = inspect.signature(model.extract_features_batch)
                params = list(sig.parameters.keys())

                # This assertion SHOULD PASS with current bug (summary not in params)
                # After fix, it should FAIL (summary should be in params)
                assert "summary" not in params, (
                    "BUG ALREADY FIXED? extract_features_batch now has summary parameter. "
                    "Update this test to verify it works correctly."
                )


if __name__ == "__main__":
    # Run RED phase tests to confirm they fail
    pytest.main([__file__, "-xvs", "-m", "red_phase"])

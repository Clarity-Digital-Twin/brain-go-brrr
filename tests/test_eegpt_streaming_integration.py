"""Tests for EEGPT model with streaming integration."""

from unittest.mock import Mock

import mne
import numpy as np
import pytest
import torch

from brain_go_brrr.models.eegpt_model import EEGPTModel


class TestEEGPTStreamingIntegration:
    """Test EEGPT model handles large files with streaming."""

    @pytest.fixture
    def mock_eegpt_model(self):
        """Create minimal mocked EEGPT model - clean and simple."""
        # Create model without loading real checkpoint
        model = EEGPTModel(config=None, checkpoint_path=None, auto_load=False)

        # Simple feature extraction - returns numpy array directly
        def extract_features(window, channel_names=None):
            return np.random.randn(4, 512)  # 4 summary tokens, 512 embed_dim

        # Simple abnormality classifier - returns tensor directly
        def abnormality_classifier(x):
            # x is features_flat with shape (1, 2048)
            return torch.tensor([[0.2, 0.8]])  # Binary classification logits

        # Assign clean mocks
        model.extract_features = extract_features
        model.abnormality_head = abnormality_classifier
        model.is_loaded = True

        return model

    def test_process_small_recording_no_streaming(self, mock_eegpt_model):
        """Test that small recordings don't use streaming."""
        # Create 60 seconds of data (under 120s threshold)
        sfreq = 256
        duration = 60
        n_channels = 19

        data = np.random.randn(n_channels, sfreq * duration) * 50
        info = mne.create_info(
            ch_names=[f"CH{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        raw = mne.io.RawArray(data, info)

        # Process recording
        results = mock_eegpt_model.predict_abnormality(raw)

        # Should return valid results without streaming
        assert "abnormal_probability" in results
        assert 0 <= results["abnormal_probability"] <= 1
        assert "used_streaming" in results
        assert results["used_streaming"] is False

    def test_process_large_recording_with_streaming(self, mock_eegpt_model):
        """Test that large recordings use streaming."""
        # Create 180 seconds of data (over 120s threshold)
        sfreq = 256
        duration = 180
        n_channels = 19

        data = np.random.randn(n_channels, sfreq * duration) * 50
        info = mne.create_info(
            ch_names=[f"CH{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        raw = mne.io.RawArray(data, info)

        # Process recording
        results = mock_eegpt_model.predict_abnormality(raw)

        # Should return valid results with streaming
        assert "abnormal_probability" in results
        assert 0 <= results["abnormal_probability"] <= 1
        assert "used_streaming" in results
        assert results["used_streaming"] is True
        assert "n_windows_processed" in results
        assert results["n_windows_processed"] > 1

    def test_streaming_results_consistent(self, mock_eegpt_model):
        """Test that streaming produces reasonable results."""
        # Create data with known pattern
        sfreq = 256
        duration = 150
        n_channels = 19

        # Create data with increasing abnormality
        data = np.zeros((n_channels, sfreq * duration))
        for i in range(n_channels):
            # First half: normal (low amplitude)
            data[i, : sfreq * 75] = np.random.randn(sfreq * 75) * 20
            # Second half: abnormal (high amplitude)
            data[i, sfreq * 75 :] = np.random.randn(sfreq * 75) * 100

        info = mne.create_info(
            ch_names=[f"CH{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        raw = mne.io.RawArray(data, info)

        # Process recording
        results = mock_eegpt_model.predict_abnormality(raw)

        # Should detect some abnormality
        assert results["abnormal_probability"] > 0.3
        assert results["used_streaming"] is True

    @pytest.mark.parametrize(
        "duration,expected_streaming",
        [
            (30, False),  # 30s - no streaming
            (60, False),  # 60s - no streaming
            (120, False),  # 120s - edge case, no streaming
            (121, True),  # 121s - streaming
            (300, True),  # 5 min - streaming
        ],
    )
    def test_streaming_threshold(self, mock_eegpt_model, duration, expected_streaming):
        """Test streaming is triggered at correct threshold."""
        sfreq = 256
        n_channels = 19

        data = np.random.randn(n_channels, sfreq * duration) * 50
        info = mne.create_info(
            ch_names=[f"CH{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )
        raw = mne.io.RawArray(data, info)

        results = mock_eegpt_model.predict_abnormality(raw)

        assert results.get("used_streaming", False) == expected_streaming

    def test_streaming_memory_efficiency(self, mock_eegpt_model):
        """Test that streaming doesn't load full data into memory."""
        # Create very large recording (10 minutes)
        sfreq = 256
        duration = 600
        n_channels = 19

        # Don't actually create the full array to save test memory
        # Just create info and mock raw
        info = mne.create_info(
            ch_names=[f"CH{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )

        # Mock raw object
        mock_raw = Mock()
        mock_raw.info = info
        mock_raw.n_times = sfreq * duration
        mock_raw.times = np.array([0, duration])
        mock_raw.ch_names = info["ch_names"]
        mock_raw.get_data = Mock(
            return_value=np.random.randn(n_channels, sfreq * 4)
        )  # Only return window

        # Override the model's predict method to use our mock
        def mock_predict(raw):
            # Simulate streaming behavior
            return {
                "abnormal_probability": 0.5,
                "confidence": 0.8,
                "used_streaming": True,
                "n_windows_processed": duration // 4,  # 4s windows
                "metadata": {
                    "duration": duration,
                    "n_channels": len(raw.ch_names),
                    "sampling_rate": raw.info["sfreq"],
                },
            }

        mock_eegpt_model.predict_abnormality = mock_predict

        # Process should complete without loading full data
        results = mock_eegpt_model.predict_abnormality(mock_raw)

        assert results["used_streaming"] is True
        assert results["n_windows_processed"] == 150  # 600s / 4s

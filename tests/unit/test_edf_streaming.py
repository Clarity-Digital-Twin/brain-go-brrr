"""Tests for EDF streaming functionality."""

import mne
import numpy as np
import pytest

from brain_go_brrr.data.edf_streaming import EDFStreamer, estimate_memory_usage


class TestEDFStreaming:
    """Test EDF streaming functionality."""

    def test_estimate_memory_usage(self, mock_edf_file):
        """Test memory usage estimation."""
        # Test with mock EDF file
        estimate = estimate_memory_usage(mock_edf_file, preload=True)

        assert "raw_data_mb" in estimate
        assert "estimated_total_mb" in estimate
        assert "duration_minutes" in estimate
        assert "n_channels" in estimate
        assert "sampling_rate" in estimate
        assert estimate["n_channels"] == 4
        assert estimate["sampling_rate"] == 256
        assert estimate["duration_minutes"] == pytest.approx(10 / 60, rel=0.1)

    def test_estimate_memory_usage_nonexistent_file(self, tmp_path):
        """Test memory usage estimation with nonexistent file."""
        fake_path = tmp_path / "nonexistent.edf"
        estimate = estimate_memory_usage(fake_path, preload=True)

        assert estimate["raw_data_mb"] == 0.0
        assert estimate["estimated_total_mb"] == 0.0
        assert estimate["n_channels"] == 0

    @pytest.fixture
    def mock_edf_file(self, tmp_path):
        """Create a mock EDF file for testing."""
        # Create minimal EEG data
        sfreq = 256
        duration = 10  # seconds
        n_channels = 4
        ch_names = ["EEG1", "EEG2", "EEG3", "EEG4"]

        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Save to EDF
        edf_path = tmp_path / "test.edf"
        raw.export(str(edf_path), fmt="edf")
        return edf_path

    def test_edf_streamer_initialization(self, mock_edf_file):
        """Test EDFStreamer initialization."""
        streamer = EDFStreamer(mock_edf_file)
        assert streamer.file_path == mock_edf_file
        assert streamer.chunk_duration == 30.0  # default

    def test_edf_streamer_get_info(self, mock_edf_file):
        """Test getting EDF file info."""
        with EDFStreamer(mock_edf_file) as streamer:
            info = streamer.get_info()

            assert "n_channels" in info
            assert "sampling_rate" in info
            assert "duration" in info
            assert "channel_names" in info

            assert info["n_channels"] == 4
            assert info["sampling_rate"] == 256
            assert info["duration"] == pytest.approx(10.0, rel=0.1)

    def test_decide_streaming_threshold(self, mock_edf_file):
        """Test decide_streaming function with threshold."""
        from brain_go_brrr.data.edf_streaming import decide_streaming

        # Small file should not use streaming
        result = decide_streaming(mock_edf_file, max_memory_mb=1000)
        assert result["used_streaming"] is False
        assert result["chunks_processed"] == 1

        # Force streaming with tiny threshold
        result = decide_streaming(mock_edf_file, max_memory_mb=0.001)
        assert result["used_streaming"] is True
        assert result["chunks_processed"] > 0

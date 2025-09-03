"""Test channel routing functionality."""

import mne
import numpy as np
import pytest

from brain_go_brrr.api.routers.channel_router import ChannelRouter


class TestChannelRouter:
    """Test the channel routing service."""

    def create_mock_raw(self, n_channels: int, sfreq: float = 256, duration: float = 60):
        """Create mock MNE Raw object for testing."""
        # Create channel names based on standard 10-20 system
        standard_channels = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "O2",
        ]

        if n_channels <= len(standard_channels):
            ch_names = standard_channels[:n_channels]
        else:
            ch_names = standard_channels + [
                f"EEG{i:03d}" for i in range(len(standard_channels), n_channels)
            ]

        # Create data
        n_samples = int(sfreq * duration)
        data = np.random.randn(n_channels, n_samples) * 1e-6  # microvolts

        # Create info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create Raw object
        raw = mne.io.RawArray(data, info)
        return raw

    def test_route_to_yasa_with_few_channels(self):
        """Test routing to YASA when <19 channels available."""
        # Test with 2 channels (like Sleep-EDF)
        raw = self.create_mock_raw(n_channels=2)
        method, metadata = ChannelRouter.determine_analysis_method(raw, "auto")

        assert method == "yasa"
        assert metadata["n_channels"] == 2
        assert "routing_reason" in metadata
        assert metadata["routing_reason"] == "insufficient_channels_for_eegpt"

    def test_route_to_eegpt_with_full_montage(self):
        """Test routing to EEGPT when 19+ channels available."""
        # Test with 19 channels (TUAB standard)
        raw = self.create_mock_raw(n_channels=19)
        method, metadata = ChannelRouter.determine_analysis_method(raw, "auto")

        assert method == "eegpt"
        assert metadata["n_channels"] == 19
        assert "routing_reason" not in metadata

    def test_force_yasa_with_full_montage(self):
        """Test forcing YASA even with 19+ channels."""
        raw = self.create_mock_raw(n_channels=20)
        method, metadata = ChannelRouter.determine_analysis_method(raw, "yasa")

        assert method == "yasa"
        assert metadata["n_channels"] == 20

    def test_force_eegpt_fails_with_few_channels(self):
        """Test that forcing EEGPT with <19 channels raises error."""
        raw = self.create_mock_raw(n_channels=10)

        with pytest.raises(ValueError, match="EEGPT requires at least 19 channels"):
            ChannelRouter.determine_analysis_method(raw, "eegpt")

    def test_single_channel_works_with_yasa(self):
        """Test that single channel EEG works with YASA."""
        raw = self.create_mock_raw(n_channels=1)
        method, metadata = ChannelRouter.determine_analysis_method(raw, "auto")

        assert method == "yasa"
        assert metadata["n_channels"] == 1

    def test_central_channel_detection(self):
        """Test detection of central channels for YASA."""
        # Create raw with C3 and C4
        ch_names = ["C3", "C4"]
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
        data = np.random.randn(2, 256 * 60) * 1e-6
        raw = mne.io.RawArray(data, info)

        method, metadata = ChannelRouter.determine_analysis_method(raw, "auto")

        assert method == "yasa"
        assert metadata["has_central_channels"] is True

    def test_validate_for_sleep_analysis(self):
        """Test validation of EEG data for sleep analysis."""
        # Valid data
        raw = self.create_mock_raw(n_channels=2, sfreq=256, duration=60)
        is_valid, message = ChannelRouter.validate_for_sleep_analysis(raw)
        assert is_valid is True

        # Too short
        raw_short = self.create_mock_raw(n_channels=2, sfreq=256, duration=10)
        is_valid, message = ChannelRouter.validate_for_sleep_analysis(raw_short)
        assert is_valid is False
        assert "too short" in message

        # Low sampling rate
        raw_low_sr = self.create_mock_raw(n_channels=2, sfreq=30, duration=60)
        is_valid, message = ChannelRouter.validate_for_sleep_analysis(raw_low_sr)
        assert is_valid is False
        assert "Sampling rate too low" in message

    def test_get_method_info(self):
        """Test getting method information."""
        info = ChannelRouter.get_method_info()

        assert "yasa" in info
        assert "eegpt" in info

        assert info["yasa"]["min_channels"] == 1
        assert info["eegpt"]["min_channels"] == 19

        assert "accuracy" in info["yasa"]
        assert "accuracy" in info["eegpt"]

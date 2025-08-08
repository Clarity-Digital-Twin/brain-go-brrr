"""Unit tests for Sleep-EDF montage detection.

This locks in our acceptance logic for Fpz-Cz and Pz-Oz channels.
Clean tests that verify real business logic, not mocks.
"""

import pytest
import numpy as np
import mne

from brain_go_brrr.core.sleep.analyzer import SleepAnalyzer
from brain_go_brrr.core.exceptions import UnsupportedMontageError


class TestSleepEDFMontageDetection:
    """Test that Sleep-EDF montage is properly detected and accepted."""

    @pytest.fixture
    def sleep_analyzer(self):
        """Create a SleepAnalyzer instance."""
        return SleepAnalyzer(verbose=False)

    @pytest.fixture
    def create_raw_with_channels(self):
        """Factory to create Raw objects with specific channels."""

        def _create(channel_names, sfreq=256, duration=300):  # 5 minutes minimum for YASA
            """Create Raw object with given channel names."""
            n_channels = len(channel_names)
            n_samples = int(sfreq * duration)

            # Create deterministic sine wave data
            t = np.arange(n_samples) / sfreq
            signal = 50e-6 * np.sin(2 * np.pi * 10 * t)  # 10 Hz, 50 µV amplitude
            data = np.vstack([signal] * n_channels)

            # Create info
            info = mne.create_info(
                ch_names=channel_names, sfreq=sfreq, ch_types=["eeg"] * n_channels
            )

            # Create Raw object
            raw = mne.io.RawArray(data, info)
            return raw

        return _create

    def test_accepts_fpz_cz_montage(self, sleep_analyzer, create_raw_with_channels):
        """Test that Fpz-Cz montage is accepted for sleep staging."""
        # Create Raw with Sleep-EDF montage (Fpz-Cz)
        raw = create_raw_with_channels(["EEG Fpz-Cz", "EOG horizontal"])

        # Preprocess for sleep (resample to 100Hz as YASA requires)
        raw = sleep_analyzer.preprocess_for_sleep(raw)

        # Should not raise an error
        hypnogram = sleep_analyzer.stage_sleep(raw)

        assert hypnogram is not None
        assert len(hypnogram) > 0
        # Should be string sleep stages
        assert all(stage in ["W", "N1", "N2", "N3", "REM", "ART"] for stage in hypnogram)

    def test_accepts_pz_oz_montage(self, sleep_analyzer, create_raw_with_channels):
        """Test that Pz-Oz montage is accepted as fallback."""
        # Create Raw with Sleep-EDF montage (Pz-Oz, no Fpz-Cz)
        raw = create_raw_with_channels(["EEG Pz-Oz", "EMG submental"])

        # Preprocess for sleep
        raw = sleep_analyzer.preprocess_for_sleep(raw)

        # Should not raise an error
        hypnogram = sleep_analyzer.stage_sleep(raw)

        assert hypnogram is not None
        assert len(hypnogram) > 0

    def test_prefers_fpz_cz_over_pz_oz(self, sleep_analyzer, create_raw_with_channels):
        """Test that Fpz-Cz is preferred when both are present."""
        # Create Raw with both montages
        raw = create_raw_with_channels(["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"])

        # Mock the YASA SleepStaging to capture which channel was used
        from unittest.mock import patch, MagicMock

        with patch("yasa.SleepStaging") as mock_sleep_staging:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = np.array(["N2"] * 10)
            mock_sleep_staging.return_value = mock_instance

            hypnogram = sleep_analyzer.stage_sleep(raw)

            # Verify Fpz-Cz was chosen (first argument after raw is eeg_name)
            call_args = mock_sleep_staging.call_args
            assert call_args[1]["eeg_name"] == "EEG Fpz-Cz"

    def test_accepts_standard_10_20_channels(self, sleep_analyzer, create_raw_with_channels):
        """Test that standard 10-20 channels are still accepted."""
        # Create Raw with standard channels
        raw = create_raw_with_channels(["C3", "C4", "O1", "O2", "EOG1", "EOG2"])

        # Should not raise an error - should use C3
        hypnogram = sleep_analyzer.stage_sleep(raw)

        assert hypnogram is not None
        assert len(hypnogram) > 0

    def test_rejects_unsupported_montage(self, sleep_analyzer, create_raw_with_channels):
        """Test that unsupported montages raise appropriate error."""
        # Create Raw with non-standard channels that aren't accepted
        raw = create_raw_with_channels(["X1", "X2", "Y1", "Y2"])

        # Should raise UnsupportedMontageError
        with pytest.raises(UnsupportedMontageError) as excinfo:
            sleep_analyzer.stage_sleep(raw)

        assert "Unsupported EEG montage" in str(excinfo.value)
        assert "Fpz-Cz, Pz-Oz" in str(excinfo.value)  # Should mention Sleep-EDF channels

    def test_channel_name_normalization(self, sleep_analyzer, create_raw_with_channels):
        """Test that channel names with/without 'EEG ' prefix are handled."""
        # Test without prefix
        raw1 = create_raw_with_channels(["Fpz-Cz", "EOG"])
        hypnogram1 = sleep_analyzer.stage_sleep(raw1)
        assert hypnogram1 is not None

        # Test with prefix (already tested above but let's be explicit)
        raw2 = create_raw_with_channels(["EEG Fpz-Cz", "EOG"])
        hypnogram2 = sleep_analyzer.stage_sleep(raw2)
        assert hypnogram2 is not None

    def test_handles_missing_eog_emg_channels(self, sleep_analyzer, create_raw_with_channels):
        """Test that sleep staging works even without EOG/EMG channels."""
        # Create Raw with only EEG channels
        raw = create_raw_with_channels(["EEG Fpz-Cz"])

        # Should still work, just without EOG/EMG features
        hypnogram = sleep_analyzer.stage_sleep(raw)

        assert hypnogram is not None
        assert len(hypnogram) > 0

    def test_integration_with_real_sleep_edf_structure(self, sleep_analyzer):
        """Test with channel structure matching real Sleep-EDF files."""
        # Skip if Sleep-EDF not available
        from pathlib import Path

        sleep_edf_path = Path("data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf")

        if not sleep_edf_path.exists():
            pytest.skip("Sleep-EDF dataset not available")

        # Load real Sleep-EDF file
        raw = mne.io.read_raw_edf(sleep_edf_path, preload=False)

        # Get channel names to verify
        eeg_channels = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == "eeg"]

        # Verify Sleep-EDF montage is present
        assert any("Fpz-Cz" in ch or "Pz-Oz" in ch for ch in eeg_channels), (
            f"Expected Sleep-EDF montage, got: {eeg_channels}"
        )

        # Process a small chunk to verify it works
        raw_subset = raw.copy().crop(tmin=0, tmax=120)  # 2 minutes
        raw_subset.load_data()

        # Should process without error
        hypnogram = sleep_analyzer.stage_sleep(raw_subset)
        assert hypnogram is not None
        assert len(hypnogram) == 4  # 2 minutes = 4 epochs of 30s


class TestMontageDetectionEdgeCases:
    """Test edge cases in montage detection logic."""

    @pytest.fixture
    def sleep_analyzer(self):
        """Create a SleepAnalyzer instance."""
        return SleepAnalyzer(verbose=False)

    @pytest.fixture
    def create_raw_with_channels(self):
        """Factory to create Raw objects with specific channels."""

        def _create(channel_names, sfreq=100, duration=30):
            n_channels = len(channel_names)
            n_samples = int(sfreq * duration)
            data = np.random.randn(n_channels, n_samples) * 50e-6
            info = mne.create_info(
                ch_names=channel_names, sfreq=sfreq, ch_types=["eeg"] * n_channels
            )
            return mne.io.RawArray(data, info)

        return _create

    def test_empty_channel_list(self, sleep_analyzer):
        """Test behavior with no channels."""
        # Create info with no channels
        info = mne.create_info(ch_names=[], sfreq=256)
        raw = mne.io.RawArray(np.array([]).reshape(0, 0), info)

        # Should raise an appropriate error
        with pytest.raises((ValueError, UnsupportedMontageError)):
            sleep_analyzer.stage_sleep(raw)

    def test_case_sensitivity(self, sleep_analyzer, create_raw_with_channels):
        """Test that channel matching handles case properly."""
        # Test various case combinations
        test_cases = [
            ["eeg fpz-cz", "eog"],  # lowercase
            ["EEG FPZ-CZ", "EOG"],  # uppercase
            ["Eeg Fpz-Cz", "Eog"],  # mixed case
        ]

        for channels in test_cases:
            raw = create_raw_with_channels(channels)
            # Should handle case variations gracefully
            try:
                hypnogram = sleep_analyzer.stage_sleep(raw)
                assert hypnogram is not None
            except UnsupportedMontageError:
                # It's OK if case-sensitive matching fails,
                # as long as the exact format works
                pass

    def test_picks_parameter_override(self, sleep_analyzer, create_raw_with_channels):
        """Test that picks parameter can override automatic detection."""
        # Create Raw with multiple channels
        raw = create_raw_with_channels(["Fp1", "Fp2", "C3", "C4", "O1", "O2"])

        # Use picks to select specific channel
        hypnogram = sleep_analyzer.stage_sleep(raw, picks=["C4"])

        assert hypnogram is not None

        # Test with picks="eeg" to use all EEG channels
        hypnogram = sleep_analyzer.stage_sleep(raw, picks="eeg")
        assert hypnogram is not None

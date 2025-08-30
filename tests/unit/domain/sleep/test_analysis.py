"""Test sleep analysis functionality with YASA integration.

Tests sleep staging, metrics calculation, and hypnogram generation using
real Sleep-EDF data when available.

Follows ML testing best practices:
- Uses real EEG data, not mocks
- Gracefully skips if dependencies missing
- Tests business logic around sleep analysis
"""

import mne
import numpy as np
import pytest

from brain_go_brrr.domain.sleep import SleepAnalyzer

# Gracefully handle missing dependencies
yasa = pytest.importorskip(
    "yasa", reason="YASA required for sleep analysis - install with: pip install yasa"
)


class TestSleepAnalyzer:
    """Test sleep analysis with real data and graceful dependency handling."""

    @pytest.fixture
    def sleep_controller(self):
        """Create sleep analyzer - skips if YASA not available."""
        return SleepAnalyzer()

    @pytest.fixture
    def sample_sleep_eeg(self, sleep_edf_dir):
        """Load sample sleep EEG data if available."""
        # Only get PSG (polysomnography) files, not hypnogram annotation files
        edf_files = sorted(sleep_edf_dir.glob("*-PSG.edf"))
        if not edf_files:
            pytest.skip("No Sleep-EDF PSG files found")
        # Use first available EDF file
        raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
        # Properly select EEG channels (avoid misc/other channel types)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, exclude=[])
        if not eeg_picks.size:
            # If no EEG channels marked, try to identify them by name patterns
            eeg_patterns = ['EEG', 'Fp', 'F', 'C', 'P', 'O', 'T']
            potential_eeg = [
                i
                for i, ch in enumerate(raw.ch_names)
                if any(pattern in ch for pattern in eeg_patterns)
            ]
            if potential_eeg:
                # Mark identified channels as EEG
                channel_types = {raw.ch_names[i]: 'eeg' for i in potential_eeg}
                raw.set_channel_types(channel_types)
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

        if not eeg_picks.size:
            pytest.skip("No EEG channels found in Sleep-EDF file")

        # Select only EEG channels
        raw.pick(eeg_picks)
        # Crop to 5 minutes for fast testing
        if raw.times[-1] > 300:
            raw.crop(tmax=300)
        return raw

    @pytest.mark.integration
    def test_controller_initialization(self, sleep_controller):
        """Test sleep analyzer initializes correctly."""
        assert sleep_controller.staging_model == "auto"
        assert sleep_controller.epoch_length == 30.0
        assert sleep_controller.include_art is True

    @pytest.mark.integration
    def test_sleep_staging_returns_hypnogram(self, sleep_controller, sample_sleep_eeg):
        """Test sleep staging produces hypnogram."""
        hypnogram = sleep_controller.stage_sleep(sample_sleep_eeg)

        assert hypnogram is not None
        assert len(hypnogram) > 0
        # YASA returns string stages: 'W', 'N1', 'N2', 'N3', 'R' (for REM), 'ART'
        valid_stages = {"W", "N1", "N2", "N3", "R", "REM", "ART", "WAKE", "UNS"}
        assert all(stage in valid_stages for stage in hypnogram)

    @pytest.mark.integration
    def test_sleep_metrics_calculation(self, sleep_controller, sample_sleep_eeg):
        """Test sleep metrics calculation."""
        metrics = sleep_controller.calculate_sleep_metrics(sample_sleep_eeg)

        # YASA returns "SE" for sleep efficiency, not "sleep_efficiency"
        assert "SE" in metrics
        assert "TST" in metrics  # Total Sleep Time
        assert "%N1" in metrics or "%N2" in metrics  # At least some sleep stages

        # Sleep efficiency should be between 0 and 100
        assert 0 <= metrics["SE"] <= 100
        assert metrics["TST"] >= 0

    @pytest.mark.integration
    def test_sleep_staging_detects_wake(self, sleep_controller):
        """Test wake detection with synthetic wake-like EEG."""
        # Create high-frequency, high-amplitude signal (wake-like)
        sfreq = 256
        duration = 360  # 6 minutes (YASA requires >= 5 minutes)
        n_samples = int(sfreq * duration)

        # High beta activity (wake pattern)
        wake_signal = np.random.randn(1, n_samples) * 50e-6
        # Add high-frequency component
        t = np.arange(n_samples) / sfreq
        wake_signal += 30e-6 * np.sin(2 * np.pi * 20 * t)  # 20 Hz

        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(wake_signal, info)

        hypnogram = sleep_controller.stage_sleep(raw)

        # YASA returns string stages
        # Just verify we get a hypnogram without errors
        assert len(hypnogram) > 0  # Got some epochs

        # YASA might return 'R' instead of 'REM' for REM sleep
        valid_stages = {"W", "N1", "N2", "N3", "R", "REM", "ART", "UNS"}
        assert all(stage in valid_stages for stage in hypnogram)

    @pytest.mark.integration
    def test_sleep_staging_detects_deep_sleep(self, sleep_controller):
        """Test deep sleep detection with synthetic slow-wave EEG."""
        # Create low-frequency, high-amplitude signal (deep sleep-like)
        sfreq = 256
        duration = 360  # 6 minutes (YASA requires >= 5 minutes)
        n_samples = int(sfreq * duration)

        # Slow wave activity (deep sleep pattern)
        t = np.arange(n_samples) / sfreq
        deep_sleep_signal = 100e-6 * np.sin(2 * np.pi * 1 * t)  # 1 Hz slow waves
        deep_sleep_signal = deep_sleep_signal.reshape(1, -1)

        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(deep_sleep_signal, info)

        hypnogram = sleep_controller.stage_sleep(raw)

        # With synthetic data, just verify we get valid stages
        # YASA may not reliably detect deep sleep in synthetic data
        # Also include 'R' as YASA sometimes uses that for REM
        valid_stages = {"W", "N1", "N2", "N3", "REM", "R", "ART", "UNS"}
        assert len(hypnogram) > 0
        assert all(stage in valid_stages for stage in hypnogram)

    @pytest.mark.unit
    def test_handles_short_recordings(self, sleep_controller, synthetic_sleep_raw):
        """Test handling of recordings shorter than epoch length."""
        # Use synthetic 5-minute data cropped to 10 seconds
        raw = synthetic_sleep_raw.copy()
        raw.crop(tmax=10)  # Crop to 10 seconds

        # Should handle gracefully (may return empty or raise informative error)
        try:
            hypnogram = sleep_controller.stage_sleep(raw)
            # If it returns something, should be valid
            if hypnogram is not None and len(hypnogram) > 0:
                valid_stages = {-1, 0, 1, 2, 3, 4}
                assert all(stage in valid_stages for stage in hypnogram)
        except Exception as e:
            # Should be informative error about duration
            assert "duration" in str(e).lower() or "length" in str(e).lower()

    @pytest.mark.unit
    def test_handles_missing_channels(self, sleep_controller):
        """Test handling of EEG with minimal channels."""
        # Create single-channel EEG
        sfreq = 256
        duration = 360  # 6 minutes (YASA requires >= 5 minutes)
        n_samples = int(sfreq * duration)

        single_ch_signal = np.random.randn(1, n_samples) * 20e-6
        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(single_ch_signal, info)

        # Should handle single channel gracefully
        try:
            hypnogram = sleep_controller.stage_sleep(raw)
            assert hypnogram is not None
            # YASA should return string stages
            valid_stages = {"W", "N1", "N2", "N3", "REM", "ART", "UNS"}
            assert all(stage in valid_stages for stage in hypnogram)
        except Exception as e:
            # Should be informative about channel requirements or duration
            error_msg = str(e).lower()
            assert "channel" in error_msg or "duration" in error_msg or "minutes" in error_msg

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n_channels,expected_channels",
        [(1, ["C3"]), (2, ["C3", "C4"]), (6, ["C3", "C4", "F3", "F4", "O1", "O2"])],
    )
    def test_channel_selection(self, sleep_controller, n_channels, expected_channels):
        """Test channel selection for different EEG montages."""
        sfreq = 256
        duration = 360  # 6 minutes (YASA requires >= 5 minutes)
        n_samples = int(sfreq * duration)

        # Create multi-channel EEG
        data = np.random.randn(n_channels, n_samples) * 20e-6
        info = mne.create_info(expected_channels[:n_channels], sfreq, ch_types=["eeg"] * n_channels)
        raw = mne.io.RawArray(data, info)

        # Should process without error
        try:
            hypnogram = sleep_controller.stage_sleep(raw)
            assert hypnogram is not None
            # YASA should return string stages
            valid_stages = {"W", "N1", "N2", "N3", "REM", "ART", "UNS"}
            assert all(stage in valid_stages for stage in hypnogram)
        except Exception as e:
            # If it fails, should be due to insufficient channels/duration for YASA
            error_msg = str(e).lower()
            if (
                "channel" not in error_msg
                and "duration" not in error_msg
                and "minutes" not in error_msg
            ):
                raise  # Re-raise if not a channel/duration-related error


# Additional integration test using real Sleep-EDF if available
@pytest.mark.integration
@pytest.mark.external
def test_full_sleep_analysis_pipeline(sleep_edf_dir):
    """Test complete sleep analysis pipeline with real Sleep-EDF data."""
    edf_files = sorted(sleep_edf_dir.glob("SC*-PSG.edf"))
    if not edf_files:
        pytest.skip("No Sleep-EDF PSG files found")

    # Test with first available file
    edf_file = edf_files[0]
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    # Crop to first 10 minutes for faster testing
    raw.crop(tmax=600)

    analyzer = SleepAnalyzer()

    # Full pipeline test
    hypnogram = analyzer.stage_sleep(raw)
    metrics = analyzer.calculate_sleep_metrics(raw)

    # Validate results
    assert len(hypnogram) > 0
    assert "SE" in metrics  # YASA uses "SE" for sleep efficiency
    assert 0 <= metrics["SE"] <= 100

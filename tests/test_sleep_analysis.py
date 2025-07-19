"""Test sleep analysis functionality with YASA integration.

Tests sleep staging, metrics calculation, and hypnogram generation using
real Sleep-EDF data when available.

Follows ML testing best practices:
- Uses real EEG data, not mocks
- Gracefully skips if dependencies missing
- Tests business logic around sleep analysis
"""

from pathlib import Path

import mne
import numpy as np
import pytest

from services.sleep_metrics import SleepAnalyzer

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
    def sample_sleep_eeg(self):
        """Load sample sleep EEG data if available."""
        # Check for Sleep-EDF data first
        sleep_edf_path = Path("data/datasets/external/sleep-edf/sleep-cassette")
        if sleep_edf_path.exists():
            # Only get PSG (polysomnography) files, not hypnogram annotation files
            edf_files = list(sleep_edf_path.glob("*-PSG.edf"))
            if edf_files:
                # Use first available EDF file
                raw = mne.io.read_raw_edf(edf_files[0], preload=True, verbose=False)
                # Set channel types if not set (common issue with EDF files)
                if all(ch_type == "misc" for ch_type in raw.get_channel_types()):
                    # Assume all channels are EEG for sleep data
                    raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))
                # Crop to 5 minutes for fast testing
                if raw.times[-1] > 300:
                    raw.crop(tmax=300)
                return raw

        # Fallback to synthetic sleep-like EEG
        pytest.skip("No Sleep-EDF data available - download dataset for integration tests")

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
        valid_stages = {"W", "N1", "N2", "N3", "R", "REM", "ART", "WAKE"}
        # Check what values we're getting
        unique_stages = set(hypnogram) if hasattr(hypnogram, "__iter__") else {hypnogram}
        print(f"Unique stages found: {unique_stages}")
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
        duration = 60  # 1 minute
        n_samples = int(sfreq * duration)

        # High beta activity (wake pattern)
        wake_signal = np.random.randn(1, n_samples) * 50e-6
        # Add high-frequency component
        t = np.arange(n_samples) / sfreq
        wake_signal += 30e-6 * np.sin(2 * np.pi * 20 * t)  # 20 Hz

        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(wake_signal, info)

        hypnogram = sleep_controller.stage_sleep(raw)

        # Should detect mostly wake (stage 0) in high-frequency signal
        wake_percentage = np.mean(np.array(hypnogram) == 0) * 100
        assert wake_percentage > 30  # At least some wake detected

    @pytest.mark.integration
    def test_sleep_staging_detects_deep_sleep(self, sleep_controller):
        """Test deep sleep detection with synthetic slow-wave EEG."""
        # Create low-frequency, high-amplitude signal (deep sleep-like)
        sfreq = 256
        duration = 60  # 1 minute
        n_samples = int(sfreq * duration)

        # Slow wave activity (deep sleep pattern)
        t = np.arange(n_samples) / sfreq
        deep_sleep_signal = 100e-6 * np.sin(2 * np.pi * 1 * t)  # 1 Hz slow waves
        deep_sleep_signal = deep_sleep_signal.reshape(1, -1)

        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(deep_sleep_signal, info)

        hypnogram = sleep_controller.stage_sleep(raw)

        # Should detect some non-wake stages in slow-wave signal
        non_wake_percentage = np.mean(np.array(hypnogram) != 0) * 100
        assert non_wake_percentage > 10  # Some sleep stages detected

    @pytest.mark.unit
    def test_handles_short_recordings(self, sleep_controller):
        """Test handling of recordings shorter than epoch length."""
        # Create 10-second recording (shorter than 30s epoch)
        sfreq = 256
        duration = 10
        n_samples = int(sfreq * duration)

        short_signal = np.random.randn(1, n_samples) * 20e-6
        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(short_signal, info)

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
        duration = 120  # 2 minutes
        n_samples = int(sfreq * duration)

        single_ch_signal = np.random.randn(1, n_samples) * 20e-6
        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(single_ch_signal, info)

        # Should handle single channel gracefully
        try:
            hypnogram = sleep_controller.stage_sleep(raw)
            assert hypnogram is not None
        except Exception as e:
            # Should be informative about channel requirements
            assert "channel" in str(e).lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n_channels,expected_channels",
        [(1, ["C3"]), (2, ["C3", "C4"]), (6, ["C3", "C4", "F3", "F4", "O1", "O2"])],
    )
    def test_channel_selection(self, sleep_controller, n_channels, expected_channels):
        """Test channel selection for different EEG montages."""
        sfreq = 256
        duration = 60
        n_samples = int(sfreq * duration)

        # Create multi-channel EEG
        data = np.random.randn(n_channels, n_samples) * 20e-6
        info = mne.create_info(expected_channels[:n_channels], sfreq, ch_types=["eeg"] * n_channels)
        raw = mne.io.RawArray(data, info)

        # Should process without error
        try:
            hypnogram = sleep_controller.stage_sleep(raw)
            assert hypnogram is not None
        except Exception as e:
            # If it fails, should be due to insufficient channels for YASA
            if "channel" not in str(e).lower():
                raise  # Re-raise if not a channel-related error


# Additional integration test using real Sleep-EDF if available
@pytest.mark.integration
@pytest.mark.external
def test_full_sleep_analysis_pipeline():
    """Test complete sleep analysis pipeline with real Sleep-EDF data."""
    sleep_edf_base = Path("data/datasets/external/sleep-edf/sleep-cassette")

    if not sleep_edf_base.exists():
        pytest.skip("Sleep-EDF dataset not available - download for full integration tests")

    edf_files = list(sleep_edf_base.glob("SC*-PSG.edf"))
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

    print(f"✅ Sleep analysis completed: {len(hypnogram)} epochs, {metrics['SE']:.1f}% efficiency")

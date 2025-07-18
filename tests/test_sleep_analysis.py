"""Tests for sleep analysis functionality."""


import mne
import numpy as np
import pytest

from services.sleep_metrics import SleepAnalysisController


class TestSleepAnalysisController:
    """Test suite for sleep analysis controller."""

    @pytest.fixture
    def mock_eeg_data(self):
        """Create mock EEG data for testing."""
        # Create 30 minutes of EEG data at 256 Hz
        sfreq = 256
        duration = 30 * 60  # 30 minutes
        n_channels = 6  # C3, C4, F3, F4, O1, O2
        n_samples = int(sfreq * duration)

        # Generate realistic-looking EEG data
        np.random.seed(42)
        times = np.arange(n_samples) / sfreq

        # Simulate different sleep stages with characteristic frequencies
        # Wake: 8-13 Hz (alpha), N1: 4-8 Hz (theta), N2: sleep spindles, N3: 0.5-4 Hz (delta)
        data = np.zeros((n_channels, n_samples))

        # First 10 minutes: Wake (alpha activity)
        wake_end = 10 * 60 * sfreq
        for ch in range(n_channels):
            alpha_freq = 10 + np.random.rand() * 2  # 10-12 Hz
            data[ch, :wake_end] = np.sin(2 * np.pi * alpha_freq * times[:wake_end])
            data[ch, :wake_end] += 0.5 * np.random.randn(wake_end)  # Add noise

        # Next 10 minutes: N2 (sleep spindles + K-complexes)
        n2_start = wake_end
        n2_end = 20 * 60 * sfreq
        for ch in range(n_channels):
            # Base theta activity
            theta_freq = 6 + np.random.rand()
            data[ch, n2_start:n2_end] = 0.5 * np.sin(2 * np.pi * theta_freq * times[:n2_end-n2_start])

            # Add sleep spindles (12-14 Hz bursts)
            for _ in range(5):  # 5 spindles
                spindle_start = n2_start + np.random.randint(0, n2_end - n2_start - sfreq*2)
                spindle_duration = int(0.5 * sfreq + np.random.rand() * sfreq)  # 0.5-1.5 seconds
                spindle_freq = 13 + np.random.rand()
                spindle_envelope = np.hanning(spindle_duration)
                spindle = spindle_envelope * np.sin(2 * np.pi * spindle_freq * times[:spindle_duration])
                data[ch, spindle_start:spindle_start+spindle_duration] += spindle

        # Last 10 minutes: N3 (slow wave sleep)
        n3_start = n2_end
        for ch in range(n_channels):
            delta_freq = 1 + np.random.rand() * 2  # 1-3 Hz
            data[ch, n3_start:] = 2 * np.sin(2 * np.pi * delta_freq * times[:n_samples-n3_start])
            data[ch, n3_start:] += 0.3 * np.random.randn(n_samples - n3_start)

        # Scale to microvolts
        data *= 50  # ~50 µV amplitude

        # Create channel names
        ch_names = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
        ch_types = ['eeg'] * n_channels

        # Create info
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create Raw object
        raw = mne.io.RawArray(data, info)

        return raw

    @pytest.fixture
    def sleep_controller(self):
        """Create sleep analysis controller."""
        return SleepAnalysisController()

    def test_controller_initialization(self, sleep_controller):
        """Test that controller initializes properly."""
        assert sleep_controller is not None
        assert hasattr(sleep_controller, 'run_full_sleep_analysis')

    def test_sleep_staging_returns_hypnogram(self, sleep_controller, mock_eeg_data):
        """Test that sleep staging returns a valid hypnogram."""
        results = sleep_controller.run_full_sleep_analysis(mock_eeg_data)

        assert 'hypnogram' in results
        assert 'sleep_stages' in results
        assert 'sleep_efficiency' in results
        assert 'total_sleep_time' in results

        # Check hypnogram properties
        hypnogram = results['hypnogram']
        assert isinstance(hypnogram, list | np.ndarray)
        assert len(hypnogram) == 60  # 30 minutes = 60 epochs of 30 seconds

        # Check that stages are valid (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        assert all(stage in [0, 1, 2, 3, 4] for stage in hypnogram)

    def test_sleep_metrics_calculation(self, sleep_controller, mock_eeg_data):
        """Test sleep metrics calculation."""
        results = sleep_controller.run_full_sleep_analysis(mock_eeg_data)

        # Check sleep efficiency
        efficiency = results['sleep_efficiency']
        assert 0 <= efficiency <= 100

        # Check total sleep time
        tst = results['total_sleep_time']
        assert tst >= 0
        assert tst <= 30  # Can't exceed recording duration

        # Check stage percentages
        stages = results['sleep_stages']
        assert 'wake_pct' in stages
        assert 'n1_pct' in stages
        assert 'n2_pct' in stages
        assert 'n3_pct' in stages
        assert 'rem_pct' in stages

        # Percentages should sum to ~100 (allowing for rounding)
        total_pct = sum([stages[k] for k in stages if k.endswith('_pct')])
        assert 99 <= total_pct <= 101

    def test_sleep_staging_detects_wake(self, sleep_controller):
        """Test that controller correctly identifies wake state."""
        # Create pure wake data (alpha rhythm)
        sfreq = 256
        duration = 5 * 60  # 5 minutes
        times = np.arange(sfreq * duration) / sfreq

        # Generate 10 Hz alpha rhythm
        data = np.sin(2 * np.pi * 10 * times)[np.newaxis, :]
        data += 0.1 * np.random.randn(*data.shape)
        data *= 30  # 30 µV

        info = mne.create_info(['C3'], sfreq=sfreq, ch_types=['eeg'])
        raw = mne.io.RawArray(data, info)

        results = sleep_controller.run_full_sleep_analysis(raw)
        hypnogram = results['hypnogram']

        # Most epochs should be wake (0)
        wake_epochs = sum(1 for stage in hypnogram if stage == 0)
        assert wake_epochs > len(hypnogram) * 0.7  # >70% wake

    def test_sleep_staging_detects_deep_sleep(self, sleep_controller):
        """Test that controller correctly identifies deep sleep."""
        # Create pure N3 data (delta waves)
        sfreq = 256
        duration = 5 * 60  # 5 minutes
        times = np.arange(sfreq * duration) / sfreq

        # Generate 1 Hz delta rhythm
        data = np.sin(2 * np.pi * 1 * times)[np.newaxis, :]
        data += 0.05 * np.random.randn(*data.shape)
        data *= 75  # 75 µV (high amplitude)

        info = mne.create_info(['C3'], sfreq=sfreq, ch_types=['eeg'])
        raw = mne.io.RawArray(data, info)

        results = sleep_controller.run_full_sleep_analysis(raw)
        hypnogram = results['hypnogram']

        # Most epochs should be N3 (3)
        n3_epochs = sum(1 for stage in hypnogram if stage == 3)
        assert n3_epochs > len(hypnogram) * 0.5  # >50% N3

    def test_handles_short_recordings(self, sleep_controller):
        """Test that controller handles recordings shorter than 30 seconds."""
        # Create 20 seconds of data
        sfreq = 256
        duration = 20
        data = np.random.randn(1, sfreq * duration) * 30

        info = mne.create_info(['C3'], sfreq=sfreq, ch_types=['eeg'])
        raw = mne.io.RawArray(data, info)

        results = sleep_controller.run_full_sleep_analysis(raw)

        # Should handle gracefully
        assert 'error' not in results
        assert results['hypnogram'] is not None

    def test_handles_missing_channels(self, sleep_controller):
        """Test that controller handles recordings without standard channels."""
        # Create data with non-standard channel
        sfreq = 256
        duration = 60
        data = np.random.randn(1, sfreq * duration) * 30

        info = mne.create_info(['Fpz'], sfreq=sfreq, ch_types=['eeg'])
        raw = mne.io.RawArray(data, info)

        results = sleep_controller.run_full_sleep_analysis(raw)

        # Should still produce results
        assert 'hypnogram' in results
        assert results['hypnogram'] is not None

    @pytest.mark.parametrize("n_channels,expected_channels", [
        (1, ['C3']),
        (2, ['C3', 'C4']),
        (6, ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']),
    ])
    def test_channel_selection(self, sleep_controller, n_channels, expected_channels):
        """Test that controller selects appropriate channels."""
        # Create data with varying channel counts
        sfreq = 256
        duration = 60
        data = np.random.randn(n_channels, sfreq * duration) * 30

        ch_names = expected_channels[:n_channels]
        info = mne.create_info(ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(data, info)

        # Process and check that it works
        results = sleep_controller.run_full_sleep_analysis(raw)
        assert 'hypnogram' in results
        assert results['processing_info']['channels_used'] == ch_names

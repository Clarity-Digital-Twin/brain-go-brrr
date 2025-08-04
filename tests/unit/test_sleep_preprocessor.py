"""Test sleep-specific preprocessing - TDD style."""

import mne
import numpy as np
import pytest


class TestSleepPreprocessor:
    """Test minimal sleep preprocessing following YASA requirements."""

    @pytest.fixture
    def raw_sleep_data(self):
        """Create mock sleep EEG data similar to Sleep-EDF."""
        # 2 EEG channels like Sleep-EDF
        sfreq = 100.0
        duration = 60.0  # 1 minute
        n_samples = int(sfreq * duration)

        # Create realistic sleep-like signals
        time = np.arange(n_samples) / sfreq

        # Channel 1: Mix of delta (1-4 Hz) and theta (4-8 Hz)
        ch1 = (
            0.5 * np.sin(2 * np.pi * 2 * time)  # 2 Hz delta
            + 0.3 * np.sin(2 * np.pi * 6 * time)  # 6 Hz theta
        ) * 50e-6  # Scale to microvolts

        # Channel 2: Similar but with phase shift
        ch2 = (
            0.5 * np.sin(2 * np.pi * 2 * time + np.pi / 4)
            + 0.3 * np.sin(2 * np.pi * 6 * time + np.pi / 3)
        ) * 50e-6

        data = np.vstack([ch1, ch2])

        # Create info with Sleep-EDF style channel names
        ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        return mne.io.RawArray(data, info)

    def test_basic_preprocessing(self, raw_sleep_data):
        """Test that basic preprocessing works correctly."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        preprocessor = SleepPreprocessor()
        processed = preprocessor.preprocess(raw_sleep_data)

        # Should not modify the original
        assert raw_sleep_data.info["sfreq"] == 100.0
        assert len(raw_sleep_data.ch_names) == 2

        # Should preserve basic properties
        assert processed.info["sfreq"] == 100.0
        assert len(processed.ch_names) == 2

        # Should have applied filters (check that highpass/lowpass are set)
        assert processed.info["highpass"] == 0.3
        assert processed.info["lowpass"] == 35.0

    def test_resampling(self):
        """Test that resampling works when needed."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        # Create 256 Hz data
        sfreq = 256.0
        data = np.random.randn(2, int(sfreq * 10)) * 50e-6
        info = mne.create_info(ch_names=["Fpz", "Pz"], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        preprocessor = SleepPreprocessor(target_sfreq=100.0)
        processed = preprocessor.preprocess(raw)

        # Should resample to 100 Hz
        assert processed.info["sfreq"] == 100.0
        # Should have fewer samples
        assert processed.n_times < raw.n_times

    def test_no_aggressive_filtering(self, raw_sleep_data):
        """Test that preprocessing is minimal - no notch, no aggressive artifact rejection."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        preprocessor = SleepPreprocessor()

        # Add some high amplitude "artifacts" that would trigger rejection
        raw_with_artifacts = raw_sleep_data.copy()
        data = raw_with_artifacts.get_data()
        data[:, 100:200] = 200e-6  # High amplitude

        processed = preprocessor.preprocess(raw_with_artifacts)

        # Should NOT mark any bad channels (no artifact rejection)
        assert len(processed.info["bads"]) == 0

        # Should NOT have annotations for bad segments
        assert len(processed.annotations) == 0

    def test_average_reference(self, raw_sleep_data):
        """Test that average reference is applied."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        preprocessor = SleepPreprocessor(reference="average")
        processed = preprocessor.preprocess(raw_sleep_data)

        # After average referencing, mean across channels should be ~0
        data = processed.get_data()
        channel_mean = np.mean(data, axis=0)
        assert np.abs(channel_mean).max() < 1e-10

    def test_no_reference_option(self, raw_sleep_data):
        """Test that reference can be skipped."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        preprocessor = SleepPreprocessor(reference=None)
        processed = preprocessor.preprocess(raw_sleep_data)

        # Should not apply average reference
        data = processed.get_data()
        channel_mean = np.mean(data, axis=0)
        # Mean should NOT be zero (no re-referencing)
        assert np.abs(channel_mean).max() > 1e-10

    def test_channel_type_setting(self):
        """Test that channel types can be set for YASA."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        # Create data with mixed channel types
        ch_names = ["Fpz", "Pz", "EOG", "EMG"]
        data = np.random.randn(4, 10000) * 50e-6
        info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="misc")
        raw = mne.io.RawArray(data, info)

        preprocessor = SleepPreprocessor()
        processed = preprocessor.preprocess_for_yasa(
            raw, eeg_channels=["Fpz", "Pz"], eog_channels=["EOG"], emg_channels=["EMG"]
        )

        # Check channel types are set correctly
        assert processed.get_channel_types() == ["eeg", "eeg", "eog", "emg"]

    def test_filter_parameters(self):
        """Test custom filter parameters."""
        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        # Test with YASA's exact parameters
        preprocessor = SleepPreprocessor(l_freq=0.4, h_freq=30.0)

        assert preprocessor.l_freq == 0.4
        assert preprocessor.h_freq == 30.0

        # Create simple data
        data = np.random.randn(1, 10000) * 50e-6
        info = mne.create_info(ch_names=["C3"], sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        processed = preprocessor.preprocess(raw)

        # Check filter was applied
        assert processed.info["highpass"] == 0.4
        assert processed.info["lowpass"] == 30.0

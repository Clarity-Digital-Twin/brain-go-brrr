"""Test flexible preprocessing that handles various EEG data formats."""

import mne
import numpy as np
import pytest

from brain_go_brrr.preprocessing.flexible_preprocessor import FlexibleEEGPreprocessor


class TestFlexibleEEGPreprocessor:
    """Test flexible preprocessing for heterogeneous EEG data."""

    @pytest.fixture
    def sleep_edf_raw(self):
        """Create mock Sleep-EDF data without channel positions."""
        # Sleep-EDF typical setup: 100 Hz, non-standard channel names
        sfreq = 100
        duration = 60  # 1 minute
        ch_names = [
            "EEG Fpz-Cz",
            "EEG Pz-Oz",
            "EOG horizontal",
            "Resp oro-nasal",
            "EMG submental",
            "Temp rectal",
            "Event marker",
        ]
        ch_types = ["eeg", "eeg", "eog", "resp", "emg", "misc", "misc"]

        n_times = int(sfreq * duration)
        data = np.random.randn(len(ch_names), n_times) * 50e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Explicitly no montage (like real Sleep-EDF files)
        assert raw.get_montage() is None

        return raw

    @pytest.fixture
    def tuh_eeg_raw(self):
        """Create mock TUH EEG data with standard channel names."""
        # TUH typical setup: 250 Hz, standard 10-20 names
        sfreq = 250
        duration = 30  # 30 seconds
        ch_names = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
        ]

        n_times = int(sfreq * duration)
        data = np.random.randn(len(ch_names), n_times) * 30e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Add standard montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False)

        return raw

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization with different modes."""
        # Default mode
        preprocessor = FlexibleEEGPreprocessor()
        assert preprocessor.mode == "auto"
        assert preprocessor.require_positions is False

        # Strict mode (for when positions are needed)
        strict_preprocessor = FlexibleEEGPreprocessor(mode="strict", require_positions=True)
        assert strict_preprocessor.require_positions is True

    def test_preprocess_sleep_edf_without_positions(self, sleep_edf_raw):
        """Test preprocessing Sleep-EDF data without channel positions."""
        preprocessor = FlexibleEEGPreprocessor(
            mode="sleep",
            target_sfreq=100,  # Keep original sampling rate
            use_autoreject=False,  # Disable since no positions
        )

        # Should succeed without positions
        processed = preprocessor.preprocess(sleep_edf_raw.copy())

        # Check basic properties
        assert processed.info["sfreq"] == 100
        # In sleep mode, we keep EEG + EOG + EMG channels (2 + 1 + 1 = 4)
        assert len(processed.ch_names) == 4
        # Check we have the expected channel types
        ch_types = processed.get_channel_types()
        assert ch_types.count("eeg") == 2
        assert ch_types.count("eog") == 1
        assert ch_types.count("emg") == 1

    def test_preprocess_tuh_with_positions(self, tuh_eeg_raw):
        """Test preprocessing TUH data with channel positions."""
        preprocessor = FlexibleEEGPreprocessor(
            mode="abnormality",
            target_sfreq=256,  # EEGPT requirement
            use_autoreject=True,
        )

        # Should succeed with positions
        processed = preprocessor.preprocess(tuh_eeg_raw.copy())

        # Check properties
        assert processed.info["sfreq"] == 256
        assert processed.get_montage() is not None
        assert len(processed.ch_names) <= 19  # May have removed bad channels

    def test_channel_name_mapping(self, sleep_edf_raw):
        """Test automatic channel name mapping."""
        preprocessor = FlexibleEEGPreprocessor()

        # Test mapping function
        mapped_names = preprocessor._map_channel_names(sleep_edf_raw.ch_names)

        assert mapped_names["EEG Fpz-Cz"] == "Fpz"
        assert mapped_names["EEG Pz-Oz"] == "Pz"
        assert mapped_names["EOG horizontal"] == "EOG"

    def test_smart_channel_selection(self, tuh_eeg_raw):
        """Test intelligent channel selection for different tasks."""
        # For sleep analysis - prefer C3, C4
        sleep_preprocessor = FlexibleEEGPreprocessor(mode="sleep")
        sleep_channels = sleep_preprocessor._select_channels_for_task(
            tuh_eeg_raw.ch_names, task="sleep"
        )
        assert "C3" in sleep_channels
        assert "C4" in sleep_channels

        # For abnormality - broader selection
        abnormal_preprocessor = FlexibleEEGPreprocessor(mode="abnormality")
        abnormal_channels = abnormal_preprocessor._select_channels_for_task(
            tuh_eeg_raw.ch_names, task="abnormality"
        )
        assert len(abnormal_channels) >= 10

    def test_fallback_artifact_rejection(self, sleep_edf_raw):
        """Test fallback artifact rejection when Autoreject unavailable."""
        preprocessor = FlexibleEEGPreprocessor(
            use_autoreject=True  # Will fallback since no positions
        )

        # Add some artifacts
        raw = sleep_edf_raw.copy()
        raw._data[0, 1000:1100] = 500e-6  # Large artifact

        processed = preprocessor.preprocess(raw)

        # Should have applied some artifact rejection
        assert processed._data.max() < 500e-6

    def test_resampling_handling(self, sleep_edf_raw, tuh_eeg_raw):
        """Test proper resampling for different sampling rates."""
        preprocessor = FlexibleEEGPreprocessor(target_sfreq=256)

        # From 100 Hz to 256 Hz (upsampling)
        processed_sleep = preprocessor.preprocess(sleep_edf_raw.copy())
        assert processed_sleep.info["sfreq"] == 256

        # From 250 Hz to 256 Hz (slight upsampling)
        processed_tuh = preprocessor.preprocess(tuh_eeg_raw.copy())
        assert processed_tuh.info["sfreq"] == 256

    def test_preprocessing_modes(self):
        """Test different preprocessing modes."""
        modes = ["auto", "sleep", "abnormality", "event_detection", "minimal"]

        for mode in modes:
            preprocessor = FlexibleEEGPreprocessor(mode=mode)
            assert preprocessor.mode == mode

            # Each mode should have different settings
            if mode == "sleep":
                assert preprocessor.lowpass_freq == 35  # Sleep typically uses lower
            elif mode == "abnormality":
                assert preprocessor.lowpass_freq == 45  # Abnormality uses higher

    def test_preprocessing_pipeline_order(self, tuh_eeg_raw):
        """Test that preprocessing steps happen in correct order."""
        preprocessor = FlexibleEEGPreprocessor()

        # Track which steps were called
        steps_called = []

        # Monkey-patch methods to track calls
        original_filter = preprocessor._apply_filters
        original_resample = preprocessor._resample

        def track_filter(raw):
            steps_called.append("filter")
            return original_filter(raw)

        def track_resample(raw):
            steps_called.append("resample")
            return original_resample(raw)

        preprocessor._apply_filters = track_filter
        preprocessor._resample = track_resample

        preprocessor.preprocess(tuh_eeg_raw.copy())

        # Resample should come after filtering
        assert steps_called.index("filter") < steps_called.index("resample")

    def test_error_handling(self):
        """Test graceful error handling."""
        preprocessor = FlexibleEEGPreprocessor(require_positions=True)

        # Create data without positions
        raw = mne.io.RawArray(
            np.random.randn(1, 1000) * 1e-6, mne.create_info(["Ch1"], sfreq=100, ch_types="eeg")
        )

        # Should raise informative error in strict mode
        with pytest.raises(ValueError, match="Channel positions required"):
            preprocessor.preprocess(raw)

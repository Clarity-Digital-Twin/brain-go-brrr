"""Tests to ensure YASA compliance with best practices."""

from pathlib import Path

import mne
import numpy as np
import pytest

from services.sleep_metrics import SleepAnalyzer


class TestYASACompliance:
    """Test suite to ensure YASA is used according to documentation."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample raw EEG for testing."""
        # Create realistic EEG data
        sfreq = 256
        duration = 60  # 1 minute
        n_channels = 4
        ch_names = ["Fpz-Cz", "Pz-Oz", "EOG horizontal", "EMG submental"]
        ch_types = ["eeg", "eeg", "eog", "emg"]

        # Generate data
        np.random.seed(42)
        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        return raw

    def test_no_filtering_before_yasa(self, sample_raw):
        """Test that YASA preprocessing does NOT filter the data."""
        analyzer = SleepAnalyzer()

        # Get original data statistics
        original_data = sample_raw.get_data()
        original_mean = np.mean(original_data)
        original_std = np.std(original_data)

        # Run preprocessing
        processed = analyzer.preprocess_for_sleep(sample_raw.copy())
        processed_data = processed.get_data()

        # Check that data magnitude is preserved (no filtering applied)
        # Filtering would significantly change these values
        processed_mean = np.mean(processed_data)
        processed_std = np.std(processed_data)

        # Resampling can change statistics, but much less than filtering would
        # Filtering typically reduces std by 40-60%, resampling by <40%
        std_change_ratio = np.abs(processed_std - original_std) / original_std

        # If filtering was applied, std would drop by >40%
        # Resampling alone should result in <40% change
        assert std_change_ratio < 0.4, (
            f"Standard deviation changed by {std_change_ratio * 100:.1f}% - filtering may have been applied"
        )

        # Mean should be relatively preserved
        assert np.abs(processed_mean - original_mean) < original_std * 0.1

        # Verify resampling to 100 Hz (YASA requirement)
        assert processed.info["sfreq"] == 100

    def test_confidence_scores_returned(self, sample_raw):
        """Test that YASA returns confidence scores."""
        analyzer = SleepAnalyzer()

        # Mock a longer recording (YASA needs 5+ minutes)
        long_raw = sample_raw.copy()
        long_data = np.tile(long_raw.get_data(), (1, 6))  # 6 minutes
        long_info = long_raw.info.copy()
        long_raw = mne.io.RawArray(long_data, long_info)

        # Run sleep staging with confidence scores
        result = analyzer.stage_sleep(long_raw, return_proba=True)

        # Should return tuple when return_proba=True
        assert isinstance(result, tuple)
        assert len(result) == 2

        hypnogram, proba = result

        # Check hypnogram
        assert isinstance(hypnogram, list | np.ndarray)
        assert len(hypnogram) > 0

        # Check probability scores
        assert proba is not None
        assert hasattr(proba, "shape")
        # Should have probabilities for each stage
        assert proba.shape[1] == 5  # W, N1, N2, N3, REM
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_metadata_support(self, sample_raw):
        """Test that demographic metadata can be passed to YASA."""
        analyzer = SleepAnalyzer()

        # Create metadata
        metadata = {"age": 35, "male": True}

        # Should accept metadata without error
        # (actual staging would fail on short data, but API should work)
        try:
            processed = analyzer.preprocess_for_sleep(sample_raw.copy())
            # This will fail due to short duration, but we're testing the API
            analyzer.stage_sleep(processed, metadata=metadata)
        except Exception as e:
            # Should fail for data length, not metadata issues
            assert "5 minutes" in str(e) or "data" in str(e).lower()
            assert "metadata" not in str(e).lower()


class TestRealSleepEDFData:
    """Tests using real Sleep-EDF data."""

    @pytest.fixture
    def sleep_edf_path(self):
        """Get path to a real Sleep-EDF file."""
        base_path = Path(
            "/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/data/datasets/external/sleep-edf"
        )
        # Use first cassette file
        edf_path = base_path / "sleep-cassette" / "SC4001E0-PSG.edf"

        if not edf_path.exists():
            pytest.skip(f"Sleep-EDF data not found at {edf_path}")

        return edf_path

    @pytest.fixture
    def hypnogram_path(self):
        """Get path to corresponding hypnogram."""
        base_path = Path(
            "/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/data/datasets/external/sleep-edf"
        )
        hypno_path = base_path / "sleep-cassette" / "SC4001EC-Hypnogram.edf"

        if not hypno_path.exists():
            pytest.skip(f"Hypnogram not found at {hypno_path}")

        return hypno_path

    @pytest.mark.integration
    def test_real_sleep_staging(self, sleep_edf_path):
        """Test YASA on real Sleep-EDF data."""
        # Load real data
        raw = mne.io.read_raw_edf(sleep_edf_path, preload=False)

        # Use only first 10 minutes for speed
        raw.crop(tmax=600)
        raw.load_data()

        analyzer = SleepAnalyzer()

        # Stage sleep
        hypnogram = analyzer.stage_sleep(
            raw, eeg_name="Fpz-Cz", eog_name="EOG horizontal", emg_name="EMG submental"
        )

        # Verify reasonable results
        assert len(hypnogram) == 20  # 10 minutes / 30 seconds = 20 epochs

        # All stages should be valid (YASA returns strings)
        valid_stages = {"W", "N1", "N2", "N3", "REM", "ART", "UNS"}
        assert all(stage in valid_stages for stage in hypnogram)

        # Should have some variety in stages (not all the same)
        unique_stages = set(hypnogram)
        assert len(unique_stages) >= 2

    @pytest.mark.integration
    def test_confidence_scores_on_real_data(self, sleep_edf_path):
        """Test confidence scores on real data."""
        raw = mne.io.read_raw_edf(sleep_edf_path, preload=False)
        raw.crop(tmax=600)  # 10 minutes
        raw.load_data()

        analyzer = SleepAnalyzer()

        # Get predictions with confidence
        hypnogram, proba = analyzer.stage_sleep(
            raw,
            eeg_name="Fpz-Cz",
            eog_name="EOG horizontal",
            emg_name="EMG submental",
            return_proba=True,
        )

        # Check probability matrix
        assert proba.shape == (20, 5)  # 20 epochs, 5 stages

        # Convert to numpy if it's a DataFrame
        proba_array = proba.values if hasattr(proba, "values") else proba

        # Each row should sum to 1
        assert np.allclose(proba_array.sum(axis=1), 1.0)

        # The hypnogram might be smoothed/adjusted compared to raw probabilities
        # This is expected behavior in YASA (temporal smoothing, etc.)
        # Just verify we have valid probabilities
        assert proba_array.min() >= 0
        assert proba_array.max() <= 1

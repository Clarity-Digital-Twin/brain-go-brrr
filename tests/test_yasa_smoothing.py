"""Test YASA temporal smoothing functionality."""

import numpy as np
import pytest

from core.sleep import SleepAnalyzer


class TestYASASmoothing:
    """Test temporal smoothing for YASA sleep staging."""

    @pytest.fixture
    def sample_hypnogram(self):
        """Create a sample hypnogram with some noise."""
        # Create a hypnogram with clear transitions and some noise
        # 20 epochs of Wake, 20 of N2, then some noisy transitions
        base_hypno = (
            ["W"] * 20  # 10 minutes wake
            + ["N2"] * 20  # 10 minutes N2
            + ["N3"] * 20  # 10 minutes N3
            + ["N2", "N3", "N2", "N3", "N2"]  # Noisy transitions
            + ["REM"] * 15  # 7.5 minutes REM
        )
        return np.array(base_hypno)

    def test_smoothing_removes_short_transitions(self, sample_hypnogram):
        """Test that smoothing removes unrealistic short transitions."""
        analyzer = SleepAnalyzer()

        # Apply smoothing
        smoothed = analyzer._smooth_hypnogram(sample_hypnogram, window_min=7.5)

        # Check that short N2/N3 transitions are smoothed out
        # The noisy section should become mostly one stage
        noisy_section = smoothed[60:65]
        unique_stages = np.unique(noisy_section)

        # Should have fewer unique stages after smoothing
        assert len(unique_stages) <= 2, f"Too many stages in smoothed section: {unique_stages}"

    def test_smoothing_preserves_long_periods(self, sample_hypnogram):
        """Test that smoothing preserves long stable periods."""
        analyzer = SleepAnalyzer()

        smoothed = analyzer._smooth_hypnogram(sample_hypnogram, window_min=7.5)

        # Long stable periods should be preserved
        assert all(s == "W" for s in smoothed[5:15])  # Middle of wake period
        assert all(s == "N2" for s in smoothed[25:35])  # Middle of N2 period
        assert all(s == "N3" for s in smoothed[45:55])  # Middle of N3 period

    def test_smoothing_window_size(self):
        """Test different smoothing window sizes."""
        analyzer = SleepAnalyzer()

        # Create alternating pattern
        hypno = np.array(["W", "N2"] * 30)  # 30 minutes of alternating stages

        # Small window (3 minutes) - less smoothing
        smoothed_3min = analyzer._smooth_hypnogram(hypno, window_min=3.0)
        transitions_3min = np.sum(smoothed_3min[:-1] != smoothed_3min[1:])

        # Large window (7.5 minutes) - more smoothing
        smoothed_7min = analyzer._smooth_hypnogram(hypno, window_min=7.5)
        transitions_7min = np.sum(smoothed_7min[:-1] != smoothed_7min[1:])

        # Larger window should produce fewer transitions
        assert transitions_7min < transitions_3min

    def test_smoothing_edge_handling(self):
        """Test that smoothing handles edges correctly."""
        analyzer = SleepAnalyzer()

        # Short hypnogram
        hypno = np.array(["W"] * 5 + ["N2"] * 5 + ["N3"] * 5)

        smoothed = analyzer._smooth_hypnogram(hypno, window_min=3.0)

        # Should not crash and should preserve length
        assert len(smoothed) == len(hypno)

        # First and last elements should be preserved
        assert smoothed[0] == "W"
        assert smoothed[-1] == "N3"

    def test_stage_sleep_with_smoothing(self):
        """Test full sleep staging with smoothing enabled."""
        import mne

        # Create 10 minutes of synthetic data
        sfreq = 256
        duration = 600
        n_samples = int(sfreq * duration)

        data = np.random.randn(1, n_samples) * 50e-6
        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(data, info)

        analyzer = SleepAnalyzer()

        # Stage without smoothing
        hypno_raw = analyzer.stage_sleep(raw, apply_smoothing=False)

        # Stage with smoothing
        hypno_smooth = analyzer.stage_sleep(raw, apply_smoothing=True)

        # Both should return valid hypnograms
        assert len(hypno_raw) == 20  # 10 min / 30 sec epochs
        assert len(hypno_smooth) == 20

        # Smoothed should have fewer or equal transitions
        transitions_raw = np.sum(hypno_raw[:-1] != hypno_raw[1:])
        transitions_smooth = np.sum(hypno_smooth[:-1] != hypno_smooth[1:])
        assert transitions_smooth <= transitions_raw

    def test_smoothing_with_confidence_scores(self):
        """Test that smoothing works with confidence score output."""
        import mne

        # Create synthetic data
        sfreq = 256
        duration = 600
        n_samples = int(sfreq * duration)

        data = np.random.randn(1, n_samples) * 50e-6
        info = mne.create_info(["C3"], sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(data, info)

        analyzer = SleepAnalyzer()

        # Get predictions with confidence and smoothing
        hypno, proba = analyzer.stage_sleep(raw, return_proba=True, apply_smoothing=True)

        # Should return both hypnogram and probabilities
        assert len(hypno) == 20
        assert proba.shape == (20, 5)

        # Probabilities should still be valid
        proba_array = proba.values if hasattr(proba, "values") else proba
        assert np.allclose(proba_array.sum(axis=1), 1.0)

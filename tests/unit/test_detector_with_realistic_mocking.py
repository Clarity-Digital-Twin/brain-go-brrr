"""Test abnormality detector with realistic EEGPT mocking."""

import mne
import numpy as np
import pytest

from services.abnormality_detector import TriageLevel
from tests.fixtures.mock_eegpt import create_mock_detector_with_realistic_model


class TestDetectorWithRealisticMocking:
    """Test abnormality detector using realistic EEGPT mock."""

    @pytest.fixture
    def detector(self):
        """Create detector with realistic mocking."""
        return create_mock_detector_with_realistic_model()

    @pytest.fixture
    def mock_eeg_data(self):
        """Create mock EEG data for testing."""
        sfreq = 256
        duration = 60  # 1 minute for faster tests
        n_channels = 19
        n_samples = int(sfreq * duration)

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

        # Generate realistic EEG data (10-50 μV range)
        data = np.random.randn(n_channels, n_samples) * 20e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        return raw

    def test_normal_eeg_classification(self, detector, mock_eeg_data):
        """Test that normal EEG is classified correctly."""
        # Create normal EEG (low amplitude, no artifacts)
        normal_data = mock_eeg_data.get_data() * 0.5  # Reduce amplitude
        normal_raw = mne.io.RawArray(normal_data, mock_eeg_data.info)

        result = detector.detect_abnormality(normal_raw)

        # With our mock, normal EEG should have low abnormality score
        assert result.abnormality_score < 0.6
        assert result.classification == "normal"
        assert result.triage_flag in [TriageLevel.NORMAL, TriageLevel.ROUTINE]

    def test_high_amplitude_artifact_detection(self, detector, mock_eeg_data):
        """Test that high amplitude artifacts are detected."""
        # Create EEG with high amplitude artifacts
        artifact_data = mock_eeg_data.get_data().copy()
        # Add high amplitude spikes to some channels
        artifact_data[0, 1000:1100] = 200e-6  # 200 μV spike
        artifact_data[5, 2000:2100] = -150e-6  # Negative spike

        artifact_raw = mne.io.RawArray(artifact_data, mock_eeg_data.info)

        result = detector.detect_abnormality(artifact_raw)

        # High amplitude artifacts should increase abnormality score
        assert result.abnormality_score > 0.4
        # Quality metrics should detect artifacts
        assert result.quality_metrics["artifacts_detected"] > 0

    def test_window_level_predictions_consistency(self, detector, mock_eeg_data):
        """Test that window-level predictions are consistent with final score."""
        result = detector.detect_abnormality(mock_eeg_data)

        # Extract window scores
        window_scores = [w.abnormality_score for w in result.window_scores]

        # Final score should be related to window scores
        mean_window_score = np.mean(window_scores)

        # Final score should be close to mean of window scores
        # (depending on aggregation method and quality weighting)
        assert abs(result.abnormality_score - mean_window_score) < 0.3

        # All window scores should be valid probabilities
        assert all(0 <= score <= 1 for score in window_scores)

    def test_different_eeg_patterns_produce_different_scores(self, detector):
        """Test that different EEG patterns produce meaningfully different scores."""
        sfreq = 256
        duration = 60
        n_channels = 19

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
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Pattern 1: Normal EEG
        normal_data = np.random.randn(n_channels, sfreq * duration) * 20e-6
        normal_raw = mne.io.RawArray(normal_data, info)
        score_normal = detector.detect_abnormality(normal_raw).abnormality_score

        # Pattern 2: High frequency noise
        noisy_data = np.random.randn(n_channels, sfreq * duration) * 50e-6
        # Add high frequency component
        for i in range(n_channels):
            noisy_data[i, :] += np.sin(2 * np.pi * 30 * np.arange(sfreq * duration) / sfreq) * 10e-6
        noisy_raw = mne.io.RawArray(noisy_data, info)
        score_noisy = detector.detect_abnormality(noisy_raw).abnormality_score

        # Pattern 3: Slow waves (simulated)
        slow_data = np.random.randn(n_channels, sfreq * duration) * 10e-6
        # Add slow wave component
        for i in range(n_channels):
            slow_data[i, :] += np.sin(2 * np.pi * 2 * np.arange(sfreq * duration) / sfreq) * 30e-6
        slow_raw = mne.io.RawArray(slow_data, info)
        score_slow = detector.detect_abnormality(slow_raw).abnormality_score

        # Scores should be meaningfully different
        scores = [score_normal, score_noisy, score_slow]

        # At least some scores should differ by more than 0.1
        score_diffs = [
            abs(scores[i] - scores[j])
            for i in range(len(scores))
            for j in range(i + 1, len(scores))
        ]
        assert max(score_diffs) > 0.1

    def test_confidence_reflects_consistency(self, detector):
        """Test that confidence score reflects consistency of predictions."""
        sfreq = 256
        duration = 60
        n_channels = 19

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
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Create consistent abnormal pattern
        consistent_data = (
            np.random.randn(n_channels, sfreq * duration) * 80e-6
        )  # High amplitude throughout
        consistent_raw = mne.io.RawArray(consistent_data, info)
        result_consistent = detector.detect_abnormality(consistent_raw)

        # Create inconsistent pattern (alternating normal and abnormal)
        inconsistent_data = np.random.randn(n_channels, sfreq * duration) * 20e-6
        # Make every other 10 seconds high amplitude
        for i in range(0, duration, 20):
            start = i * sfreq
            end = min((i + 10) * sfreq, inconsistent_data.shape[1])
            inconsistent_data[:, start:end] *= 4  # Increase amplitude

        inconsistent_raw = mne.io.RawArray(inconsistent_data, info)
        result_inconsistent = detector.detect_abnormality(inconsistent_raw)

        # Consistent pattern should have higher confidence
        assert result_consistent.confidence > result_inconsistent.confidence

"""Unit tests for EEG Abnormality Detection Service.

Following TDD approach - write tests first based on specifications.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from services.abnormality_detector import (
    AbnormalityDetector,
    AbnormalityResult,
    AggregationMethod,
    TriageLevel,
    WindowResult,
)


class TestAbnormalityDetector:
    """Test suite for AbnormalityDetector service."""

    @pytest.fixture
    def mock_eeg_data(self):
        """Create mock EEG data for testing."""
        # 19 channels, 20 minutes at 256 Hz
        sfreq = 256
        duration = 20 * 60  # 20 minutes
        n_channels = 19
        n_samples = int(sfreq * duration)

        # Standard 10-20 channel names
        ch_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]

        # Generate realistic EEG data (10-50 μV range)
        data = np.random.randn(n_channels, n_samples) * 20e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        return raw

    @pytest.fixture
    def mock_eegpt_model(self):
        """Mock EEGPT model for testing."""
        model = MagicMock()
        # Mock extract_features to return 512-dim embeddings
        model.extract_features.return_value = np.random.randn(1, 512).astype(np.float32)
        model.is_loaded = True
        return model

    @pytest.fixture
    def detector(self, mock_eegpt_model):
        """Create detector instance with mocked model."""
        with patch('services.abnormality_detector.EEGPTModel') as mock_model_class, \
             patch('services.abnormality_detector.ModelConfig'):
            mock_model_class.return_value = mock_eegpt_model
            detector = AbnormalityDetector(
                model_path=Path("fake/path.ckpt"),
                device="cpu"
            )
            detector.model = mock_eegpt_model
            return detector

    def test_detector_initialization(self):
        """Test detector initializes with correct parameters."""
        with patch('services.abnormality_detector.EEGPTModel'), \
             patch('services.abnormality_detector.ModelConfig'):
            detector = AbnormalityDetector(
                model_path=Path("fake/path.ckpt"),
                device="cpu",
                window_duration=4.0,
                overlap_ratio=0.5
            )

            assert detector.window_duration == 4.0
            assert detector.overlap_ratio == 0.5
            assert detector.target_sfreq == 256
            assert detector.device == "cpu"

    def test_preprocessing_pipeline(self, detector, mock_eeg_data):
        """Test preprocessing applies correct filters and normalization."""
        preprocessed = detector._preprocess_eeg(mock_eeg_data)

        # Check sampling rate is correct
        assert preprocessed.info['sfreq'] == 256

        # Check all channels present
        assert len(preprocessed.ch_names) == 19

        # Check data is normalized (roughly zero mean, unit variance)
        data = preprocessed.get_data()
        assert np.abs(data.mean()) < 0.1
        assert 0.8 < data.std() < 1.2

    def test_window_extraction(self, detector, mock_eeg_data):
        """Test sliding window extraction with overlap."""
        windows = detector._extract_windows(mock_eeg_data)

        # Calculate expected number of windows
        window_samples = int(detector.window_duration * mock_eeg_data.info['sfreq'])
        step_samples = int(window_samples * (1 - detector.overlap_ratio))
        expected_windows = int((len(mock_eeg_data.times) - window_samples) / step_samples) + 1

        assert len(windows) == pytest.approx(expected_windows, abs=1)

        # Check window dimensions
        assert all(w.shape == (19, window_samples) for w in windows)

    def test_window_quality_assessment(self, detector):
        """Test quality scoring for individual windows."""
        # Create window with artifacts
        window_good = np.random.randn(19, 1024) * 20e-6
        window_bad = np.random.randn(19, 1024) * 100e-6  # High amplitude
        window_bad[0, :] = 1000e-6  # Saturated channel

        quality_good = detector._assess_window_quality(window_good)
        quality_bad = detector._assess_window_quality(window_bad)

        assert 0.8 < quality_good <= 1.0
        assert 0.0 <= quality_bad < 0.5

    def test_single_window_prediction(self, detector, mock_eegpt_model):
        """Test abnormality prediction for a single window."""
        window = np.random.randn(19, 1024).astype(np.float32)

        # Mock model to return high abnormality score (2D predictions)
        mock_eegpt_model.extract_features.return_value = np.array([[0.2, 0.8]])  # [normal, abnormal]

        # Set the model to use the mock
        detector.model = mock_eegpt_model

        score = detector._predict_window(window)

        assert 0.0 <= score <= 1.0
        assert score == 0.8  # Should return abnormal probability
        mock_eegpt_model.extract_features.assert_called_once()

    def test_aggregation_methods(self, detector):
        """Test different aggregation strategies."""
        window_scores = [0.3, 0.8, 0.6, 0.9, 0.4, 0.7]
        quality_scores = [0.9, 0.8, 0.7, 0.6, 0.9, 0.8]

        # Test weighted average
        avg_score = detector._aggregate_scores(
            window_scores,
            quality_scores,
            method=AggregationMethod.WEIGHTED_AVERAGE
        )
        assert 0.0 <= avg_score <= 1.0

        # Test voting
        vote_score = detector._aggregate_scores(
            window_scores,
            quality_scores,
            method=AggregationMethod.VOTING
        )
        assert vote_score in [0.0, 1.0] or 0.0 <= vote_score <= 1.0

        # Test attention (mock for now)
        attn_score = detector._aggregate_scores(
            window_scores,
            quality_scores,
            method=AggregationMethod.ATTENTION
        )
        assert 0.0 <= attn_score <= 1.0

    def test_triage_decision_logic(self, detector):
        """Test triage flag assignment based on scores."""
        # Test URGENT
        result = detector._determine_triage(
            abnormality_score=0.85,
            quality_grade="GOOD"
        )
        assert result == TriageLevel.URGENT

        # Test URGENT due to poor quality
        result = detector._determine_triage(
            abnormality_score=0.5,
            quality_grade="POOR"
        )
        assert result == TriageLevel.URGENT

        # Test EXPEDITE
        result = detector._determine_triage(
            abnormality_score=0.65,
            quality_grade="GOOD"
        )
        assert result == TriageLevel.EXPEDITE

        # Test ROUTINE
        result = detector._determine_triage(
            abnormality_score=0.45,
            quality_grade="GOOD"
        )
        assert result == TriageLevel.ROUTINE

        # Test NORMAL
        result = detector._determine_triage(
            abnormality_score=0.2,
            quality_grade="EXCELLENT"
        )
        assert result == TriageLevel.NORMAL

    def test_confidence_calculation(self, detector):
        """Test confidence score calculation."""
        # High agreement between windows = high confidence
        window_scores = [0.8, 0.82, 0.79, 0.81, 0.78]
        confidence = detector._calculate_confidence(window_scores)
        assert confidence > 0.8

        # High disagreement = low confidence
        window_scores = [0.2, 0.8, 0.3, 0.9, 0.5]
        confidence = detector._calculate_confidence(window_scores)
        assert confidence < 0.5

    def test_full_pipeline_normal_eeg(self, detector, mock_eeg_data, mock_eegpt_model):
        """Test full pipeline on normal EEG."""
        # Mock model to return low abnormality scores (2D predictions)
        mock_eegpt_model.extract_features.return_value = np.array([[0.8, 0.2]])  # [normal, abnormal]

        result = detector.detect_abnormality(mock_eeg_data)

        assert isinstance(result, AbnormalityResult)
        assert result.classification == "normal"
        assert result.abnormality_score < 0.5
        assert result.triage_flag == TriageLevel.NORMAL
        assert result.confidence > 0.5
        assert result.processing_time > 0
        assert len(result.window_scores) > 0

    def test_full_pipeline_abnormal_eeg(self, detector, mock_eeg_data, mock_eegpt_model):
        """Test full pipeline on abnormal EEG."""
        # Mock model to return high abnormality scores (2D predictions)
        mock_eegpt_model.extract_features.return_value = np.array([[0.15, 0.85]])  # [normal, abnormal]

        result = detector.detect_abnormality(mock_eeg_data)

        assert result.classification == "abnormal"
        assert result.abnormality_score > 0.8
        assert result.triage_flag == TriageLevel.URGENT
        assert result.confidence > 0.5

    def test_edge_case_short_recording(self, detector, mock_eegpt_model):
        """Test handling of recordings shorter than minimum duration."""
        # Create 30-second recording (too short)
        short_data = np.random.randn(19, 256 * 30) * 20e-6
        ch_names = ['Fp1'] * 19  # Simplified for test
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        short_raw = mne.io.RawArray(short_data, info)

        with pytest.raises(ValueError, match="Recording too short"):
            detector.detect_abnormality(short_raw)

    def test_edge_case_bad_channels(self, detector, mock_eegpt_model):
        """Test handling of recordings with many bad channels."""
        # Create data with 50% bad channels
        bad_data = np.random.randn(19, 256 * 60) * 20e-6
        bad_data[:10, :] = 0  # Flat channels

        ch_names = ['Fp1'] * 19
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
        bad_raw = mne.io.RawArray(bad_data, info)

        result = detector.detect_abnormality(bad_raw)

        # Should still process but flag poor quality
        assert result.quality_metrics['quality_grade'] == 'POOR'
        assert result.triage_flag == TriageLevel.URGENT

    def test_batch_processing(self, detector, mock_eeg_data, mock_eegpt_model):
        """Test batch processing of multiple recordings."""
        recordings = [mock_eeg_data.copy() for _ in range(3)]

        results = detector.detect_abnormality_batch(recordings)

        assert len(results) == 3
        assert all(isinstance(r, AbnormalityResult) for r in results)

    def test_gpu_cpu_fallback(self):
        """Test graceful fallback from GPU to CPU."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('services.abnormality_detector.EEGPTModel'), \
             patch('services.abnormality_detector.ModelConfig'):
            detector = AbnormalityDetector(
                model_path=Path("fake/path.ckpt"),
                device="cuda"  # Request GPU
            )

            # Should fallback to CPU
            assert detector.device == "cpu"

    def test_result_serialization(self, detector, mock_eeg_data, mock_eegpt_model):
        """Test result can be serialized to JSON."""
        result = detector.detect_abnormality(mock_eeg_data)

        # Should be able to convert to dict
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'abnormality_score' in result_dict
        assert 'classification' in result_dict
        assert 'triage_flag' in result_dict
        assert 'confidence' in result_dict
        assert 'window_scores' in result_dict
        assert 'quality_metrics' in result_dict

    @pytest.mark.parametrize("sfreq,expected_sfreq", [
        (250, 256),
        (256, 256),
        (500, 256),
        (128, 256)
    ])
    def test_resampling_handling(self, detector, sfreq, expected_sfreq):
        """Test proper resampling of different sampling rates."""
        data = np.random.randn(19, sfreq * 60) * 20e-6
        ch_names = ['Fp1'] * 19
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        preprocessed = detector._preprocess_eeg(raw)

        assert preprocessed.info['sfreq'] == expected_sfreq

    def test_performance_requirements(self, detector, mock_eeg_data, mock_eegpt_model):
        """Test processing completes within time requirements."""
        import time

        start = time.time()
        result = detector.detect_abnormality(mock_eeg_data)
        elapsed = time.time() - start

        # Should complete within 30 seconds for 20-minute recording
        assert elapsed < 30.0
        assert result.processing_time == pytest.approx(elapsed, rel=0.1)

    def test_model_versioning(self, detector):
        """Test model version is tracked in results."""
        assert hasattr(detector, 'model_version')
        assert detector.model_version == "eegpt-v1.0"  # Default version

    def test_concurrent_processing_safety(self, detector, mock_eeg_data, mock_eegpt_model):
        """Test thread safety for concurrent processing."""
        import concurrent.futures

        def process_recording():
            return detector.detect_abnormality(mock_eeg_data.copy())

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_recording) for _ in range(3)]
            results = [f.result() for f in futures]

        assert len(results) == 3
        assert all(isinstance(r, AbnormalityResult) for r in results)


class TestWindowResult:
    """Test WindowResult data class."""

    def test_window_result_creation(self):
        """Test creating WindowResult instance."""
        result = WindowResult(
            index=0,
            start_time=0.0,
            end_time=4.0,
            abnormality_score=0.75,
            quality_score=0.9
        )

        assert result.index == 0
        assert result.start_time == 0.0
        assert result.end_time == 4.0
        assert result.abnormality_score == 0.75
        assert result.quality_score == 0.9


class TestAbnormalityResult:
    """Test AbnormalityResult data class."""

    def test_abnormality_result_creation(self):
        """Test creating AbnormalityResult instance."""
        result = AbnormalityResult(
            abnormality_score=0.75,
            classification="abnormal",
            confidence=0.85,
            triage_flag=TriageLevel.EXPEDITE,
            window_scores=[0.7, 0.8, 0.75],
            quality_metrics={
                "bad_channels": ["T3"],
                "quality_grade": "GOOD"
            },
            processing_time=15.3,
            model_version="eegpt-v1.0"
        )

        assert result.abnormality_score == 0.75
        assert result.classification == "abnormal"
        assert result.triage_flag == TriageLevel.EXPEDITE

    def test_result_dict_conversion(self):
        """Test converting result to dictionary."""
        window_results = [
            WindowResult(0, 0.0, 4.0, 0.7, 0.9),
            WindowResult(1, 2.0, 6.0, 0.8, 0.85)
        ]

        result = AbnormalityResult(
            abnormality_score=0.75,
            classification="abnormal",
            confidence=0.85,
            triage_flag=TriageLevel.EXPEDITE,
            window_scores=window_results,
            quality_metrics={"bad_channels": []},
            processing_time=15.3,
            model_version="eegpt-v1.0"
        )

        result_dict = result.to_dict()

        assert result_dict['abnormality_score'] == 0.75
        assert result_dict['triage_flag'] == "EXPEDITE"
        assert len(result_dict['window_scores']) == 2

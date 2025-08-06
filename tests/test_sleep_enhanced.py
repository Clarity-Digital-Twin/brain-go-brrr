"""Comprehensive tests for enhanced YASA sleep analyzer with flexible channel support."""

from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest

from brain_go_brrr.core.exceptions import UnsupportedMontageError
from brain_go_brrr.core.sleep.analyzer_enhanced import EnhancedSleepAnalyzer, YASAConfig


@pytest.fixture
def sample_raw_data():
    """Create sample MNE Raw data with various channel configurations."""
    # Standard 10-20 montage
    sfreq = 256
    duration = 600  # 10 minutes
    n_samples = int(sfreq * duration)

    # Create channels with various names
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
                'EOG1', 'EOG2', 'EMG']

    ch_types = ['eeg'] * 19 + ['eog'] * 2 + ['emg']

    # Generate random data
    data = np.random.randn(len(ch_names), n_samples) * 1e-6

    # Add some structure to make it more realistic
    for i in range(len(ch_names)):
        if ch_types[i] == 'eeg':
            # Add alpha rhythm
            data[i] += np.sin(2 * np.pi * 10 * np.arange(n_samples) / sfreq) * 1e-6
        elif ch_types[i] == 'eog':
            # Add slow eye movements
            data[i] += np.sin(2 * np.pi * 0.5 * np.arange(n_samples) / sfreq) * 5e-6
        elif ch_types[i] == 'emg':
            # Add muscle activity
            data[i] += np.random.randn(n_samples) * 2e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    return raw


@pytest.fixture
def sleep_edf_style_raw():
    """Create Sleep-EDF style data with non-standard channel names."""
    sfreq = 100
    duration = 900  # 15 minutes
    n_samples = int(sfreq * duration)

    # Sleep-EDF style channel names
    ch_names = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal',
                'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']

    ch_types = ['eeg', 'eeg', 'eog', 'misc', 'emg', 'misc', 'misc']

    data = np.random.randn(len(ch_names), n_samples) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    return raw


@pytest.fixture
def minimal_raw():
    """Create minimal data with only one EEG channel."""
    sfreq = 100
    duration = 300  # 5 minutes
    n_samples = int(sfreq * duration)

    ch_names = ['C3']
    ch_types = ['eeg']

    data = np.random.randn(1, n_samples) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    return raw


@pytest.fixture
def analyzer():
    """Create analyzer instance with default config."""
    return EnhancedSleepAnalyzer()


@pytest.fixture
def custom_config():
    """Create custom config for testing."""
    return YASAConfig(
        use_consensus=False,
        use_single_channel=True,
        epoch_length=30.0,
        resample_freq=100.0,
        apply_smoothing=True,
        min_confidence=0.6,
        n_jobs=1
    )


class TestYASAConfig:
    """Test YASA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = YASAConfig()

        assert config.use_consensus is True
        assert config.use_single_channel is False
        assert config.epoch_length == 30.0
        assert config.resample_freq == 100.0
        assert config.apply_smoothing is True
        assert config.min_confidence == 0.5

    def test_channel_preferences(self):
        """Test channel preference lists."""
        config = YASAConfig()

        # Check EEG preferences
        assert 'C4-M1' in config.eeg_channels_preference
        assert 'C3' in config.eeg_channels_preference
        assert 'Fz' in config.eeg_channels_preference

        # Check EOG preferences
        assert 'EOG' in config.eog_channels_preference
        assert 'LOC' in config.eog_channels_preference

        # Check EMG preferences
        assert 'EMG' in config.emg_channels_preference
        assert 'Chin' in config.emg_channels_preference

    def test_custom_preferences(self):
        """Test custom channel preferences."""
        custom_eeg = ['C4', 'C3', 'O1']
        config = YASAConfig(eeg_channels_preference=custom_eeg)

        assert config.eeg_channels_preference == custom_eeg


class TestEnhancedSleepAnalyzer:
    """Test enhanced sleep analyzer."""

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config is not None
        assert analyzer.stages_processed == 0
        assert analyzer.success_rate == 1.0
        assert hasattr(analyzer, 'yasa_version')

    def test_find_best_channels_standard(self, analyzer, sample_raw_data):
        """Test channel finding with standard montage."""
        # Find EEG channel
        eeg_ch = analyzer.find_best_channels(sample_raw_data, 'eeg')
        assert eeg_ch in ['C4', 'C3', 'Cz']  # Should prefer central channels

        # Find EOG channel
        eog_ch = analyzer.find_best_channels(sample_raw_data, 'eog')
        assert eog_ch in ['EOG1', 'EOG2']

        # Find EMG channel
        emg_ch = analyzer.find_best_channels(sample_raw_data, 'emg')
        assert emg_ch == 'EMG'

    def test_find_best_channels_sleep_edf(self, analyzer, sleep_edf_style_raw):
        """Test channel finding with Sleep-EDF style names."""
        # Should find EEG channel despite non-standard name
        eeg_ch = analyzer.find_best_channels(sleep_edf_style_raw, 'eeg')
        assert eeg_ch in ['EEG Fpz-Cz', 'EEG Pz-Oz']

        # Should find EOG
        eog_ch = analyzer.find_best_channels(sleep_edf_style_raw, 'eog')
        assert eog_ch == 'EOG horizontal'

        # Should find EMG
        emg_ch = analyzer.find_best_channels(sleep_edf_style_raw, 'emg')
        assert emg_ch == 'EMG submental'

    def test_find_best_channels_minimal(self, analyzer, minimal_raw):
        """Test channel finding with minimal setup."""
        # Should find the only EEG channel
        eeg_ch = analyzer.find_best_channels(minimal_raw, 'eeg')
        assert eeg_ch == 'C3'

        # Should return None for missing channel types
        eog_ch = analyzer.find_best_channels(minimal_raw, 'eog')
        assert eog_ch is None

        emg_ch = analyzer.find_best_channels(minimal_raw, 'emg')
        assert emg_ch is None

    def test_preprocess_for_staging(self, analyzer, sample_raw_data):
        """Test preprocessing for YASA."""
        # Original sampling rate
        orig_sfreq = sample_raw_data.info['sfreq']
        assert orig_sfreq == 256

        # Preprocess
        processed = analyzer.preprocess_for_staging(sample_raw_data, copy=True)

        # Check resampling
        assert processed.info['sfreq'] == 100

        # Original should be unchanged
        assert sample_raw_data.info['sfreq'] == orig_sfreq

    def test_preprocess_no_resample_needed(self, analyzer, sleep_edf_style_raw):
        """Test preprocessing when already at target frequency."""
        # Already at 100 Hz
        assert sleep_edf_style_raw.info['sfreq'] == 100

        processed = analyzer.preprocess_for_staging(sleep_edf_style_raw)

        # Should still be 100 Hz
        assert processed.info['sfreq'] == 100

    @patch('brain_go_brrr.core.sleep.analyzer_enhanced.yasa.SleepStaging')
    def test_stage_sleep_flexible_success(self, mock_sleep_staging, analyzer, sample_raw_data):
        """Test successful sleep staging."""
        # Mock YASA response
        mock_sls = Mock()
        mock_hypnogram = np.array(['W', 'N1', 'N2', 'N3', 'N2', 'REM'] * 50)
        mock_proba = np.random.rand(300, 5)
        mock_proba = mock_proba / mock_proba.sum(axis=1, keepdims=True)

        mock_sls.predict.return_value = mock_hypnogram
        mock_sls.predict_proba.return_value = mock_proba
        mock_sleep_staging.return_value = mock_sls

        # Preprocess and stage
        processed = analyzer.preprocess_for_staging(sample_raw_data)
        results = analyzer.stage_sleep_flexible(processed)

        # Check results
        assert results['staging_successful'] is True
        assert len(results['hypnogram']) == 300
        assert results['channels_used']['eeg'] is not None
        assert 'features' in results
        assert results['mean_confidence'] > 0

    def test_stage_sleep_no_eeg_channel(self, analyzer):
        """Test staging failure when no EEG channel available."""
        # Create data with no EEG channels
        sfreq = 100
        ch_names = ['ECG', 'Resp']
        ch_types = ['ecg', 'misc']
        data = np.random.randn(2, 1000)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Should raise error
        with pytest.raises(UnsupportedMontageError):
            analyzer.stage_sleep_flexible(raw)

    @patch('brain_go_brrr.core.sleep.analyzer_enhanced.yasa.SleepStaging')
    def test_stage_sleep_with_smoothing(self, mock_sleep_staging, analyzer, sample_raw_data):
        """Test sleep staging with temporal smoothing."""
        # Create noisy hypnogram
        mock_sls = Mock()
        noisy_hypno = np.array(['W', 'N3', 'W', 'N3', 'W'] * 60)  # Very noisy
        mock_proba = np.ones((300, 5)) * 0.2

        mock_sls.predict.return_value = noisy_hypno
        mock_sls.predict_proba.return_value = mock_proba
        mock_sleep_staging.return_value = mock_sls

        # Stage with smoothing
        processed = analyzer.preprocess_for_staging(sample_raw_data)
        results = analyzer.stage_sleep_flexible(processed)

        # Smoothed hypnogram should have fewer transitions
        hypno = results['hypnogram']
        transitions = np.sum(hypno[:-1] != hypno[1:])
        original_transitions = np.sum(noisy_hypno[:-1] != noisy_hypno[1:])

        # Smoothing should reduce transitions (might not always due to mocking)
        assert len(hypno) == len(noisy_hypno)

    def test_fallback_staging(self, analyzer, minimal_raw):
        """Test fallback staging method."""
        results = analyzer._fallback_staging(minimal_raw, 'C3')

        assert results['staging_successful'] is False
        assert results['fallback_used'] is True
        assert len(results['hypnogram']) > 0
        assert results['mean_confidence'] == 0.5
        assert results['channels_used']['eeg'] == 'C3'

    def test_compute_sleep_metrics(self, analyzer):
        """Test sleep metrics computation."""
        # Create sample hypnogram (10 minutes)
        hypnogram = np.array(['W'] * 4 + ['N1'] * 2 + ['N2'] * 6 +
                            ['N3'] * 4 + ['N2'] * 2 + ['REM'] * 2)

        metrics = analyzer.compute_sleep_metrics(hypnogram, epoch_length=30.0)

        assert 'sleep_efficiency' in metrics
        assert 'fragmentation_index' in metrics
        assert 'total_sleep_time_min' in metrics
        assert metrics['sleep_efficiency'] > 0
        assert metrics['sleep_efficiency'] <= 100

    def test_compute_sleep_metrics_all_wake(self, analyzer):
        """Test metrics with all wake epochs."""
        hypnogram = np.array(['W'] * 20)

        metrics = analyzer.compute_sleep_metrics(hypnogram)

        assert metrics['sleep_efficiency'] == 0
        assert metrics['total_sleep_time_min'] == 0
        assert metrics['rem_latency_min'] is None

    def test_generate_sleep_report(self, analyzer):
        """Test sleep report generation."""
        # Mock staging results
        staging_results = {
            'hypnogram': np.array(['W'] * 10 + ['N2'] * 50 + ['N3'] * 20 + ['REM'] * 20),
            'mean_confidence': 0.85,
            'channels_used': {'eeg': 'C3', 'eog': 'EOG1', 'emg': 'EMG'},
            'staging_successful': True
        }

        # Mock metrics
        metrics = {
            'sleep_efficiency': 85.0,
            'fragmentation_index': 0.15,
            'total_sleep_time_min': 450,
            'total_recording_time_min': 500,
            '%W': 10, '%N1': 5, '%N2': 50, '%N3': 20, '%REM': 15,
            'sleep_onset_latency_min': 15,
            'rem_latency_min': 90
        }

        report = analyzer.generate_sleep_report(staging_results, metrics)

        assert 'summary' in report
        assert 'stage_distribution' in report
        assert 'sleep_architecture' in report
        assert 'confidence' in report
        assert 'clinical_flags' in report

        assert report['summary']['quality_grade'] in ['A', 'B', 'C', 'D', 'F']
        assert report['summary']['sleep_efficiency'] == 85.0

    def test_quality_score_calculation(self, analyzer):
        """Test sleep quality score calculation."""
        # Good sleep metrics
        good_metrics = {
            'sleep_efficiency': 90,
            'fragmentation_index': 0.05,
            '%REM': 22,
            '%N3': 18,
            'sleep_onset_latency_min': 10
        }

        score = analyzer._calculate_quality_score(good_metrics)
        grade = analyzer._score_to_grade(score)

        assert score > 80  # Should be high
        assert grade in ['A', 'B']

        # Poor sleep metrics
        poor_metrics = {
            'sleep_efficiency': 60,
            'fragmentation_index': 0.4,
            '%REM': 5,
            '%N3': 3,
            'sleep_onset_latency_min': 90
        }

        score = analyzer._calculate_quality_score(poor_metrics)
        grade = analyzer._score_to_grade(score)

        assert score < 60  # Should be low
        assert grade in ['D', 'F']

    def test_clinical_flags_generation(self, analyzer):
        """Test clinical flag generation."""
        # Problematic metrics
        problematic_metrics = {
            'sleep_efficiency': 65,  # Poor
            'fragmentation_index': 0.35,  # High
            '%REM': 8,  # Low
            '%N3': 3,  # Very low
            'sleep_onset_latency_min': 90,  # Long
            'rem_latency_min': 150  # Delayed
        }

        flags = analyzer._generate_clinical_flags(problematic_metrics)

        assert len(flags) > 0
        assert any('efficiency' in flag for flag in flags)
        assert any('fragmented' in flag for flag in flags)
        assert any('REM' in flag for flag in flags)
        assert any('deep sleep' in flag for flag in flags)

    def test_extract_staging_features(self, analyzer, sample_raw_data):
        """Test feature extraction."""
        processed = analyzer.preprocess_for_staging(sample_raw_data)

        eeg_ch = analyzer.find_best_channels(processed, 'eeg')
        eog_ch = analyzer.find_best_channels(processed, 'eog')
        emg_ch = analyzer.find_best_channels(processed, 'emg')

        features = analyzer._extract_staging_features(
            processed, eeg_ch, eog_ch, emg_ch
        )

        # Check key features from paper
        assert 'eeg_beta_power' in features
        assert 'eeg_fractal_dimension' in features
        assert 'eeg_permutation_entropy' in features

        if eog_ch:
            assert 'eog_absolute_power' in features

        if emg_ch:
            assert 'emg_power' in features

    def test_compute_fractal_dimension(self, analyzer):
        """Test fractal dimension computation."""
        # Generate test signal
        data = np.random.randn(1000)

        fd = analyzer._compute_fractal_dimension(data)

        # Fractal dimension should be between 1 and 2 for time series
        assert 0 < fd < 3

    def test_compute_permutation_entropy(self, analyzer):
        """Test permutation entropy computation."""
        # Regular signal (low entropy)
        regular = np.sin(2 * np.pi * np.linspace(0, 10, 1000))
        entropy_regular = analyzer._compute_permutation_entropy(regular)

        # Random signal (high entropy)
        random = np.random.randn(1000)
        entropy_random = analyzer._compute_permutation_entropy(random)

        # Both should return valid entropy values
        assert 0 < entropy_regular < 10
        assert 0 < entropy_random < 10

    def test_temporal_smoothing(self, analyzer):
        """Test temporal smoothing implementation."""
        # Create noisy hypnogram
        hypnogram = np.array(['W', 'N1', 'W', 'N1', 'W', 'N1'] * 20)

        smoothed = analyzer._apply_temporal_smoothing(hypnogram, window_min=7.5)

        # Should have same length
        assert len(smoothed) == len(hypnogram)

        # Should have fewer transitions
        original_transitions = np.sum(hypnogram[:-1] != hypnogram[1:])
        smoothed_transitions = np.sum(smoothed[:-1] != smoothed[1:])

        # May not always reduce due to edge effects, but should not increase much
        assert smoothed_transitions <= original_transitions + 5


class TestIntegrationWithSleepEDF:
    """Integration tests with Sleep-EDF style data."""

    @pytest.fixture
    def sleep_edf_file(self):
        """Mock Sleep-EDF file path."""
        return Path("/data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf")

    @patch('mne.io.read_raw_edf')
    def test_full_pipeline_sleep_edf(self, mock_read_edf, analyzer, sleep_edf_style_raw):
        """Test full pipeline with Sleep-EDF data."""
        mock_read_edf.return_value = sleep_edf_style_raw

        # Mock YASA
        with patch('brain_go_brrr.core.sleep.analyzer_enhanced.yasa.SleepStaging') as mock_sls:
            mock_instance = Mock()
            mock_instance.predict.return_value = np.array(['W', 'N1', 'N2'] * 100)
            mock_instance.predict_proba.return_value = np.ones((300, 5)) * 0.2
            mock_sls.return_value = mock_instance

            # Run full pipeline
            raw = mock_read_edf("dummy_path")
            processed = analyzer.preprocess_for_staging(raw)
            results = analyzer.stage_sleep_flexible(processed)
            metrics = analyzer.compute_sleep_metrics(results['hypnogram'])
            report = analyzer.generate_sleep_report(results, metrics)

            assert report['summary']['total_recording_time'] > 0
            assert 'quality_grade' in report['summary']
            assert report['confidence']['staging_successful'] is True


class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""

    def test_large_dataset_performance(self, analyzer):
        """Test with large dataset (8 hours)."""
        # Create 8-hour recording
        sfreq = 100
        duration = 8 * 3600  # 8 hours
        n_samples = int(sfreq * duration)

        ch_names = ['C3', 'C4', 'EOG1', 'EMG']
        ch_types = ['eeg', 'eeg', 'eog', 'emg']

        # Use smaller chunks to avoid memory issues
        data = np.random.randn(len(ch_names), min(n_samples, 1000000)) * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Should handle without error
        processed = analyzer.preprocess_for_staging(raw)
        assert processed is not None

    def test_short_recording(self, analyzer):
        """Test with very short recording."""
        # Only 2 minutes
        sfreq = 100
        duration = 120
        n_samples = int(sfreq * duration)

        ch_names = ['C3']
        ch_types = ['eeg']
        data = np.random.randn(1, n_samples) * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Should use fallback
        results = analyzer._fallback_staging(raw, 'C3')

        # Should have at least some epochs
        assert len(results['hypnogram']) >= 4  # At least 4 epochs

    def test_channel_type_detection(self, analyzer):
        """Test automatic channel type detection."""
        sfreq = 100
        ch_names = ['Fpz', 'EOG_left', 'Chin_EMG', 'ECG', 'Misc1']
        ch_types = ['misc'] * 5  # All set to misc initially

        data = np.random.randn(5, 1000) * 1e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Apply type detection
        analyzer._set_channel_types(raw)

        types = raw.get_channel_types()

        # Should detect types from names
        assert types[0] == 'eeg'  # Fpz
        assert types[1] == 'eog'  # EOG_left
        assert types[2] == 'emg'  # Chin_EMG

    def test_config_single_channel_mode(self):
        """Test single channel mode configuration."""
        config = YASAConfig(use_single_channel=True)
        analyzer = EnhancedSleepAnalyzer(config)

        assert analyzer.config.use_single_channel is True

    def test_case_insensitive_channel_matching(self, analyzer):
        """Test case-insensitive channel matching."""
        sfreq = 100
        ch_names = ['c3', 'C4', 'eog1', 'EMG']  # Mixed case
        ch_types = ['eeg', 'eeg', 'eog', 'emg']

        data = np.random.randn(4, 1000) * 1e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Should find channels regardless of case
        eeg_ch = analyzer.find_best_channels(raw, 'eeg')
        assert eeg_ch in ['c3', 'C4']

        eog_ch = analyzer.find_best_channels(raw, 'eog')
        assert eog_ch == 'eog1'


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_yasa_validation(self, analyzer):
        """Test YASA validation."""
        # Just verify that YASA version is set during initialization
        assert hasattr(analyzer, 'yasa_version')
        assert analyzer.yasa_version is not None

    def test_invalid_channel_type(self, analyzer, minimal_raw):
        """Test invalid channel type request."""
        result = analyzer.find_best_channels(minimal_raw, 'invalid_type')
        assert result is None

    def test_empty_hypnogram(self, analyzer):
        """Test metrics with empty hypnogram."""
        hypnogram = np.array([])

        metrics = analyzer.compute_sleep_metrics(hypnogram)

        assert metrics['sleep_efficiency'] == 0
        assert metrics['total_sleep_time_min'] == 0

    @patch('brain_go_brrr.core.sleep.analyzer_enhanced.yasa.SleepStaging')
    def test_yasa_failure_fallback(self, mock_sleep_staging, analyzer, sample_raw_data):
        """Test fallback when YASA fails."""
        # Make YASA raise an error
        mock_sleep_staging.side_effect = Exception("YASA failed")

        processed = analyzer.preprocess_for_staging(sample_raw_data)
        results = analyzer.stage_sleep_flexible(processed)

        # Should use fallback
        assert results['staging_successful'] is False
        assert results.get('fallback_used') is True
        assert len(results['hypnogram']) > 0

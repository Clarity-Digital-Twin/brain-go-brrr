#!/usr/bin/env python
"""Integration tests for YASA sleep staging in the pipeline."""


import numpy as np
import pytest

from brain_go_brrr.services.hierarchical_pipeline import HierarchicalEEGAnalyzer, PipelineConfig
from brain_go_brrr.services.yasa_adapter import YASASleepStager


@pytest.mark.integration
class TestYASAIntegration:
    """Test YASA integration with real data."""

    def test_yasa_with_sleep_edf_data(self, sleep_edf_raw_cropped):
        """Test YASA on real Sleep-EDF data."""
        # Get data from MNE Raw
        data = sleep_edf_raw_cropped.get_data()
        sfreq = sleep_edf_raw_cropped.info['sfreq']
        ch_names = sleep_edf_raw_cropped.ch_names

        # Initialize YASA stager
        stager = YASASleepStager()

        # Need at least 30s for one epoch
        if data.shape[1] / sfreq >= 30:
            stages, confidences, metrics = stager.stage_sleep(
                data, sfreq, ch_names
            )

            # Check results
            assert len(stages) > 0
            assert all(s in ['W', 'N1', 'N2', 'N3', 'REM'] for s in stages)
            assert all(0 <= c <= 1 for c in confidences)
            assert metrics['n_epochs'] == len(stages)

    def test_pipeline_with_yasa_enabled(self, sleep_edf_raw_cropped):
        """Test full pipeline with YASA enabled."""
        config = PipelineConfig(
            enable_abnormality_screening=True,
            enable_sleep_staging=True,
            use_yasa_sleep_staging=True,
            parallel_execution=False  # Easier to debug
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Get 30s of data minimum
        data = sleep_edf_raw_cropped.get_data()
        sfreq = sleep_edf_raw_cropped.info['sfreq']

        # Ensure we have at least 30s
        min_samples = int(30 * sfreq)
        if data.shape[1] >= min_samples:
            result = pipeline.analyze(data[:, :min_samples])

            assert result.sleep_stage is not None
            assert result.sleep_stage in ['W', 'N1', 'N2', 'N3', 'REM']
            assert result.sleep_confidence is not None
            assert 0 <= result.sleep_confidence <= 1

    def test_pipeline_yasa_vs_mock(self, mock_eeg_data):
        """Compare YASA vs mock implementation."""
        # Create 60s of data for better testing
        data = mock_eeg_data.get_data()
        sfreq = mock_eeg_data.info['sfreq']
        n_samples = int(60 * sfreq)

        if data.shape[1] < n_samples:
            # Extend data by repeating
            repeats = (n_samples // data.shape[1]) + 1
            data = np.tile(data, (1, repeats))[:, :n_samples]

        # Test with YASA
        config_yasa = PipelineConfig(
            enable_sleep_staging=True,
            use_yasa_sleep_staging=True
        )
        pipeline_yasa = HierarchicalEEGAnalyzer(config_yasa)
        result_yasa = pipeline_yasa.analyze(data)

        # Test with mock
        config_mock = PipelineConfig(
            enable_sleep_staging=True,
            use_yasa_sleep_staging=False
        )
        pipeline_mock = HierarchicalEEGAnalyzer(config_mock)
        result_mock = pipeline_mock.analyze(data)

        # Both should return valid results
        assert result_yasa.sleep_stage in ['W', 'N1', 'N2', 'N3', 'REM']
        assert result_mock.sleep_stage in ['W', 'N1', 'N2', 'N3', 'REM']

        # YASA should generally have different (likely better) results
        # but we can't guarantee they'll be different on random data
        assert result_yasa.sleep_confidence is not None
        assert result_mock.sleep_confidence is not None

    def test_yasa_error_handling(self):
        """Test YASA handles errors gracefully."""
        config = PipelineConfig(
            enable_sleep_staging=True,
            use_yasa_sleep_staging=True
        )

        pipeline = HierarchicalEEGAnalyzer(config)

        # Test with corrupted data
        corrupted_data = np.full((19, 7680), np.nan)  # All NaN

        # Should not crash, but might return None or use fallback
        result = pipeline.analyze(corrupted_data)

        # Pipeline should handle the error
        assert result is not None
        # Sleep staging might be None or fallback value
        if result.sleep_stage is not None:
            assert result.sleep_stage in ['W', 'N1', 'N2', 'N3', 'REM']

#!/usr/bin/env python
"""Unit tests for YASA sleep staging adapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brain_go_brrr.services.yasa_adapter import (
    HierarchicalPipelineYASAAdapter,
    YASAConfig,
    YASASleepStager,
)


class TestYASAConfig:
    """Test YASA configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = YASAConfig()

        assert config.use_consensus is True
        assert config.eeg_backend == "lightgbm"
        assert config.freq_broad == (0.5, 35.0)
        assert config.min_confidence == 0.5
        assert config.n_jobs == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = YASAConfig(
            use_consensus=False, eeg_backend="perceptron", min_confidence=0.7, n_jobs=4
        )

        assert config.use_consensus is False
        assert config.eeg_backend == "perceptron"
        assert config.min_confidence == 0.7
        assert config.n_jobs == 4


class TestYASASleepStager:
    """Test YASA sleep stager."""

    @pytest.fixture
    def mock_yasa(self):
        """Mock YASA module."""
        with patch("brain_go_brrr.services.yasa_adapter.yasa") as mock:
            # Mock SleepStaging class
            mock_sls = MagicMock()
            mock_sls.predict.return_value = np.array([0, 1, 2, 3, 4])  # W, N1, N2, N3, REM
            mock_sls.predict_proba.return_value = np.array(
                [
                    [0.9, 0.05, 0.03, 0.01, 0.01],  # W with high confidence
                    [0.1, 0.8, 0.05, 0.03, 0.02],  # N1 with high confidence
                    [0.05, 0.1, 0.7, 0.1, 0.05],  # N2 with good confidence
                    [0.02, 0.03, 0.15, 0.75, 0.05],  # N3 with good confidence
                    [0.05, 0.05, 0.1, 0.1, 0.7],  # REM with good confidence
                ]
            )
            mock.SleepStaging.return_value = mock_sls
            yield mock

    def test_initialization(self):
        """Test stager initialization."""
        stager = YASASleepStager()

        assert stager.config is not None
        assert stager.stages_processed == 0
        assert stager.avg_confidence == 0.0

    def test_stage_sleep_basic(self, mock_yasa):
        """Test basic sleep staging."""
        stager = YASASleepStager()

        # Create 5 minutes of data (10 epochs)
        eeg_data = np.random.randn(19, 256 * 300)  # 19 channels, 5 min @ 256Hz

        stages, confidences, metrics = stager.stage_sleep(eeg_data)

        # Check outputs
        assert len(stages) == 5
        assert stages == ["W", "N1", "N2", "N3", "REM"]
        assert len(confidences) == 5
        assert all(0 <= c <= 1 for c in confidences)
        assert confidences[0] > 0.8  # High confidence for wake

        # Check metrics
        assert "stage_counts" in metrics
        assert "sleep_efficiency" in metrics
        assert metrics["n_epochs"] == 5

    def test_stage_sleep_short_data_error(self):
        """Test error handling for too-short data."""
        stager = YASASleepStager()

        # Only 10 seconds of data (less than 30s epoch)
        short_data = np.random.randn(19, 256 * 10)

        with pytest.raises(ValueError, match="Data too short"):
            stager.stage_sleep(short_data)

    def test_channel_selection(self):
        """Test EEG channel selection logic."""
        stager = YASASleepStager()

        # Test with preferred channels
        ch_names = ["F3", "C3", "O1", "T3"]
        selected = stager._select_eeg_channel(ch_names)
        assert selected == "C3"  # C3 is preferred

        # Test without central/frontal channels - should pick occipital
        ch_names = ["T3", "T4", "O1", "O2"]
        selected = stager._select_eeg_channel(ch_names)
        assert selected == "O2"  # O2 is in preferred list before others

        # Test with only non-preferred channels
        ch_names = ["T3", "T4", "T5", "T6"]
        selected = stager._select_eeg_channel(ch_names)
        assert selected == "T3"  # Falls back to first if no preferred

    def test_yasa_to_standard_stage(self):
        """Test stage conversion."""
        stager = YASASleepStager()

        assert stager._yasa_to_standard_stage(0) == "W"
        assert stager._yasa_to_standard_stage(1) == "N1"
        assert stager._yasa_to_standard_stage(2) == "N2"
        assert stager._yasa_to_standard_stage(3) == "N3"
        assert stager._yasa_to_standard_stage(4) == "REM"
        assert stager._yasa_to_standard_stage(99) == "W"  # Unknown -> Wake

    def test_calculate_sleep_metrics(self):
        """Test sleep metrics calculation."""
        stager = YASASleepStager()

        # Create hypnogram
        stages = ["W", "W", "N1", "N2", "N2", "N3", "N3", "REM", "W", "N2"]
        confidences = [0.9, 0.8, 0.7, 0.8, 0.85, 0.9, 0.85, 0.75, 0.6, 0.8]

        metrics = stager._calculate_sleep_metrics(stages, confidences)

        # Check counts
        assert metrics["stage_counts"]["W"] == 3
        assert metrics["stage_counts"]["N1"] == 1
        assert metrics["stage_counts"]["N2"] == 3
        assert metrics["stage_counts"]["N3"] == 2
        assert metrics["stage_counts"]["REM"] == 1

        # Check percentages
        assert metrics["stage_percentages"]["W"] == 30.0
        assert metrics["stage_percentages"]["N2"] == 30.0

        # Check sleep efficiency
        assert metrics["sleep_efficiency"] == 70.0  # 7/10 epochs are sleep

        # Check WASO
        assert metrics["waso_epochs"] == 1  # One W epoch after sleep onset

        # Check confidence
        assert 0.7 < metrics["mean_confidence"] < 0.9

    @patch("brain_go_brrr.services.yasa_adapter.mne.io.read_raw")
    def test_process_full_night(self, mock_read_raw, mock_yasa):
        """Test processing full night recording."""
        # Mock raw data
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(4, 256 * 3600)  # 4 channels, 1 hour
        mock_raw.info = {"sfreq": 256}
        mock_raw.ch_names = ["C3", "C4", "O1", "O2"]
        mock_read_raw.return_value = mock_raw

        stager = YASASleepStager()

        results = stager.process_full_night(Path("/fake/eeg.edf"))

        assert "file" in results
        assert "duration_hours" in results
        assert "metrics" in results
        assert "quality_check" in results
        assert "hypnogram" in results

        # Check quality warnings
        qc = results["quality_check"]
        assert "mean_confidence" in qc
        assert "low_confidence_epochs" in qc
        assert isinstance(qc["confidence_warning"], bool)


class TestHierarchicalPipelineYASAAdapter:
    """Test the pipeline adapter."""

    @patch("brain_go_brrr.services.yasa_adapter.yasa")
    def test_adapter_interface(self, mock_yasa):
        """Test adapter matches our pipeline interface."""
        # Setup mock
        mock_sls = MagicMock()
        mock_sls.predict.return_value = np.array([2, 2, 2, 3, 3])  # Mostly N2/N3
        mock_sls.predict_proba.return_value = np.ones((5, 5)) * 0.2
        np.fill_diagonal(mock_sls.predict_proba.return_value, 0.8)
        mock_yasa.SleepStaging.return_value = mock_sls

        adapter = HierarchicalPipelineYASAAdapter()

        # Test with 5 minutes of data
        eeg = np.random.randn(19, 256 * 300)

        stage, confidence = adapter.stage(eeg)

        assert stage in ["W", "N1", "N2", "N3", "REM"]
        assert 0 <= confidence <= 1
        assert stage == "N2"  # Most common stage
        assert confidence > 0.7  # Good confidence

    def test_adapter_error_handling(self):
        """Test adapter handles errors gracefully."""
        adapter = HierarchicalPipelineYASAAdapter()

        # Mock stager to raise error
        adapter.stager.stage_sleep = MagicMock(side_effect=Exception("YASA error"))

        # Should return wake with zero confidence
        eeg = np.random.randn(19, 2048)
        stage, confidence = adapter.stage(eeg)

        assert stage == "W"
        assert confidence == 0.0


class TestLightGBMFallback:
    """Test LightGBM fallback behavior."""

    def test_lightgbm_available(self):
        """Test when LightGBM is available."""
        # Since we installed it, it should be available
        stager = YASASleepStager()
        assert stager.config.eeg_backend == "lightgbm"

    def test_lightgbm_not_available(self):
        """Test fallback when LightGBM not available."""
        # Create custom config with perceptron backend
        config = YASAConfig(eeg_backend="perceptron")
        stager = YASASleepStager(config=config)

        # Should use perceptron as configured
        assert stager.config.eeg_backend == "perceptron"

"""Tests for MNE preprocessor implementation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brain_go_brrr.infra.preprocessing.mne_preprocessor import MNEPreprocessor


class TestMNEPreprocessor:
    """Test MNE preprocessing functionality."""

    @pytest.fixture
    def mock_raw(self):
        """Create mock MNE raw object."""
        raw = MagicMock()
        raw.info = {'sfreq': 256, 'ch_names': ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2']}
        raw.get_data.return_value = np.random.randn(6, 256 * 10)
        raw.filter = MagicMock(return_value=raw)
        raw.resample = MagicMock(return_value=raw)
        raw.notch_filter = MagicMock(return_value=raw)
        raw.pick_channels = MagicMock(return_value=raw)
        raw.copy = MagicMock(return_value=raw)
        return raw

    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = MNEPreprocessor(
            sampling_rate=256,
            bandpass=(0.5, 50),
            notch=60
        )
        assert preprocessor.sampling_rate == 256
        assert preprocessor.bandpass == (0.5, 50)
        assert preprocessor.notch == 60

    def test_preprocess_raw(self, mock_raw):
        """Test preprocessing raw data."""
        preprocessor = MNEPreprocessor()
        result = preprocessor.preprocess(mock_raw)
        
        # Should apply filters
        mock_raw.filter.assert_called()
        mock_raw.notch_filter.assert_called()
        assert result is not None

    def test_resampling(self, mock_raw):
        """Test resampling when needed."""
        mock_raw.info['sfreq'] = 512  # Different from target
        
        preprocessor = MNEPreprocessor(sampling_rate=256)
        preprocessor.preprocess(mock_raw)
        
        # Should resample
        mock_raw.resample.assert_called_with(256)

    def test_channel_selection(self, mock_raw):
        """Test channel selection."""
        preprocessor = MNEPreprocessor(
            channels=['Fp1', 'C3', 'O1']
        )
        preprocessor.preprocess(mock_raw)
        
        # Should pick specified channels
        mock_raw.pick_channels.assert_called()

    def test_autoreject_integration(self, mock_raw):
        """Test autoreject integration."""
        with patch('brain_go_brrr.infra.preprocessing.mne_preprocessor.AutoReject') as mock_ar:
            preprocessor = MNEPreprocessor(use_autoreject=True)
            preprocessor.preprocess(mock_raw)
            
            # Should use autoreject if enabled
            mock_ar.assert_called()

    def test_normalization(self):
        """Test data normalization."""
        preprocessor = MNEPreprocessor(normalize=True)
        data = np.random.randn(6, 1000) * 100 + 50
        
        normalized = preprocessor._normalize(data)
        
        # Check normalized
        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1.0) < 0.1
"""Simplified unit tests for AutoReject fallback mechanisms."""

import logging
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset


class TestAutoRejectFallbacksSimple:
    """Test AutoReject fallback mechanisms with simplified mocking."""

    @pytest.fixture
    def mock_raw(self):
        """Create mock raw EEG data."""
        ch_names = ['C3', 'C4', 'CZ', 'F3', 'F4']
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 2560)
        return mne.io.RawArray(data, info)

    def test_amplitude_based_cleaning_method(self, mock_raw):
        """Test the amplitude-based cleaning method directly."""
        # Create instance just to access the method
        dataset = object.__new__(TUABEnhancedDataset)

        # Test the amplitude cleaning method
        result = TUABEnhancedDataset._amplitude_based_cleaning(dataset, mock_raw)

        # Should return cleaned data
        assert result is not None
        assert isinstance(result.info['bads'], list)

    def test_apply_autoreject_fallback(self, mock_raw):
        """Test that _apply_autoreject_to_raw falls back gracefully."""
        # Create minimal mock dataset
        dataset = Mock()
        dataset.use_autoreject = True
        dataset.ar_processor = Mock()
        dataset.window_adapter = Mock()
        dataset.position_generator = Mock()

        # Mock AR processor to raise error
        dataset.ar_processor.is_fitted = False
        dataset.ar_processor._load_parameters.side_effect = ValueError("No cached params")

        # Call the method directly
        with patch.object(TUABEnhancedDataset, '_amplitude_based_cleaning', return_value=mock_raw):
            result = TUABEnhancedDataset._apply_autoreject_to_raw(dataset, mock_raw)

        # Should fall back and return data
        assert result is not None

    def test_fallback_logging(self, mock_raw, caplog):
        """Test that fallbacks are logged properly."""
        dataset = Mock()
        dataset.use_autoreject = True
        dataset.ar_processor = Mock()
        dataset.window_adapter = Mock()
        dataset.position_generator = Mock()

        # Mock ar_processor.transform_raw to raise a generic error
        dataset.ar_processor.transform_raw.side_effect = ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with patch.object(TUABEnhancedDataset, '_amplitude_based_cleaning', return_value=mock_raw):
                result = TUABEnhancedDataset._apply_autoreject_to_raw(dataset, mock_raw)

        # Check logging
        assert "AutoReject failed" in caplog.text
        assert result is not None

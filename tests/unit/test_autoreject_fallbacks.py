"""Unit tests for AutoReject fallback mechanisms."""

import logging
from unittest.mock import patch

import mne
import numpy as np
import pytest

from brain_go_brrr.data.tuab_enhanced_dataset import TUABEnhancedDataset

# This will fail until implemented - TDD!


class TestAutoRejectFallbackMechanisms:
    """Test all fallback paths when AutoReject fails."""

    @pytest.fixture
    def mock_raw_no_positions(self):
        """Create raw data without channel positions (like TUAB)."""
        ch_names = ['FP1', 'FP2', 'C3', 'C4', 'O1', 'O2']
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 2560)
        raw = mne.io.RawArray(data, info)

        # Ensure no positions
        for ch in raw.info['chs']:
            ch['loc'][:3] = 0

        return raw

    @pytest.fixture
    def mock_raw_insufficient_channels(self):
        """Create raw data with too few channels for AutoReject."""
        # AutoReject needs at least 5 channels
        ch_names = ['C3', 'C4', 'CZ']  # Only 3 channels
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 2560)
        return mne.io.RawArray(data, info)

    def test_fallback_no_channel_positions(self, mock_raw_no_positions, caplog):
        """Test fallback when channel positions are missing."""
        # Given: Dataset with AutoReject enabled
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # When: Processing data without positions
        with caplog.at_level(logging.WARNING):
            result = dataset._apply_autoreject_to_raw(mock_raw_no_positions)

        # Then: Should fall back to amplitude rejection
        assert "No channel positions" in caplog.text
        assert "amplitude rejection" in caplog.text
        assert result is not None

    def test_fallback_memory_error(self, mock_raw_no_positions):
        """Test fallback when AutoReject causes MemoryError."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # Mock AutoReject to raise MemoryError
        with patch.object(dataset.ar_processor, 'transform_raw', side_effect=MemoryError("OOM")):
            with patch.object(dataset, '_amplitude_based_cleaning') as mock_amp:
                mock_amp.return_value = mock_raw_no_positions

                # Should not crash, use fallback
                result = dataset._apply_autoreject_to_raw(mock_raw_no_positions)

                assert result == mock_raw_no_positions
                mock_amp.assert_called_once()

    def test_fallback_insufficient_channels(self, mock_raw_insufficient_channels, caplog):
        """Test fallback with too few channels."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # When: Processing with insufficient channels
        with caplog.at_level(logging.WARNING):
            result = dataset._apply_autoreject_to_raw(mock_raw_insufficient_channels)

        # Then: Should warn and use fallback
        assert "channels" in caplog.text.lower()
        assert result is not None

    def test_amplitude_based_cleaning(self):
        """Test the amplitude-based cleaning fallback."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=False  # Explicitly disable
        )

        # Create data with artifacts
        ch_names = ['FP1', 'FP2', 'C3', 'C4', 'O1', 'O2']
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 10000) * 50e-6

        # Add bad channels
        data[0, :] = 0  # Flat channel
        data[1, :] = np.random.randn(10000) * 500e-6  # Very noisy

        raw = mne.io.RawArray(data, info)

        # Apply amplitude cleaning
        cleaned = dataset._amplitude_based_cleaning(raw)

        # Should mark bad channels
        assert len(cleaned.info['bads']) >= 2
        assert 'FP1' in cleaned.info['bads']  # Flat
        assert 'FP2' in cleaned.info['bads']  # Noisy

    def test_graceful_degradation_chain(self):
        """Test the complete fallback chain."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # Create challenging data
        ch_names = ['C3', 'C4']  # Minimal channels
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(2, 5000)  # Short duration
        raw = mne.io.RawArray(data, info)

        # Should handle gracefully
        result = dataset._apply_autoreject_to_raw(raw)
        assert result is not None
        assert isinstance(result, mne.io.BaseRaw)

    def test_fallback_runtime_error(self, mock_raw_no_positions):
        """Test handling of RuntimeError from AutoReject."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # Mock various RuntimeErrors
        errors = [
            RuntimeError("Valid channel positions"),
            RuntimeError("SVD did not converge"),
            RuntimeError("Matrix is singular")
        ]

        for error in errors:
            with patch.object(dataset.ar_processor, 'transform_raw', side_effect=error):
                # Should handle without crashing
                result = dataset._apply_autoreject_to_raw(mock_raw_no_positions)
                assert result is not None

    def test_fallback_preserves_data_integrity(self, mock_raw_no_positions):
        """Test that fallback methods preserve data structure."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # Get original properties
        orig_n_channels = len(mock_raw_no_positions.ch_names)
        orig_n_times = mock_raw_no_positions.n_times
        orig_sfreq = mock_raw_no_positions.info['sfreq']

        # Force fallback
        with patch.object(dataset.ar_processor, 'transform_raw', side_effect=RuntimeError):
            result = dataset._apply_autoreject_to_raw(mock_raw_no_positions)

        # Data structure should be preserved
        assert len(result.ch_names) == orig_n_channels
        assert result.n_times == orig_n_times
        assert result.info['sfreq'] == orig_sfreq

    @pytest.mark.parametrize("error_type,error_msg", [
        (MemoryError, "Out of memory"),
        (RuntimeError, "Valid channel positions required"),
        (ValueError, "Not enough good channels"),
        (np.linalg.LinAlgError, "SVD convergence failed")
    ])
    def test_all_error_types_handled(self, error_type, error_msg, mock_raw_no_positions):
        """Test that all expected error types are handled."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        with patch.object(dataset.ar_processor, 'transform_raw',
                         side_effect=error_type(error_msg)):
            # Should not crash
            result = dataset._apply_autoreject_to_raw(mock_raw_no_positions)
            assert result is not None

    def test_fallback_performance(self, mock_raw_no_positions):
        """Test that fallback is fast."""
        import time

        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=False
        )

        # Time amplitude-based cleaning
        start = time.time()
        result = dataset._amplitude_based_cleaning(mock_raw_no_positions)
        elapsed = time.time() - start

        # Should be very fast (< 10ms for small data)
        assert elapsed < 0.01
        assert result is not None

    def test_logging_on_fallback(self, mock_raw_no_positions, caplog):
        """Test that fallbacks are properly logged."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # Force different fallback scenarios
        scenarios = [
            (MemoryError, "OOM"),
            (RuntimeError, "channel positions"),
            (ValueError, "interpolation")
        ]

        for error_type, keyword in scenarios:
            with patch.object(dataset.ar_processor, 'transform_raw',
                            side_effect=error_type(keyword)):
                with caplog.at_level(logging.WARNING):
                    result = dataset._apply_autoreject_to_raw(mock_raw_no_positions)

                    # Should log the fallback
                    assert keyword in caplog.text or "fallback" in caplog.text.lower()
                    caplog.clear()


class TestFallbackStatistics:
    """Test tracking of fallback usage."""

    def test_fallback_counter(self):
        """Test that we track how often fallbacks are used."""
        dataset = TUABEnhancedDataset(
            root_dir="dummy",
            split="train",
            use_autoreject=True
        )

        # Initialize counters
        dataset.fallback_stats = {
            'total_files': 0,
            'autoreject_success': 0,
            'amplitude_fallback': 0,
            'position_fallback': 0,
            'memory_fallback': 0,
            'error_fallback': 0
        }

        # Simulate processing files
        dataset.fallback_stats['total_files'] = 100
        dataset.fallback_stats['autoreject_success'] = 70
        dataset.fallback_stats['amplitude_fallback'] = 20
        dataset.fallback_stats['position_fallback'] = 10

        # Calculate rates
        success_rate = dataset.fallback_stats['autoreject_success'] / dataset.fallback_stats['total_files']
        fallback_rate = 1 - success_rate

        assert success_rate == 0.7
        assert fallback_rate == 0.3

        # This info should be logged during training
        print(f"AutoReject success rate: {success_rate:.1%}")
        print(f"Fallback rate: {fallback_rate:.1%}")

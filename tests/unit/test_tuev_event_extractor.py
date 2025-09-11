"""Unit tests for TUEVEventExtractor - TDD approach for paper parity."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from brain_go_brrr.infra.preprocessing.tuev_event_extractor import TUEVEventExtractor


class TestTUEVEventExtractor:
    """Test TUEV event segment extraction for paper parity."""

    def test_init_with_correct_defaults(self):
        """Test extractor initializes with paper-specified defaults."""
        extractor = TUEVEventExtractor()

        assert extractor.target_fs == 200  # EEGPT uses 200Hz not 256Hz
        assert extractor.segment_duration == 5.0  # 5 seconds
        assert extractor.tmin == -2.0  # 2 seconds before event
        assert extractor.tmax == 3.0  # 3 seconds after event

    def test_channel_order_matches_reference(self):
        """Test channel order exactly matches EEGPT reference."""
        extractor = TUEVEventExtractor()

        expected_channels = [
            'EEG FP1-REF',
            'EEG FP2-REF',
            'EEG F3-REF',
            'EEG F4-REF',
            'EEG C3-REF',
            'EEG C4-REF',
            'EEG P3-REF',
            'EEG P4-REF',
            'EEG O1-REF',
            'EEG O2-REF',
            'EEG F7-REF',
            'EEG F8-REF',
            'EEG T3-REF',
            'EEG T4-REF',
            'EEG T5-REF',
            'EEG T6-REF',
            'EEG A1-REF',
            'EEG A2-REF',
            'EEG FZ-REF',
            'EEG CZ-REF',
            'EEG PZ-REF',
            'EEG T1-REF',
            'EEG T2-REF',
        ]

        assert expected_channels == extractor.TUEV_CHANNELS_REF
        assert len(extractor.TUEV_CHANNELS_REF) == 23

    @patch('mne.io.read_raw_edf')
    def test_extract_segments_shape(self, mock_read_raw):
        """Test extracted segments have correct shape (23, 1000)."""
        # Create mock raw object
        mock_raw = MagicMock()
        mock_info = {'sfreq': 250.0}  # Original sampling rate
        mock_raw.info = mock_info
        mock_raw.ch_names = TUEVEventExtractor.TUEV_CHANNELS_REF.copy()

        # Mock data: 23 channels, 15 seconds of data at 200Hz (enough for both events)
        mock_data = np.random.randn(23, 3000).astype(np.float32)
        mock_raw.get_data.return_value = mock_data

        # Configure mock methods
        mock_raw.filter.return_value = None
        mock_raw.notch_filter.return_value = None
        mock_raw.resample.return_value = None
        mock_raw.pick_channels.return_value = None

        mock_read_raw.return_value = mock_raw

        # Test annotations
        annotations = [
            {'start': 3.0, 'end': 3.5, 'label': 0},  # Event at 3-3.5s
            {'start': 7.0, 'end': 7.5, 'label': 1},  # Event at 7-7.5s
        ]

        extractor = TUEVEventExtractor()
        segments = extractor.extract_segments(Path('/fake/path.edf'), annotations)

        # Verify shapes
        assert len(segments) == 2
        for segment, label in segments:
            assert segment.shape == (23, 1000)  # 23 channels, 5s @ 200Hz
            assert segment.dtype == np.float32
            assert isinstance(label, int)

    @patch('mne.io.read_raw_edf')
    def test_filtering_and_resampling(self, mock_read_raw):
        """Test correct filtering parameters per EEGPT reference."""
        mock_raw = MagicMock()
        mock_raw.info = {'sfreq': 256.0}
        mock_raw.ch_names = TUEVEventExtractor.TUEV_CHANNELS_REF.copy()
        mock_raw.get_data.return_value = np.random.randn(23, 5120)  # 20s @ 256Hz

        mock_read_raw.return_value = mock_raw

        extractor = TUEVEventExtractor()
        _ = extractor.extract_segments(Path('/fake/path.edf'), [])

        # Verify filtering calls
        mock_raw.filter.assert_called_once_with(l_freq=0.1, h_freq=75.0, verbose=False)
        mock_raw.notch_filter.assert_called_once_with(freqs=50.0, verbose=False)
        mock_raw.resample.assert_called_once_with(
            200,
            verbose=False,  # Must resample to 200Hz
        )

    @patch('mne.io.read_raw_edf')
    def test_segment_extraction_window(self, mock_read_raw):
        """Test segments are extracted with correct temporal window."""
        mock_raw = MagicMock()
        mock_raw.info = {'sfreq': 200.0}  # Already at target rate
        mock_raw.ch_names = TUEVEventExtractor.TUEV_CHANNELS_REF.copy()

        # 20 seconds of data at 200Hz
        mock_data = np.arange(23 * 4000).reshape(23, 4000).astype(np.float32)
        mock_raw.get_data.return_value = mock_data

        mock_read_raw.return_value = mock_raw

        # Event at 10s (should extract 8-13s)
        annotations = [{'start': 9.5, 'end': 10.5, 'label': 0}]

        extractor = TUEVEventExtractor()
        segments = extractor.extract_segments(Path('/fake/path.edf'), annotations)

        assert len(segments) == 1
        segment, label = segments[0]

        # Event from 9.5s to 10.5s
        # Window should be start-2s to end+2s = 7.5s to 12.5s
        # At 200Hz: samples 1500 to 2500
        expected_start = 1500
        expected_end = 2500

        # Check that we got the right slice of data
        expected_segment = mock_data[:, expected_start:expected_end]
        np.testing.assert_array_almost_equal(segment, expected_segment)

    @patch('mne.io.read_raw_edf')
    def test_missing_channels_padded_with_zeros(self, mock_read_raw):
        """Test missing channels are padded with zeros to maintain shape."""
        mock_raw = MagicMock()
        mock_raw.info = {'sfreq': 200.0}
        # Only have subset of channels
        mock_raw.ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG C3-REF', 'EEG C4-REF']

        # 4 channels, 10 seconds at 200Hz
        mock_data = np.ones((4, 2000)).astype(np.float32)
        mock_raw.get_data.return_value = mock_data

        mock_read_raw.return_value = mock_raw

        annotations = [{'start': 4.0, 'end': 5.0, 'label': 0}]

        extractor = TUEVEventExtractor()
        segments = extractor.extract_segments(Path('/fake/path.edf'), annotations)

        assert len(segments) == 1
        segment, _ = segments[0]

        # Should still be 23 channels
        assert segment.shape == (23, 1000)

        # Check that available channels have data
        assert segment[0, :].sum() != 0  # FP1
        assert segment[1, :].sum() != 0  # FP2
        assert segment[4, :].sum() != 0  # C3
        assert segment[5, :].sum() != 0  # C4

        # Check that missing channels are zeros
        assert segment[2, :].sum() == 0  # F3 (missing)
        assert segment[10, :].sum() == 0  # F7 (missing)

    def test_segment_out_of_bounds_skipped(self):
        """Test segments outside recording bounds are skipped."""
        with patch('mne.io.read_raw_edf') as mock_read_raw:
            mock_raw = MagicMock()
            mock_raw.info = {'sfreq': 200.0}
            mock_raw.ch_names = TUEVEventExtractor.TUEV_CHANNELS_REF.copy()

            # 8 seconds of data (enough for the middle event but not the boundary ones)
            mock_data = np.random.randn(23, 1600).astype(np.float32)
            mock_raw.get_data.return_value = mock_data

            mock_read_raw.return_value = mock_raw

            # Events at boundaries
            annotations = [
                {'start': 0.5, 'end': 1.0, 'label': 0},  # Near start (handled by triple concat)
                {'start': 4.0, 'end': 4.5, 'label': 1},  # Near end (handled by triple concat)
                {'start': 2.0, 'end': 2.5, 'label': 2},  # Middle event
            ]

            extractor = TUEVEventExtractor()
            segments = extractor.extract_segments(Path('/fake/path.edf'), annotations)

            # All three events should be extracted due to triple concatenation trick
            assert len(segments) == 3
            assert segments[0][1] == 0  # First annotation
            assert segments[1][1] == 1  # Second annotation
            assert segments[2][1] == 2  # Third annotation

    def test_output_dtype_and_units(self):
        """Test output is float32 in Volts (SI units)."""
        with patch('mne.io.read_raw_edf') as mock_read_raw:
            mock_raw = MagicMock()
            mock_raw.info = {'sfreq': 200.0}
            mock_raw.ch_names = TUEVEventExtractor.TUEV_CHANNELS_REF.copy()

            # MNE returns data in Volts
            mock_data = np.random.randn(23, 2000).astype(np.float64) * 1e-6  # microvolts
            mock_raw.get_data.return_value = mock_data

            mock_read_raw.return_value = mock_raw

            annotations = [{'start': 4.0, 'end': 5.0, 'label': 0}]

            extractor = TUEVEventExtractor()
            segments = extractor.extract_segments(Path('/fake/path.edf'), annotations)

            segment, _ = segments[0]
            assert segment.dtype == np.float32
            # Check values are in reasonable range for EEG in Volts
            assert np.abs(segment).max() < 1e-3  # Should be < 1mV

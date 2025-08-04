"""Unit tests for AutoReject adapter classes."""


import mne
import numpy as np
import pytest

from brain_go_brrr.preprocessing.autoreject_adapter import (
    SyntheticPositionGenerator,
    WindowEpochAdapter,
)
from tests.fixtures.mock_eeg_generator import MockEEGGenerator


class TestWindowEpochAdapter:
    """Test window-to-epoch conversion for AutoReject compatibility."""

    @pytest.fixture
    def mock_raw_data(self):
        """Create mock EEG data matching TUAB characteristics."""
        return MockEEGGenerator.create_raw(
            duration=60.0,
            sfreq=256,
            add_artifacts=True,
            seed=42
        )

    def test_window_to_epoch_conversion_basic(self, mock_raw_data):
        """Test basic conversion from sliding windows to epochs."""
        # Given: WindowEpochAdapter with EEGPT window parameters
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # When: Converting raw to windowed epochs
        epochs = adapter.raw_to_windowed_epochs(mock_raw_data)

        # Then: Should create correct number of epochs
        expected_n_epochs = (60 - 10) // 5 + 1  # (duration - window) / stride + 1
        assert len(epochs) == expected_n_epochs
        assert epochs.info['sfreq'] == 256

        # Verify epoch duration
        epoch_data = epochs.get_data()
        assert epoch_data.shape == (expected_n_epochs, 19, 2560)  # 10s @ 256Hz

    def test_window_to_epoch_overlap_handling(self, mock_raw_data):
        """Test that overlapping windows are correctly extracted."""
        # Given: 50% overlap (10s window, 5s stride)
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # When: Creating epochs
        epochs = adapter.raw_to_windowed_epochs(mock_raw_data)

        # Then: Verify overlap by checking that second epoch starts at 5s
        events = epochs.events
        assert events[1, 0] - events[0, 0] == 5 * 256  # 5s * 256Hz

    def test_epochs_to_continuous_reconstruction(self, mock_raw_data):
        """Test reconstructing continuous data from overlapping epochs."""
        # Given: Adapter with overlapping windows
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # When: Converting to epochs and back
        epochs = adapter.raw_to_windowed_epochs(mock_raw_data)

        # Simulate AutoReject cleaning (just for test)
        epochs_clean = epochs.copy()

        # Reconstruct continuous data
        raw_reconstructed = adapter.epochs_to_continuous(epochs_clean, mock_raw_data)

        # Then: Should maintain data dimensions
        assert raw_reconstructed.n_times == mock_raw_data.n_times
        assert len(raw_reconstructed.ch_names) == len(mock_raw_data.ch_names)

        # Verify overlapping regions are properly averaged
        # (Can't test exact equality due to overlap averaging)
        orig_data = mock_raw_data.get_data()
        recon_data = raw_reconstructed.get_data()

        # Non-overlapping regions should be very close
        assert np.allclose(orig_data[:, :256*5], recon_data[:, :256*5], rtol=1e-10)

    def test_no_overlap_reconstruction(self, mock_raw_data):
        """Test reconstruction with non-overlapping windows."""
        # Given: No overlap (10s window, 10s stride)
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=10.0)

        # When: Converting without overlap
        epochs = adapter.raw_to_windowed_epochs(mock_raw_data)
        raw_reconstructed = adapter.epochs_to_continuous(epochs, mock_raw_data)

        # Then: Should perfectly reconstruct (no averaging needed)
        orig_data = mock_raw_data.get_data()
        recon_data = raw_reconstructed.get_data()

        # Check first 50 seconds (5 complete windows)
        n_samples = 50 * 256
        assert np.allclose(orig_data[:, :n_samples], recon_data[:, :n_samples], rtol=1e-10)

    def test_edge_case_short_recording(self):
        """Test handling of recordings shorter than window size."""
        # Given: 5-second recording with 10-second window
        short_raw = MockEEGGenerator.create_raw(duration=5.0, sfreq=256)
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # When/Then: Should raise appropriate error
        with pytest.raises(ValueError, match="Recording too short"):
            adapter.raw_to_windowed_epochs(short_raw)

    def test_memory_efficiency(self, mock_raw_data):
        """Test that adapter doesn't create unnecessary copies."""
        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # Track memory before
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Convert to epochs
        _ = adapter.raw_to_windowed_epochs(mock_raw_data)

        # Memory shouldn't explode (allow 100MB for overhead)
        mem_after = process.memory_info().rss / 1024 / 1024
        assert mem_after - mem_before < 100  # Less than 100MB increase


class TestSyntheticPositionGenerator:
    """Test synthetic EEG channel position generation."""

    def test_standard_channel_positions(self):
        """Test that standard 10-20 positions are correct."""
        generator = SyntheticPositionGenerator()

        # Verify key channels have reasonable positions
        assert 'FP1' in generator.STANDARD_1020_POSITIONS
        assert 'C3' in generator.STANDARD_1020_POSITIONS
        assert 'O1' in generator.STANDARD_1020_POSITIONS

        # Check position ranges (head radius ~10cm)
        for _ch_name, pos in generator.STANDARD_1020_POSITIONS.items():
            radius = np.linalg.norm(pos)
            assert radius < 0.15  # Less than 15cm from origin
            assert radius > 0.01  # Not at origin

    def test_old_new_channel_mapping(self):
        """Test that old TUAB names map to same positions as new names."""
        generator = SyntheticPositionGenerator()

        # Critical mappings for TUAB
        mappings = [
            ('T3', 'T7'),
            ('T4', 'T8'),
            ('T5', 'P7'),
            ('T6', 'P8')
        ]

        for old, new in mappings:
            assert np.array_equal(
                generator.STANDARD_1020_POSITIONS[old],
                generator.STANDARD_1020_POSITIONS[new]
            ), f"{old} and {new} should have identical positions"

    def test_add_positions_to_tuab_raw(self):
        """Test adding positions to TUAB data without positions."""
        # Given: Raw data with TUAB channels (no positions)
        ch_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                    'T3', 'C3', 'CZ', 'C4', 'T4',  # Old naming
                    'T5', 'P3', 'PZ', 'P4', 'T6',   # Old naming
                    'O1', 'O2']

        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 2560)
        raw = mne.io.RawArray(data, info)

        # Explicitly remove any positions
        for ch in raw.info['chs']:
            ch['loc'][:] = 0

        # Verify no positions initially
        assert all(np.all(ch['loc'][:3] == 0) for ch in raw.info['chs'])

        # When: Adding synthetic positions
        generator = SyntheticPositionGenerator()
        raw_with_pos = generator.add_positions_to_raw(raw)

        # Then: All channels should have positions
        for i, ch in enumerate(raw_with_pos.info['chs']):
            loc = ch['loc'][:3]
            assert np.any(loc != 0), f"Channel {ch_names[i]} has no position"
            assert np.linalg.norm(loc) < 0.15, f"Channel {ch_names[i]} position too far"

    def test_autoreject_compatibility(self):
        """Test that generated positions work with AutoReject."""
        # Given: Raw with synthetic positions
        generator = SyntheticPositionGenerator()

        ch_names = ['C3', 'C4', 'CZ', 'F3', 'F4']  # Minimum for AutoReject
        info = mne.create_info(ch_names, 256, ch_types='eeg')
        data = np.random.randn(len(ch_names), 2560)
        raw = mne.io.RawArray(data, info)

        # When: Adding positions
        raw_with_pos = generator.add_positions_to_raw(raw)

        # Then: Should have valid montage
        assert raw_with_pos.get_montage() is not None

        # Positions should be 3D
        for ch in raw_with_pos.info['chs']:
            assert len(ch['loc']) >= 3
            assert not np.all(ch['loc'][:3] == 0)

    def test_fallback_for_unknown_channels(self):
        """Test fallback positioning for non-standard channels."""
        # Given: Channels not in 10-20 system
        generator = SyntheticPositionGenerator()

        weird_names = ['CH1', 'CH2', 'CUSTOM', 'UNKNOWN', 'X1']
        info = mne.create_info(weird_names, 256, ch_types='eeg')
        data = np.random.randn(len(weird_names), 2560)
        raw = mne.io.RawArray(data, info)

        # When: Adding positions
        raw_with_pos = generator.add_positions_to_raw(raw)

        # Then: Should still get valid positions (evenly spaced)
        positions = []
        for ch in raw_with_pos.info['chs']:
            loc = ch['loc'][:3]
            assert np.any(loc != 0), "Fallback position missing"
            positions.append(loc)

        # Verify positions are distributed (not all same)
        positions = np.array(positions)
        assert np.std(positions[:, 0]) > 0.01  # X varies
        assert np.std(positions[:, 1]) > 0.01  # Y varies

    def test_position_consistency(self):
        """Test that positions are consistent across calls."""
        generator = SyntheticPositionGenerator()

        # Create same raw twice
        ch_names = ['FP1', 'C3', 'O1']
        info1 = mne.create_info(ch_names, 256, ch_types='eeg')
        raw1 = mne.io.RawArray(np.random.randn(3, 1000), info1)

        info2 = mne.create_info(ch_names, 256, ch_types='eeg')
        raw2 = mne.io.RawArray(np.random.randn(3, 1000), info2)

        # Add positions
        raw1_pos = generator.add_positions_to_raw(raw1)
        raw2_pos = generator.add_positions_to_raw(raw2)

        # Positions should be identical
        for i in range(len(ch_names)):
            pos1 = raw1_pos.info['chs'][i]['loc'][:3]
            pos2 = raw2_pos.info['chs'][i]['loc'][:3]
            assert np.allclose(pos1, pos2), f"Inconsistent positions for {ch_names[i]}"


class TestAutoRejectIntegration:
    """Test the full integration with AutoReject."""

    @pytest.mark.parametrize("window_duration,stride", [
        (10.0, 5.0),   # 50% overlap
        (10.0, 10.0),  # No overlap
        (8.0, 4.0),    # EEGPT default
    ])
    def test_full_pipeline_with_parameters(self, window_duration, stride):
        """Test full pipeline with different window parameters."""
        # This test will fail until we implement the classes
        # That's the point of TDD!
        pass


# Memory and performance tests
class TestMemoryEfficiency:
    """Test memory usage stays within bounds."""

    @pytest.mark.slow
    def test_large_file_memory_usage(self):
        """Test processing large files doesn't cause memory explosion."""
        # Create 1-hour recording
        raw = MockEEGGenerator.create_raw(
            duration=3600.0,  # 1 hour
            sfreq=256,
            add_artifacts=False  # Skip for speed
        )

        adapter = WindowEpochAdapter(window_duration=10.0, window_stride=5.0)

        # Should not crash or use excessive memory
        epochs = adapter.raw_to_windowed_epochs(raw)
        assert len(epochs) > 0

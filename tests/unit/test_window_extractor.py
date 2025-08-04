"""Test-driven development for sliding window extraction.

Tests for 8-second windows with 50% overlap.
"""

import numpy as np


class TestWindowExtractor:
    """Test sliding window extraction from continuous EEG."""

    def test_extracts_correct_window_size(self):
        """Should extract windows of specified duration."""
        # Given - 30 seconds of data at 256Hz
        sfreq = 256
        duration = 30
        n_samples = duration * sfreq
        n_channels = 20
        data = np.random.randn(n_channels, n_samples)

        # When - extract 8-second windows
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows = extractor.extract(data, sfreq)

        # Then
        expected_samples_per_window = 8 * 256  # 2048 samples
        assert all(w.shape == (n_channels, expected_samples_per_window) for w in windows)

    def test_calculates_correct_number_of_windows(self):
        """Should calculate correct number of windows with overlap."""
        # Given - various durations
        test_cases = [
            (16, 8, 4, 3),  # 16s data, 8s window, 4s overlap -> 3 windows
            (20, 8, 4, 4),  # 20s data, 8s window, 4s overlap -> 4 windows
            (30, 8, 4, 6),  # 30s data, 8s window, 4s overlap -> 6 windows
            (8, 8, 4, 1),  # 8s data, 8s window -> 1 window
            (10, 8, 0, 1),  # 10s data, 8s window, no overlap -> 1 window
        ]

        for duration, window_size, overlap, expected_windows in test_cases:
            # Given
            sfreq = 256
            data = np.random.randn(20, duration * sfreq)

            # When
            from brain_go_brrr.core.window_extractor import WindowExtractor

            extractor = WindowExtractor(window_seconds=window_size, overlap_seconds=overlap)
            windows = extractor.extract(data, sfreq)

            # Then
            assert (
                len(windows) == expected_windows
            ), f"Duration={duration}s, window={window_size}s, overlap={overlap}s"

    def test_handles_50_percent_overlap(self):
        """Should correctly implement 50% overlap between windows."""
        # Given - simple pattern to verify overlap
        sfreq = 256
        8 * sfreq  # 2048
        data = np.arange(20 * sfreq).reshape(1, -1)  # Single channel, incrementing values

        # When
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows = extractor.extract(data, sfreq)

        # Then - verify 50% overlap
        # First window: samples 0-2047
        # Second window: samples 1024-3071 (50% overlap)
        assert windows[0][0, 0] == 0
        assert windows[0][0, -1] == 2047
        assert windows[1][0, 0] == 1024  # 50% overlap
        assert windows[1][0, -1] == 3071

    def test_returns_empty_for_insufficient_data(self):
        """Should return empty list if data shorter than window."""
        # Given - only 5 seconds of data
        sfreq = 256
        data = np.random.randn(20, 5 * sfreq)

        # When - try to extract 8-second windows
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows = extractor.extract(data, sfreq)

        # Then
        assert windows == []

    def test_preserves_channel_order(self):
        """Should maintain channel order in extracted windows."""
        # Given - data with channel-specific patterns
        sfreq = 256
        n_channels = 20
        n_samples = 16 * sfreq
        data = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            data[i, :] = i  # Each channel has constant value = channel index

        # When
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows = extractor.extract(data, sfreq)

        # Then - verify channel order preserved
        for window in windows:
            for ch_idx in range(n_channels):
                assert np.all(window[ch_idx, :] == ch_idx)

    def test_handles_different_sampling_rates(self):
        """Should work correctly with different sampling rates."""
        # Given
        test_rates = [128, 256, 512, 1000]

        for sfreq in test_rates:
            data = np.random.randn(20, 16 * sfreq)  # 16 seconds

            # When
            from brain_go_brrr.core.window_extractor import WindowExtractor

            extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
            windows = extractor.extract(data, sfreq)

            # Then
            expected_samples = int(8 * sfreq)
            assert all(w.shape[1] == expected_samples for w in windows)

    def test_get_window_timestamps(self):
        """Should return correct timestamps for each window."""
        # Given
        sfreq = 256
        duration = 20
        data = np.random.randn(20, duration * sfreq)

        # When
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        windows, timestamps = extractor.extract_with_timestamps(data, sfreq)

        # Then
        expected_starts = [0.0, 4.0, 8.0, 12.0]  # 50% overlap
        assert len(timestamps) == len(windows)
        for i, (start, end) in enumerate(timestamps):
            assert abs(start - expected_starts[i]) < 0.001
            assert abs(end - start - 8.0) < 0.001  # 8-second windows


class TestWindowValidator:
    """Test window validation for quality checks."""

    def test_validates_window_shape(self):
        """Should validate correct window dimensions."""
        # Given
        valid_window = np.random.randn(20, 2048)  # 20 channels, 8s @ 256Hz
        invalid_window = np.random.randn(20, 1024)  # Wrong number of samples

        # When
        from brain_go_brrr.core.window_extractor import WindowValidator

        validator = WindowValidator(expected_channels=20, expected_samples=2048)

        # Then
        assert validator.is_valid(valid_window) is True
        assert validator.is_valid(invalid_window) is False

    def test_detects_corrupted_windows(self):
        """Should detect windows with NaN or Inf values."""
        # Given
        window_with_nan = np.random.randn(20, 2048)
        window_with_nan[5, 100:200] = np.nan

        window_with_inf = np.random.randn(20, 2048)
        window_with_inf[10, 500] = np.inf

        # When
        from brain_go_brrr.core.window_extractor import WindowValidator

        validator = WindowValidator(expected_channels=20, expected_samples=2048)

        # Then
        assert validator.is_valid(window_with_nan) is False
        assert validator.is_valid(window_with_inf) is False


class TestBatchWindowExtractor:
    """Test batch processing of multiple files."""

    def test_extracts_windows_from_multiple_recordings(self):
        """Should extract windows from list of recordings."""
        # Given - 3 recordings of different lengths
        recordings = [
            np.random.randn(20, 16 * 256),  # 16 seconds
            np.random.randn(20, 24 * 256),  # 24 seconds
            np.random.randn(20, 12 * 256),  # 12 seconds
        ]

        # When
        from brain_go_brrr.core.window_extractor import BatchWindowExtractor

        extractor = BatchWindowExtractor(window_seconds=8.0, overlap_seconds=4.0)
        all_windows, recording_indices = extractor.extract_batch(recordings, sfreq=256)

        # Then
        # Recording 0: 16s -> 3 windows
        # Recording 1: 24s -> 5 windows
        # Recording 2: 12s -> 2 windows
        assert len(all_windows) == 3 + 5 + 2
        assert recording_indices == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]

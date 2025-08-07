"""REAL tests for window generation - Critical for training pipeline."""

import numpy as np
import pytest


class TestWindowGenerator:
    """Test sliding window generation for EEG data."""

    def test_window_generation_no_overlap(self):
        """Test window generation without overlap."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        # 10 seconds of data at 256 Hz
        data = np.random.randn(20, 2560)
        
        # 2-second windows, no overlap
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=2.0,
            sfreq=256
        ))
        
        # Should get 5 windows
        assert len(windows) == 5
        
        # Each window should be 2 seconds
        for window in windows:
            assert window.shape == (20, 512)  # 2 * 256
        
        # Windows should not overlap
        # Reconstruct and check
        reconstructed = np.hstack(windows)
        np.testing.assert_array_almost_equal(reconstructed, data)

    def test_window_generation_with_overlap(self):
        """Test window generation with overlap."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        # 5 seconds of data
        data = np.random.randn(20, 1280)
        
        # 2-second windows with 1-second stride (50% overlap)
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=1.0,
            sfreq=256
        ))
        
        # Should get 4 windows
        # 0-2, 1-3, 2-4, 3-5
        assert len(windows) == 4
        
        # Check overlap
        # Second sample of window 1 should equal first sample of window 2
        # (offset by stride)
        stride_samples = 256  # 1 second
        np.testing.assert_array_almost_equal(
            windows[0][:, stride_samples:],
            windows[1][:, :stride_samples]
        )

    def test_window_generation_drop_last(self):
        """Test dropping incomplete last window."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        # 5.5 seconds of data
        data = np.random.randn(20, 1408)
        
        # 2-second windows
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=2.0,
            sfreq=256,
            drop_last=True
        ))
        
        # Should get 2 complete windows, drop the 0.5s remainder
        assert len(windows) == 2

    def test_window_generation_pad_last(self):
        """Test padding incomplete last window."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        # 5.5 seconds of data
        data = np.random.randn(20, 1408)
        
        # 2-second windows
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=2.0,
            sfreq=256,
            drop_last=False,
            pad_last=True
        ))
        
        # Should get 3 windows (last one padded)
        assert len(windows) == 3
        
        # All windows should have same shape
        for window in windows:
            assert window.shape == (20, 512)
        
        # Last window should have padding (zeros)
        # Check last samples are zero
        assert np.allclose(windows[-1][:, -128:], 0)  # Last 0.5s padded

    def test_window_indices(self):
        """Test getting window indices instead of data."""
        from brain_go_brrr.data.window_generator import get_window_indices
        
        # 10 seconds of data
        n_samples = 2560
        
        indices = list(get_window_indices(
            n_samples=n_samples,
            window_size=512,  # 2 seconds at 256 Hz
            stride=256  # 1 second stride
        ))
        
        # Check indices
        expected = [
            (0, 512),
            (256, 768),
            (512, 1024),
            (768, 1280),
            (1024, 1536),
            (1280, 1792),
            (1536, 2048),
            (1792, 2304),
            (2048, 2560)
        ]
        
        assert len(indices) == len(expected)
        for actual, exp in zip(indices, expected):
            assert actual == exp


class TestAugmentedWindowGenerator:
    """Test window generation with augmentation."""

    def test_window_with_jitter(self):
        """Test window generation with temporal jitter."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        np.random.seed(42)
        data = np.random.randn(20, 2560)
        
        # Generate with jitter
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=2.0,
            sfreq=256,
            jitter_samples=10  # ±10 samples random offset
        ))
        
        # Should still get expected number of windows
        assert len(windows) == 5
        
        # Each window should still be correct size
        for window in windows:
            assert window.shape == (20, 512)

    def test_window_with_augmentation(self):
        """Test window generation with data augmentation."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        data = np.ones((20, 2560))  # All ones for testing
        
        def augment_fn(window):
            """Simple augmentation: multiply by 2."""
            return window * 2
        
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=2.0,
            sfreq=256,
            augment_fn=augment_fn
        ))
        
        # Check augmentation applied
        for window in windows:
            assert np.allclose(window, 2.0)


class TestMultiChannelWindowing:
    """Test windowing with channel-specific operations."""

    def test_channel_wise_windowing(self):
        """Test that windowing preserves channel structure."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        # Create data with channel-specific patterns
        n_channels = 20
        n_samples = 1024
        data = np.zeros((n_channels, n_samples))
        
        # Each channel has unique value
        for i in range(n_channels):
            data[i, :] = i
        
        windows = list(generate_windows(
            data,
            window_size=1.0,
            stride=1.0,
            sfreq=256
        ))
        
        # Check channel structure preserved
        for window in windows:
            for i in range(n_channels):
                assert np.allclose(window[i, :], i)

    def test_bad_channel_masking(self):
        """Test windowing with bad channel masking."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        data = np.random.randn(20, 2560)
        bad_channels = [0, 5, 10]  # Channels to mask
        
        windows = list(generate_windows(
            data,
            window_size=2.0,
            stride=2.0,
            sfreq=256,
            bad_channels=bad_channels,
            mask_value=0.0
        ))
        
        # Check bad channels are masked
        for window in windows:
            for ch in bad_channels:
                assert np.allclose(window[ch, :], 0.0)


class TestWindowGeneratorMemory:
    """Test memory-efficient window generation."""

    def test_generator_memory_efficiency(self):
        """Test that generator doesn't load all windows at once."""
        from brain_go_brrr.data.window_generator import generate_windows
        
        # Large data
        data = np.random.randn(20, 25600)  # 100 seconds
        
        # Create generator (not list)
        gen = generate_windows(
            data,
            window_size=4.0,
            stride=2.0,
            sfreq=256
        )
        
        # Should be a generator
        assert hasattr(gen, '__iter__')
        assert hasattr(gen, '__next__')
        
        # Get first window without loading all
        first_window = next(gen)
        assert first_window.shape == (20, 1024)  # 4 seconds

    def test_window_count_calculation(self):
        """Test calculating number of windows without generating."""
        from brain_go_brrr.data.window_generator import calculate_n_windows
        
        # Various data lengths
        test_cases = [
            (2560, 512, 512, 5),   # No overlap
            (2560, 512, 256, 9),   # 50% overlap
            (2560, 1024, 512, 3),  # 50% overlap, larger window
            (2048, 512, 512, 4),   # Exact fit
            (2049, 512, 512, 4),   # One sample extra (drop last)
        ]
        
        for n_samples, window_size, stride, expected in test_cases:
            n_windows = calculate_n_windows(
                n_samples=n_samples,
                window_size=window_size,
                stride=stride,
                drop_last=True
            )
            assert n_windows == expected, f"Failed for {n_samples}, {window_size}, {stride}"
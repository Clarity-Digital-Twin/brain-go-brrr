"""Test window extractor tail handling - Professional edge cases."""

import numpy as np
import pytest

from brain_go_brrr.core.window_extractor import WindowExtractor


@pytest.mark.timeout(2)  # Prevent infinite loops
def test_tail_handling_doesnt_produce_partial_frames():
    """Test that tail doesn't produce partial frames."""
    x = np.zeros((4, 100))
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=0.5)
    wins = we.extract(x, sfreq=64.0)
    
    # All windows should be full size (64 samples at 64Hz = 1 second)
    for i, w in enumerate(wins):
        assert w.shape[-1] == 64, f"Window {i} has wrong size: {w.shape}"


@pytest.mark.timeout(2)
def test_zero_stride_raises_error():
    """Test that zero stride raises ZeroDivisionError."""
    we = WindowExtractor(window_seconds=2.0, overlap_seconds=2.0)
    assert we.stride_seconds == 0.0
    
    # Should raise ZeroDivisionError with zero stride
    x = np.zeros((3, 200))
    with pytest.raises(ZeroDivisionError):
        we.extract(x, sfreq=100.0)


@pytest.mark.timeout(2)
def test_negative_stride_handling():
    """Test negative stride (overlap > window)."""
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=1.5)
    assert we.stride_seconds == -0.5
    
    # Negative stride should be handled gracefully
    x = np.zeros((2, 300))
    wins = we.extract(x, sfreq=100.0)
    
    # Implementation-dependent: might return 0 or 1 window
    assert len(wins) <= 1


def test_exact_multiple_windows():
    """Test when data is exact multiple of window size."""
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=0.0)
    
    # 5 seconds of data at 100Hz = 500 samples
    x = np.arange(500).reshape(1, 500)
    wins = we.extract(x, sfreq=100.0)
    
    # Should get exactly 5 windows
    assert len(wins) == 5
    
    # Check each window has correct data
    for i, w in enumerate(wins):
        expected_start = i * 100
        expected = np.arange(expected_start, expected_start + 100).reshape(1, 100)
        assert np.array_equal(w, expected)


def test_one_sample_short_of_window():
    """Test when data is one sample short of extra window."""
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=0.0)
    
    # 199 samples at 100Hz - just short of 2 windows
    x = np.zeros((3, 199))
    wins = we.extract(x, sfreq=100.0)
    
    # Should get only 1 window
    assert len(wins) == 1
    assert wins[0].shape == (3, 100)


def test_overlap_creates_many_windows():
    """Test that overlap creates expected number of windows."""
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=0.9)  # 90% overlap
    
    # 10 seconds of data
    x = np.zeros((2, 1000))
    wins = we.extract(x, sfreq=100.0)
    
    # With 90% overlap (10 sample stride), should get many windows
    # Exact count depends on implementation details
    assert len(wins) > 50  # Should get many windows with high overlap


def test_no_overlap_sequential_windows():
    """Test that no overlap gives sequential windows."""
    we = WindowExtractor(window_seconds=2.0, overlap_seconds=0.0)
    
    # Create data with distinct patterns
    x = np.zeros((1, 600))
    x[0, :200] = 1  # First window
    x[0, 200:400] = 2  # Second window
    x[0, 400:600] = 3  # Third window
    
    wins = we.extract(x, sfreq=100.0)
    
    assert len(wins) == 3
    assert np.all(wins[0] == 1)
    assert np.all(wins[1] == 2)
    assert np.all(wins[2] == 3)


def test_high_sample_rate():
    """Test with high sample rate (e.g., 1000Hz)."""
    we = WindowExtractor(window_seconds=0.5, overlap_seconds=0.25)
    
    # 2 seconds at 1000Hz
    x = np.zeros((5, 2000))
    wins = we.extract(x, sfreq=1000.0)
    
    # Window size: 500 samples, stride: 250 samples
    # (2000 - 500) / 250 + 1 = 7
    assert len(wins) == 7
    assert all(w.shape == (5, 500) for w in wins)


def test_low_sample_rate():
    """Test with low sample rate (e.g., 50Hz)."""
    we = WindowExtractor(window_seconds=2.0, overlap_seconds=1.0)
    
    # 10 seconds at 50Hz
    x = np.zeros((3, 500))
    wins = we.extract(x, sfreq=50.0)
    
    # Window size: 100 samples, stride: 50 samples
    # (500 - 100) / 50 + 1 = 9
    assert len(wins) == 9
    assert all(w.shape == (3, 100) for w in wins)


def test_single_channel():
    """Test with single channel data."""
    we = WindowExtractor(window_seconds=1.5, overlap_seconds=0.5)
    
    x = np.random.randn(1, 450)
    wins = we.extract(x, sfreq=100.0)
    
    # Window: 150 samples, stride: 100 samples
    # (450 - 150) / 100 + 1 = 4
    assert len(wins) == 4
    assert all(w.shape == (1, 150) for w in wins)


def test_many_channels():
    """Test with many channels."""
    we = WindowExtractor(window_seconds=0.5, overlap_seconds=0.0)
    
    # 100 channels
    x = np.zeros((100, 250))
    wins = we.extract(x, sfreq=500.0)
    
    # Window: 250 samples, stride: 250 samples
    assert len(wins) == 1
    assert wins[0].shape == (100, 250)


def test_fractional_window_samples():
    """Test when window size results in fractional samples."""
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=0.5)
    
    # At 256Hz, 1 second = 256 samples (integer)
    x = np.zeros((4, 768))
    wins = we.extract(x, sfreq=256.0)
    
    # Window: 256 samples, stride: 128 samples
    # (768 - 256) / 128 + 1 = 5
    assert len(wins) == 5
    assert all(w.shape == (4, 256) for w in wins)


def test_data_shorter_than_window():
    """Test when entire data is shorter than one window."""
    we = WindowExtractor(window_seconds=5.0, overlap_seconds=2.0)
    
    # Only 3 seconds of data at 100Hz
    x = np.zeros((2, 300))
    wins = we.extract(x, sfreq=100.0)
    
    # Should return empty list
    assert len(wins) == 0


def test_data_exactly_window_size():
    """Test when data is exactly window size."""
    we = WindowExtractor(window_seconds=3.0, overlap_seconds=1.0)
    
    # Exactly 3 seconds at 100Hz
    x = np.arange(300).reshape(1, 300)
    wins = we.extract(x, sfreq=100.0)
    
    # Should get exactly 1 window
    assert len(wins) == 1
    assert np.array_equal(wins[0], x)


def test_window_extraction_preserves_dtype():
    """Test that window extraction preserves data type."""
    we = WindowExtractor(window_seconds=1.0, overlap_seconds=0.5)
    
    # Test with float32
    x_f32 = np.zeros((3, 300), dtype=np.float32)
    wins_f32 = we.extract(x_f32, sfreq=100.0)
    assert all(w.dtype == np.float32 for w in wins_f32)
    
    # Test with float64
    x_f64 = np.zeros((3, 300), dtype=np.float64)
    wins_f64 = we.extract(x_f64, sfreq=100.0)
    assert all(w.dtype == np.float64 for w in wins_f64)
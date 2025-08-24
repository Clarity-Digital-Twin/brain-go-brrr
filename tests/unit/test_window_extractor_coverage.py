"""Test coverage for window_extractor module - addressing coverage gap.

Heavy hitter module that needs better test coverage.
"""

import numpy as np
import pytest

from brain_go_brrr.domain.preprocessing.window_extractor import WindowExtractor


class TestWindowExtractorCoverage:
    """Comprehensive test coverage for WindowExtractor."""
    
    def test_window_extraction_no_overlap(self):
        """Test basic window extraction without overlap."""
        # 10 seconds of data at 256 Hz
        data = np.random.randn(19, 2560).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=2.0,  # 2 second windows
            overlap=0.0,
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        # Should get 5 windows of 2 seconds each
        assert len(windows) == 5
        assert all(w.shape == (19, 512) for w in windows)
    
    def test_window_extraction_with_overlap(self):
        """Test window extraction with 50% overlap."""
        # 10 seconds of data at 256 Hz
        data = np.random.randn(19, 2560).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=2.0,
            overlap=0.5,  # 50% overlap
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        # With 50% overlap: 9 windows
        assert len(windows) == 9
        assert all(w.shape == (19, 512) for w in windows)
    
    def test_window_extraction_exact_fit(self):
        """Test when data length is exact multiple of window size."""
        # Exactly 4 seconds at 256 Hz
        data = np.random.randn(19, 1024).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.0,
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        assert len(windows) == 1
        assert windows[0].shape == (19, 1024)
    
    def test_window_extraction_partial_last_window(self):
        """Test handling of partial last window."""
        # 4.5 seconds at 256 Hz
        data = np.random.randn(19, 1152).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.0,
            sampling_rate=256,
            drop_last=False  # Keep partial windows
        )
        
        windows = extractor.extract(data)
        
        # Should get 2 windows: one full, one partial
        assert len(windows) == 2
        assert windows[0].shape == (19, 1024)
        assert windows[1].shape == (19, 128)  # Partial window
    
    def test_window_extraction_drop_last(self):
        """Test dropping partial last window."""
        # 4.5 seconds at 256 Hz
        data = np.random.randn(19, 1152).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.0,
            sampling_rate=256,
            drop_last=True  # Drop partial windows
        )
        
        windows = extractor.extract(data)
        
        # Should get only 1 full window
        assert len(windows) == 1
        assert windows[0].shape == (19, 1024)
    
    def test_window_extraction_short_data(self):
        """Test with data shorter than window size."""
        # 2 seconds at 256 Hz (shorter than 4s window)
        data = np.random.randn(19, 512).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.0,
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        # Should get no windows (data too short)
        assert len(windows) == 0
    
    def test_window_extraction_single_channel(self):
        """Test with single channel data."""
        # Single channel, 8 seconds at 256 Hz
        data = np.random.randn(1, 2048).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=2.0,
            overlap=0.0,
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        assert len(windows) == 4
        assert all(w.shape == (1, 512) for w in windows)
    
    def test_window_extraction_different_sampling_rates(self):
        """Test with various sampling rates."""
        sampling_rates = [128, 256, 512, 1000]
        
        for sfreq in sampling_rates:
            # 4 seconds of data
            n_samples = 4 * sfreq
            data = np.random.randn(19, n_samples).astype(np.float32)
            
            extractor = WindowExtractor(
                window_duration=2.0,
                overlap=0.0,
                sampling_rate=sfreq
            )
            
            windows = extractor.extract(data)
            
            assert len(windows) == 2
            assert all(w.shape == (19, 2 * sfreq) for w in windows)
    
    def test_window_extraction_preserves_data(self):
        """Test that windowing preserves original data values."""
        # Create distinctive pattern
        data = np.arange(19 * 1024).reshape(19, 1024).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=2.0,
            overlap=0.0,
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        # Reconstruct and compare
        reconstructed = np.hstack(windows)
        np.testing.assert_array_equal(reconstructed, data[:, :1024])
    
    def test_window_extraction_with_nan(self):
        """Test handling of NaN values in data."""
        data = np.random.randn(19, 1024).astype(np.float32)
        data[5, 100:200] = np.nan
        
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.0,
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        # Should extract windows even with NaN
        assert len(windows) == 1
        # NaN should be preserved
        assert np.isnan(windows[0][5, 100:200]).all()
    
    def test_overlapping_window_indices(self):
        """Test that overlapping windows have correct indices."""
        # 10 seconds at 256 Hz
        data = np.arange(19 * 2560).reshape(19, 2560).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.5,  # 50% overlap = 2 second stride
            sampling_rate=256
        )
        
        windows = extractor.extract(data)
        
        # Check starting indices
        # Window 0: samples 0-1023
        # Window 1: samples 512-1535 (2s stride)
        # Window 2: samples 1024-2047
        # Window 3: samples 1536-2559
        
        assert windows[0][0, 0] == 0
        assert windows[1][0, 0] == 512
        assert windows[2][0, 0] == 1024
        assert windows[3][0, 0] == 1536
    
    def test_edge_case_single_sample(self):
        """Test with single sample (edge case)."""
        data = np.random.randn(19, 1).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=1.0,
            overlap=0.0,
            sampling_rate=1
        )
        
        windows = extractor.extract(data)
        
        # Single sample = one window
        assert len(windows) == 1
        assert windows[0].shape == (19, 1)
    
    def test_high_overlap_ratio(self):
        """Test with very high overlap (90%)."""
        # 10 seconds at 100 Hz for faster test
        data = np.random.randn(19, 1000).astype(np.float32)
        
        extractor = WindowExtractor(
            window_duration=2.0,  # 200 samples
            overlap=0.9,  # 90% overlap = 20 sample stride
            sampling_rate=100
        )
        
        windows = extractor.extract(data)
        
        # (1000 - 200) / 20 + 1 = 41 windows
        assert len(windows) == 41
        assert all(w.shape == (19, 200) for w in windows)


class TestWindowExtractorInit:
    """Test WindowExtractor initialization and validation."""
    
    def test_default_initialization(self):
        """Test default parameters."""
        extractor = WindowExtractor()
        
        assert extractor.window_duration == 4.0
        assert extractor.overlap == 0.5
        assert extractor.sampling_rate == 256
        assert extractor.drop_last is True
    
    def test_custom_initialization(self):
        """Test custom parameters."""
        extractor = WindowExtractor(
            window_duration=8.0,
            overlap=0.25,
            sampling_rate=512,
            drop_last=False
        )
        
        assert extractor.window_duration == 8.0
        assert extractor.overlap == 0.25
        assert extractor.sampling_rate == 512
        assert extractor.drop_last is False
    
    def test_invalid_window_duration(self):
        """Test validation of window duration."""
        with pytest.raises(ValueError, match="window_duration"):
            WindowExtractor(window_duration=0)
        
        with pytest.raises(ValueError, match="window_duration"):
            WindowExtractor(window_duration=-1)
    
    def test_invalid_overlap(self):
        """Test validation of overlap ratio."""
        with pytest.raises(ValueError, match="overlap"):
            WindowExtractor(overlap=-0.1)
        
        with pytest.raises(ValueError, match="overlap"):
            WindowExtractor(overlap=1.0)  # 100% overlap invalid
        
        with pytest.raises(ValueError, match="overlap"):
            WindowExtractor(overlap=1.5)
    
    def test_invalid_sampling_rate(self):
        """Test validation of sampling rate."""
        with pytest.raises(ValueError, match="sampling_rate"):
            WindowExtractor(sampling_rate=0)
        
        with pytest.raises(ValueError, match="sampling_rate"):
            WindowExtractor(sampling_rate=-100)
    
    def test_window_samples_calculation(self):
        """Test calculation of samples per window."""
        extractor = WindowExtractor(
            window_duration=4.0,
            sampling_rate=256
        )
        
        assert extractor.window_samples == 1024
    
    def test_stride_calculation(self):
        """Test calculation of stride between windows."""
        extractor = WindowExtractor(
            window_duration=4.0,
            overlap=0.5,
            sampling_rate=256
        )
        
        # 50% overlap = 512 sample stride
        assert extractor.stride_samples == 512
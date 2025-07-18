"""Tests for EDF streaming utilities."""

import numpy as np
import pytest
import mne
from pathlib import Path
import tempfile

from src.brain_go_brrr.data.edf_streaming import (
    EDFStreamer, 
    estimate_memory_usage,
    process_large_edf
)


class TestEDFStreamer:
    """Test EDF streaming functionality."""
    
    @pytest.fixture
    def small_edf_file(self, tmp_path):
        """Create a small EDF file for testing."""
        # Create 1 minute of data
        sfreq = 256
        duration = 60
        n_channels = 19
        
        data = np.random.randn(n_channels, sfreq * duration) * 50
        
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        edf_path = tmp_path / "test_small.edf"
        # Scale data for EDF export
        raw._data = raw._data / 1e6
        raw.export(edf_path, fmt='edf', overwrite=True, physical_range=(-200, 200))
        
        return edf_path
    
    @pytest.fixture
    def large_edf_file(self, tmp_path):
        """Create a larger EDF file for testing streaming."""
        # Create 10 minutes of data
        sfreq = 256
        duration = 600  # 10 minutes
        n_channels = 32
        
        # Don't actually create all data at once to save test memory
        # Just create metadata
        ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Create file in chunks to avoid memory issues in tests
        edf_path = tmp_path / "test_large.edf"
        
        # Create small chunk and save
        chunk_data = np.random.randn(n_channels, sfreq * 60) * 50  # 1 minute
        raw = mne.io.RawArray(chunk_data, info)
        raw._data = raw._data / 1e6
        raw.export(edf_path, fmt='edf', overwrite=True, physical_range=(-200, 200))
        
        return edf_path
    
    def test_streamer_context_manager(self, small_edf_file):
        """Test streamer works as context manager."""
        with EDFStreamer(small_edf_file) as streamer:
            assert streamer._raw is not None
            assert streamer._sfreq == 256
            assert streamer._n_channels == 19
            assert streamer._duration > 0
        
        # Should be closed after context
        assert streamer._raw is None
    
    def test_stream_chunks(self, small_edf_file):
        """Test streaming data in chunks."""
        chunk_duration = 10.0  # 10 second chunks
        
        with EDFStreamer(small_edf_file, chunk_duration=chunk_duration) as streamer:
            chunks = list(streamer.stream_chunks())
            
        # Should have 6 chunks for 60 seconds
        assert len(chunks) == 6
        
        # Check first chunk
        data, start_time = chunks[0]
        assert data.shape == (19, 2560)  # 19 channels, 10s * 256Hz
        assert start_time == 0.0
        
        # Check last chunk
        data, start_time = chunks[-1]
        assert start_time == pytest.approx(50.0, rel=0.1)
    
    def test_process_in_windows(self, small_edf_file):
        """Test processing in analysis windows."""
        window_duration = 4.0
        
        with EDFStreamer(small_edf_file, chunk_duration=30.0) as streamer:
            windows = list(streamer.process_in_windows(window_duration=window_duration))
        
        # 60 seconds / 4 seconds = 15 windows (non-overlapping)
        assert len(windows) == 15
        
        # Check window shape
        data, start_time = windows[0]
        assert data.shape == (19, 1024)  # 19 channels, 4s * 256Hz
    
    def test_process_in_windows_with_overlap(self, small_edf_file):
        """Test processing with overlapping windows."""
        window_duration = 4.0
        overlap = 0.5  # 50% overlap
        
        with EDFStreamer(small_edf_file, chunk_duration=30.0) as streamer:
            windows = list(streamer.process_in_windows(
                window_duration=window_duration,
                overlap=overlap
            ))
        
        # With 50% overlap: ~29 windows for 60 seconds
        assert len(windows) >= 28
        
        # Check overlap by comparing start times
        _, time1 = windows[0]
        _, time2 = windows[1]
        assert (time2 - time1) == pytest.approx(2.0, rel=0.01)  # 2s step with 50% overlap
    
    def test_get_info(self, small_edf_file):
        """Test getting file info without loading data."""
        info = EDFStreamer(small_edf_file).get_info()
        
        assert info['n_channels'] == 19
        assert info['sampling_rate'] == 256
        assert info['duration'] == pytest.approx(60.0, rel=0.1)
        assert len(info['channel_names']) == 19
    
    def test_invalid_overlap(self, small_edf_file):
        """Test error handling for invalid overlap."""
        with EDFStreamer(small_edf_file) as streamer:
            with pytest.raises(ValueError):
                list(streamer.process_in_windows(overlap=1.5))
            
            with pytest.raises(ValueError):
                list(streamer.process_in_windows(overlap=-0.1))


class TestMemoryEstimation:
    """Test memory estimation utilities."""
    
    def test_estimate_memory_usage(self, tmp_path):
        """Test memory usage estimation."""
        # Create known-size file
        sfreq = 256
        duration = 60
        n_channels = 19
        n_samples = sfreq * duration
        
        data = np.random.randn(n_channels, n_samples).astype(np.float64)
        info = mne.create_info(
            ch_names=[f'CH{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels
        )
        raw = mne.io.RawArray(data, info)
        
        edf_path = tmp_path / "test_memory.edf"
        raw._data = raw._data / 1e6
        raw.export(edf_path, fmt='edf', overwrite=True, physical_range=(-200, 200))
        
        # Test estimation
        estimate = estimate_memory_usage(edf_path, preload=True)
        
        # Raw data size: 19 * 15360 * 8 bytes = 2.34 MB
        expected_raw_mb = (n_channels * n_samples * 8) / (1024 * 1024)
        assert estimate['raw_data_mb'] == pytest.approx(expected_raw_mb, rel=0.01)
        
        # With overhead
        assert estimate['estimated_total_mb'] > estimate['raw_data_mb']
        
        # Without preload
        estimate_no_preload = estimate_memory_usage(edf_path, preload=False)
        assert estimate_no_preload['estimated_total_mb'] < estimate['estimated_total_mb']
    
    def test_process_large_edf_streaming_decision(self, tmp_path):
        """Test that large files trigger streaming mode."""
        # Create a file that appears large based on metadata
        sfreq = 256
        duration = 60
        n_channels = 64  # More channels to increase size
        
        data = np.random.randn(n_channels, sfreq * duration) * 50
        info = mne.create_info(
            ch_names=[f'CH{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels
        )
        raw = mne.io.RawArray(data, info)
        
        edf_path = tmp_path / "test_large.edf"
        raw.export(edf_path, fmt='edf', overwrite=True)
        
        # Process with low memory limit
        results = process_large_edf(edf_path, max_memory_mb=5.0)
        
        # Should use streaming for this "large" file
        assert results['used_streaming'] is True
        assert results['chunks_processed'] > 0
    
    def test_process_small_edf_no_streaming(self, tmp_path):
        """Test that small files don't trigger streaming."""
        # Create small file
        sfreq = 256
        duration = 10
        n_channels = 4
        
        data = np.random.randn(n_channels, sfreq * duration) * 50
        info = mne.create_info(
            ch_names=[f'CH{i}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types=['eeg'] * n_channels
        )
        raw = mne.io.RawArray(data, info)
        
        edf_path = tmp_path / "test_tiny.edf"
        raw.export(edf_path, fmt='edf', overwrite=True)
        
        # Process with high memory limit
        results = process_large_edf(edf_path, max_memory_mb=1000.0)
        
        # Should not use streaming
        assert results['used_streaming'] is False
        assert results['chunks_processed'] == 1
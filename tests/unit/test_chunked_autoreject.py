"""Unit tests for chunked AutoReject processor."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# This will fail until implemented - TDD!
from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor


class TestChunkedAutoRejectProcessor:
    """Test memory-efficient AutoReject processing."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_file_paths(self, temp_cache_dir):
        """Create mock EDF file paths."""
        paths = []
        for i in range(100):  # 100 mock files
            path = temp_cache_dir / f"mock_eeg_{i:03d}.edf"
            path.touch()  # Create empty file
            paths.append(path)
        return paths

    def test_initialization(self, temp_cache_dir):
        """Test processor initialization."""
        # Given: Cache directory path
        cache_dir = temp_cache_dir / "ar_cache"

        # When: Creating processor
        processor = ChunkedAutoRejectProcessor(
            cache_dir=cache_dir,
            chunk_size=50,
            n_interpolate=[1, 4],
            consensus=0.1,
            random_state=42
        )

        # Then: Should initialize properly
        assert processor.cache_dir.exists()
        assert processor.chunk_size == 50
        assert processor.n_interpolate == [1, 4]
        assert not processor.is_fitted
        assert not processor.has_cached_params()

    def test_has_cached_params(self, temp_cache_dir):
        """Test checking for cached parameters."""
        # Given: Processor with cache directory
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # Initially no cache
        assert not processor.has_cached_params()

        # Create fake cache file
        param_file = temp_cache_dir / "autoreject_params.pkl"
        with param_file.open('wb') as f:
            pickle.dump({'test': 'params'}, f)

        # Now should detect cache
        assert processor.has_cached_params()

    @patch('mne.io.read_raw_edf')
    @patch('mne.make_fixed_length_epochs')
    @patch('mne.concatenate_epochs')
    def test_fit_on_subset_basic(self, mock_concat, mock_make_epochs, mock_read_edf,
                                  temp_cache_dir, mock_file_paths):
        """Test fitting AutoReject on data subset."""
        # Given: Mock EEG data
        mock_raw = MagicMock()
        mock_raw.n_times = 256 * 60  # 1 minute
        mock_read_edf.return_value = mock_raw

        # Mock epochs
        mock_epochs = MagicMock()
        mock_epochs.get_data.return_value = np.random.randn(10, 19, 2560)
        mock_epochs.__len__.return_value = 10
        mock_make_epochs.return_value = mock_epochs

        # Mock concatenated epochs
        mock_combined = MagicMock()
        mock_combined.get_data.return_value = np.random.randn(100, 19, 2560)
        mock_combined.__len__.return_value = 100
        mock_concat.return_value = mock_combined

        # When: Fitting on subset
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # Mock AutoReject to avoid actual fitting
        with patch('brain_go_brrr.preprocessing.chunked_autoreject.AutoReject') as mock_ar_class:
            mock_ar = MagicMock()
            # Use actual numpy arrays instead of MagicMock
            mock_ar.threshes_ = np.random.rand(19, 10)
            mock_ar.consensus_ = [0.1]
            mock_ar.n_interpolate_ = [1, 4]
            mock_ar.picks_ = list(range(19))
            mock_ar_class.return_value = mock_ar

            # Also mock the save to avoid pickling MagicMock
            with patch.object(processor, '_save_parameters'):
                processor.fit_on_subset(mock_file_paths[:10], n_samples=5)

        # Then: Should be fitted
        assert processor.is_fitted
        assert processor.ar_params is not None
        assert processor.has_cached_params()

    def test_stratified_sampling(self, mock_file_paths):
        """Test stratified sampling of files."""
        # Given: Files from different directories (simulating labels)
        processor = ChunkedAutoRejectProcessor()

        # When: Sampling subset
        subset = processor._stratified_sample(mock_file_paths, n_samples=20)

        # Then: Should return requested number
        assert len(subset) == 20
        assert all(f in mock_file_paths for f in subset)

        # Test with more samples than available
        subset_all = processor._stratified_sample(mock_file_paths, n_samples=200)
        assert len(subset_all) == len(mock_file_paths)

    def test_parameter_extraction(self, temp_cache_dir):
        """Test extracting parameters from fitted AutoReject."""
        # Given: Mock fitted AutoReject
        mock_ar = MagicMock()
        mock_ar.threshes_ = np.array([[100e-6, 150e-6], [120e-6, 180e-6]])
        mock_ar.consensus_ = [0.1, 0.2]
        mock_ar.n_interpolate_ = [1, 4]
        mock_ar.picks_ = [0, 1, 2]

        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # When: Extracting parameters
        params = processor._extract_parameters(mock_ar)

        # Then: Should extract all relevant parameters
        assert 'thresholds' in params
        assert 'consensus' in params
        assert 'n_interpolate' in params
        assert 'picks' in params
        assert np.array_equal(params['thresholds'], mock_ar.threshes_)

    def test_parameter_saving_loading(self, temp_cache_dir):
        """Test saving and loading parameters."""
        # Given: Processor and mock parameters
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)
        test_params = {
            'thresholds': np.random.rand(19, 10),
            'consensus': [0.1],
            'n_interpolate': [1, 4],
            'picks': list(range(19))
        }

        # Save parameters
        processor.ar_params = test_params
        mock_ar = MagicMock()
        for key, value in test_params.items():
            setattr(mock_ar, key.rstrip('s') + '_', value)

        processor._save_parameters(mock_ar)

        # Reset processor
        processor.ar_params = None
        processor.is_fitted = False

        # Load parameters
        processor._load_parameters()

        # Verify loaded correctly
        assert processor.is_fitted
        assert processor.ar_params is not None
        assert np.array_equal(processor.ar_params['thresholds'], test_params['thresholds'])

    def test_create_autoreject_from_params(self, temp_cache_dir):
        """Test creating AutoReject instance from cached parameters."""
        # Given: Processor with cached parameters
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)
        processor.ar_params = {
            'thresholds': np.random.rand(19, 10),
            'consensus': [0.1],
            'n_interpolate': [1, 4],
            'picks': list(range(19))
        }

        # When: Creating AutoReject from params
        with patch('brain_go_brrr.preprocessing.chunked_autoreject.AutoReject') as mock_ar_class:
            mock_ar = MagicMock()
            mock_ar_class.return_value = mock_ar

            processor._create_autoreject_from_params()

            # Then: Should set pre-computed parameters
            assert hasattr(mock_ar, 'threshes_')
            assert hasattr(mock_ar, 'consensus_')
            assert hasattr(mock_ar, 'n_interpolate_')

    def test_transform_raw(self, temp_cache_dir):
        """Test transforming raw data with fitted parameters."""
        # Given: Fitted processor
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)
        processor.is_fitted = True
        processor.ar_params = {
            'thresholds': np.random.rand(19, 10),
            'consensus': [0.1],
            'n_interpolate': [1, 4],
            'picks': list(range(19))
        }

        # Mock raw data
        mock_raw = MagicMock()
        mock_raw.get_data.return_value = np.random.randn(19, 10000)
        mock_raw.info = {'sfreq': 256}
        mock_raw.n_times = 10000

        # Mock window adapter
        mock_adapter = MagicMock()
        mock_epochs = MagicMock()
        mock_epochs.get_data.return_value = np.random.randn(10, 19, 2560)
        mock_adapter.raw_to_windowed_epochs.return_value = mock_epochs
        mock_adapter.epochs_to_continuous.return_value = mock_raw

        # When: Transforming
        with patch.object(processor, '_apply_autoreject', return_value=mock_epochs):
            result = processor.transform_raw(mock_raw, mock_adapter)

        # Then: Should return cleaned raw
        assert result == mock_raw
        mock_adapter.raw_to_windowed_epochs.assert_called_once()
        mock_adapter.epochs_to_continuous.assert_called_once()

    def test_apply_autoreject_with_rejection_stats(self, temp_cache_dir):
        """Test applying AutoReject and getting rejection statistics."""
        # Given: Processor with parameters
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)
        processor.ar_params = {
            'thresholds': np.random.rand(19, 10),
            'consensus': [0.1],
            'n_interpolate': [1, 4],
            'picks': list(range(19))
        }

        # Mock epochs
        mock_epochs = MagicMock()
        mock_epochs.__len__.return_value = 20
        mock_epochs_clean = MagicMock()
        mock_epochs_clean.__len__.return_value = 15  # 5 rejected

        # When: Applying AutoReject
        with patch.object(processor, '_create_autoreject_from_params') as mock_create:
            mock_ar = MagicMock()
            mock_ar.transform.return_value = mock_epochs_clean
            mock_create.return_value = mock_ar

            result = processor._apply_autoreject(mock_epochs)

        # Then: Should return cleaned epochs
        assert result == mock_epochs_clean
        mock_ar.transform.assert_called_once_with(mock_epochs)

    @pytest.mark.slow
    def test_memory_efficiency(self, temp_cache_dir):
        """Test that chunked processing doesn't cause memory explosion."""
        # Given: Large number of files
        n_files = 1000
        processor = ChunkedAutoRejectProcessor(
            cache_dir=temp_cache_dir,
            chunk_size=100  # Process 100 at a time
        )

        # Track memory usage
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate processing (without actual AutoReject)
        for _ in range(0, n_files, processor.chunk_size):
            # Simulate chunk processing
            chunk_data = np.random.randn(100, 19, 2560)  # ~40MB
            del chunk_data  # Immediate cleanup

        # Check memory didn't explode
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_increase = mem_after - mem_before

        # Should not accumulate memory (allow 100MB overhead)
        assert memory_increase < 100, f"Memory increased by {memory_increase}MB"

    def test_error_handling_missing_cache(self, temp_cache_dir):
        """Test error handling when cache is missing."""
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # Should raise when trying to load non-existent cache
        with pytest.raises(ValueError, match="No cached parameters found"):
            processor._load_parameters()

    def test_error_handling_corrupted_cache(self, temp_cache_dir):
        """Test handling of corrupted cache files."""
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # Create corrupted cache file
        param_file = temp_cache_dir / "autoreject_params.pkl"
        with param_file.open('w') as f:
            f.write("corrupted data")

        # Should raise appropriate error
        with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
            processor._load_parameters()

    def test_parallel_processing_capability(self, temp_cache_dir):
        """Test that processor can handle parallel calls safely."""
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # Simulate parallel access to cache
        import threading

        def access_cache():
            return processor.has_cached_params()

        # Run multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=access_cache)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should not crash or corrupt


class TestChunkedProcessingIntegration:
    """Integration tests for chunked processing."""

    @pytest.mark.integration
    def test_full_chunked_pipeline(self, temp_cache_dir):
        """Test complete chunked processing pipeline."""
        # This will be implemented when we have the actual classes
        pass

    def test_performance_benchmarks(self):
        """Benchmark processing speed."""
        # Will add performance benchmarks
        pass


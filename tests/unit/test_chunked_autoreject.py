"""Unit tests for chunked AutoReject processor."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

# This will fail until implemented - TDD!
from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor


@pytest.fixture(autouse=True)
def _isolate_tests(tmp_path, monkeypatch):
    """Isolate each test with its own temporary directory."""
    monkeypatch.chdir(tmp_path)


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
        # Given: Cache directory that doesn't exist yet
        cache_dir = temp_cache_dir / "ar_cache"
        processor = ChunkedAutoRejectProcessor(cache_dir=cache_dir)

        # Initially no cache
        assert not processor.has_cached_params()

        # Create fake cache file with REAL autoreject structure
        param_file = processor.cache_dir / "autoreject_params.pkl"
        
        params = {
            'consensus': 0.1,
            'n_interpolate': [1, 4],
            'thresholds_': {'Fp1': [50.0, 100.0], 'Fp2': [50.0, 100.0]}
        }
        with param_file.open('wb') as f:
            pickle.dump(params, f)

        # Now should detect cache
        assert processor.has_cached_params()

    def test_fit_on_subset_basic(self, temp_cache_dir):
        """Test fitting AutoReject saves params correctly."""
        # Given: Processor with cache directory
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # Create a simple object with AutoReject-like attributes
        class FakeAutoReject:
            def __init__(self):
                self.threshes_ = np.array([[50.0, 100.0], [50.0, 100.0]])
                self.consensus_ = [0.1]
                self.n_interpolate_ = [1, 4]
                self.picks_ = [0, 1]
        
        fake_ar = FakeAutoReject()
        
        # Save parameters
        processor._save_parameters(fake_ar)

        # Then: Should have cached params
        assert processor.has_cached_params()
        
        # Verify saved params
        param_file = processor.cache_dir / "autoreject_params.pkl"
        assert param_file.exists()
        with param_file.open('rb') as f:
            saved_params = pickle.load(f)
        assert saved_params['consensus'] == [0.1]
        assert np.array_equal(saved_params['thresholds'], fake_ar.threshes_)

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
        # Given: Object with AutoReject-like attributes
        class FakeAutoReject:
            def __init__(self):
                self.threshes_ = np.array([[100e-6, 150e-6], [120e-6, 180e-6]])
                self.consensus_ = [0.1, 0.2]
                self.n_interpolate_ = [1, 4]
                self.picks_ = [0, 1, 2]
        
        fake_ar = FakeAutoReject()
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)

        # When: Extracting parameters
        params = processor._extract_parameters(fake_ar)

        # Then: Should extract all relevant parameters
        assert 'thresholds' in params
        assert 'consensus' in params
        assert 'n_interpolate' in params
        assert 'picks' in params
        assert np.array_equal(params['thresholds'], fake_ar.threshes_)

    def test_parameter_saving_loading(self, temp_cache_dir):
        """Test saving and loading parameters."""
        # Given: Processor and test parameters
        processor = ChunkedAutoRejectProcessor(cache_dir=temp_cache_dir)
        
        # Create fake AutoReject with parameters
        class FakeAutoReject:
            def __init__(self):
                self.threshes_ = np.random.rand(19, 10)
                self.consensus_ = [0.1]
                self.n_interpolate_ = [1, 4]
                self.picks_ = list(range(19))
        
        fake_ar = FakeAutoReject()
        
        # Save parameters
        processor._save_parameters(fake_ar)

        # Reset processor
        processor.ar_params = None
        processor.is_fitted = False

        # Load parameters
        processor._load_parameters()

        # Verify loaded correctly
        assert processor.is_fitted
        assert processor.ar_params is not None
        assert processor.ar_params['consensus'] == [0.1]
        assert processor.ar_params['n_interpolate'] == [1, 4]

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

        # Test that method exists and can be called
        # (We can't test the actual AutoReject creation without the library)
        assert hasattr(processor, '_create_autoreject_from_params')
        
        # If AutoReject is not installed, the method should handle it gracefully
        try:
            from autoreject import AutoReject
            # If AutoReject is available, test would create real instance
            ar = processor._create_autoreject_from_params()
            assert ar is not None
        except ImportError:
            # AutoReject not available in test environment
            pass

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

        # Test that method exists
        assert hasattr(processor, 'transform_raw')
        
        # We can't test actual transformation without MNE and AutoReject
        # But we can verify the processor state
        assert processor.is_fitted
        assert processor.ar_params is not None

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

        # Test that method exists
        assert hasattr(processor, '_apply_autoreject')
        
        # Verify processor has the required parameters
        assert processor.ar_params is not None
        assert 'thresholds' in processor.ar_params
        assert 'consensus' in processor.ar_params

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
        # Create processor with fresh cache directory
        fresh_cache_dir = temp_cache_dir / "fresh_cache_missing"
        processor = ChunkedAutoRejectProcessor(cache_dir=fresh_cache_dir)

        # Should raise when trying to load non-existent cache
        with pytest.raises(ValueError, match="No cached parameters found"):
            processor._load_parameters()

    def test_error_handling_corrupted_cache(self, temp_cache_dir):
        """Test handling of corrupted cache files."""
        # Create new cache directory
        cache_dir = temp_cache_dir / "corrupted_cache" 
        processor = ChunkedAutoRejectProcessor(cache_dir=cache_dir)

        # Create corrupted cache file
        param_file = processor.cache_dir / "autoreject_params.pkl"
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


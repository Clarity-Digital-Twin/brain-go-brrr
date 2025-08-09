"""Performance benchmark tests for Brain-Go-Brrr.

These tests are marked with @pytest.mark.perf and should be run separately
in CI to avoid slowing down the main test suite.
"""

import os

import numpy as np
import psutil
import pytest

from brain_go_brrr.models.eegpt_model import EEGPTModel


@pytest.mark.perf
class TestPerformanceBenchmarks:
    """Performance benchmarks for production requirements."""

    @pytest.fixture
    def eegpt_model(self):
        """Load EEGPT model for performance tests."""
        from brain_go_brrr.core.config import ModelConfig
        from brain_go_brrr.models.eegpt_wrapper import create_normalized_eegpt

        config = ModelConfig(device="cpu")
        model = EEGPTModel(config=config, auto_load=False)
        # Create architecture without checkpoint - use wrapper for proper API
        model.encoder = create_normalized_eegpt(checkpoint_path=None, normalize=False)
        model.encoder.to(model.device)

        # Initialize abnormality head using helper (required for predict_abnormality)
        if model.abnormality_head is None:
            model.abnormality_head = model._create_abnormality_head()
        model.abnormality_head.eval()  # Set to eval mode

        model.is_loaded = True
        return model

    @pytest.mark.slow
    def test_inference_speed(self, eegpt_model, benchmark):
        """Test inference speed meets requirements."""
        import mne

        # Test with 2-minute recording (sufficient to measure performance)
        # Full 20-minute test would be done in dedicated benchmarks
        duration = 2 * 60  # seconds
        sfreq = 256
        n_channels = 19
        data = np.random.randn(n_channels, int(sfreq * duration)) * 50e-6

        # Create MNE Raw object
        ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Use pytest-benchmark for accurate timing
        benchmark(eegpt_model.predict_abnormality, raw)

        # Should process at a rate that would complete 20-min in <2 minutes
        # Expected: 2 min data should process in <12 seconds (proportional)
        assert benchmark.stats["mean"] < 12, (
            f"Processing too slow: {benchmark.stats['mean']:.2f}s mean time"
        )

    @pytest.mark.slow
    def test_memory_usage(self, eegpt_model, benchmark):
        """Test memory usage is reasonable."""
        import mne

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Test with 5-minute recording (sufficient to detect memory leaks)
        # Full 30-minute test would be done in dedicated benchmarks
        duration = 5 * 60  # seconds
        sfreq = 256
        n_channels = 58  # Max channels
        data = np.random.randn(n_channels, sfreq * duration) * 50e-6

        # Create MNE Raw object
        ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Process with memory tracking
        def process_and_track():
            _ = eegpt_model.predict_abnormality(raw)
            return process.memory_info().rss / 1024 / 1024  # MB

        final_memory = benchmark(process_and_track)
        memory_increase = final_memory - initial_memory

        # Should use reasonable memory (proportionally less than 4GB for full 30min)
        # Expected: ~2-2.5GB for 5 minutes with 1GB model loaded
        # Allow some overhead for Python/PyTorch memory management
        assert memory_increase < 2500, f"Memory usage too high: {memory_increase:.2f} MB"

    @pytest.mark.perf
    def test_api_response_time(self, client, benchmark):
        """Test API response time requirement (NFR1.4: <100ms response time)."""
        # Note: This tests the endpoint itself, not the processing
        from fastapi.testclient import TestClient

        from brain_go_brrr.api.main import app

        test_client = TestClient(app)

        # Benchmark the health check endpoint
        benchmark(test_client.get, "/api/v1/health")

        # Should respond in <100ms (allow some buffer for test environment)
        assert benchmark.stats["mean"] < 0.1, (
            f"API response too slow: {benchmark.stats['mean'] * 1000:.2f}ms mean time"
        )

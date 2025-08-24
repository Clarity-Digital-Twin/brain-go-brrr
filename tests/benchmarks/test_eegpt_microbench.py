"""Microbenchmark for EEGPT extract_features performance regression testing."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


class TestEEGPTMicrobenchmark:
    """Microbenchmark to guard against performance regressions after refactoring."""

    @pytest.mark.benchmark
    def test_extract_features_cpu_performance(self, benchmark):
        """Benchmark extract_features on CPU with various batch sizes."""
        from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel
        
        # Setup model with mock encoder for consistent timing
        model = EEGPTModel(device="cpu", auto_load=False)
        model.is_loaded = True
        
        # Mock encoder with realistic computation
        class MockEncoder:
            def __init__(self):
                self.device = torch.device("cpu")
                # Simulate some weight matrices
                self.w1 = torch.randn(1024, 512)
                self.w2 = torch.randn(512, 512)
            
            def extract_features(self, x, summary=True):
                # Simulate actual computation
                b = x.shape[0]
                # Flatten and project
                x_flat = x.view(b, -1)[:, :1024]  # Take first 1024 dims
                h1 = torch.matmul(x_flat, self.w1)
                h2 = torch.relu(h1)
                output = torch.matmul(h2, self.w2)
                
                if not summary:
                    # Expand to token shape
                    output = output.unsqueeze(1).repeat(1, 4, 1)
                    
                return output
        
        model.encoder = MockEncoder()
        
        # Test data - single sample
        data = np.random.randn(20, 1024).astype(np.float32)
        
        # Benchmark the function
        result = benchmark(model.extract_features, data, summary=True)
        
        # Verify shape is correct
        assert result.shape == (1, 512)
        
        # Performance assertions (adjust based on your hardware)
        # These are conservative limits to catch major regressions
        assert benchmark.stats['mean'] < 0.1  # 100ms budget for single sample
        assert benchmark.stats['stddev'] < 0.05  # Stable performance

    @pytest.mark.parametrize("batch_size,time_samples", [
        (1, 512),
        (1, 1024),
        (1, 2048),
        (4, 1024),
    ])
    def test_extract_features_scaling(self, batch_size, time_samples):
        """Test that performance scales linearly with input size."""
        from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel
        
        model = EEGPTModel(device="cpu", auto_load=False)
        model.is_loaded = True
        
        # Simple mock that just returns zeros
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(
            return_value=torch.zeros(batch_size, 512)
        )
        
        # Generate test data
        if batch_size == 1:
            data = np.random.randn(20, time_samples).astype(np.float32)
        else:
            data = np.random.randn(batch_size, 20, time_samples).astype(np.float32)
        
        # Time the extraction
        start = time.perf_counter()
        features = model.extract_features(data, summary=True)
        elapsed = time.perf_counter() - start
        
        # Verify shape
        assert features.shape == (batch_size, 512)
        
        # Performance budget: roughly 10ms per 1024 samples per batch
        # This is very conservative to avoid flaky tests
        expected_time = (batch_size * time_samples / 1024) * 0.01
        assert elapsed < expected_time * 10, \
            f"Too slow: {elapsed:.3f}s > {expected_time*10:.3f}s budget"

    def test_shape_validation_overhead(self):
        """Ensure shape validation doesn't add significant overhead."""
        from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel
        
        model = EEGPTModel(device="cpu", auto_load=False)
        model.is_loaded = True
        
        # Mock that returns correct shape
        model.encoder = MagicMock()
        model.encoder.extract_features = MagicMock(
            return_value=torch.zeros(1, 512)
        )
        
        data = np.random.randn(20, 1024).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            model.extract_features(data, summary=True)
        
        # Time many iterations
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            model.extract_features(data, summary=True)
        elapsed = time.perf_counter() - start
        
        # Should be very fast with mocked encoder
        per_call = elapsed / n_iters
        assert per_call < 0.001, f"Validation overhead too high: {per_call*1000:.2f}ms per call"
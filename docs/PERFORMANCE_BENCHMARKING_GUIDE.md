# Performance Benchmarking Guide for Brain-Go-Brrr

## Executive Summary

This guide provides comprehensive performance benchmarking specifications for the Brain-Go-Brrr EEG analysis pipeline. We define metrics, targets, and testing procedures to ensure the system meets real-time processing requirements while maintaining accuracy.

## Performance Requirements Overview

### Primary Targets
- **Processing Speed**: 20-minute EEG in <2 minutes
- **Concurrent Users**: Support 50 simultaneous analyses
- **API Response**: <100ms for all endpoints
- **Memory Usage**: <4GB peak per analysis
- **GPU Utilization**: >80% during inference
- **Cache Hit Rate**: >90% for repeated analyses

### Model-Specific Targets

| Component | Accuracy Target | Speed Target | Memory Budget |
|-----------|----------------|--------------|---------------|
| EEGPT | - | 32 windows/sec | 1.5GB |
| AutoReject | 87.5% agreement | <5s per recording | 500MB |
| Abnormal Detection | 80% BAC, 0.93 AUROC | <1s per window | 200MB |
| YASA Sleep | 87.46% accuracy | <30s per hour | 1GB |
| Event Detection | 80% sensitivity | <2s per minute | 300MB |

## Benchmarking Framework

### 1. Performance Test Suite
```python
# tests/benchmarks/test_performance.py
import pytest
import time
import psutil
import torch
from memory_profiler import profile
from pathlib import Path
import numpy as np

class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            "duration": [],
            "memory_peak": [],
            "gpu_memory": [],
            "cpu_percent": []
        }
    
    def measure(self, func, *args, **kwargs):
        """Measure performance of a function."""
        # Pre-measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Measurement
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Post-measurement
        duration = end_time - start_time
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        gpu_memory = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Record metrics
        self.metrics["duration"].append(duration)
        self.metrics["memory_peak"].append(memory_increase)
        self.metrics["gpu_memory"].append(gpu_memory)
        self.metrics["cpu_percent"].append(process.cpu_percent())
        
        return result, duration, memory_increase, gpu_memory
    
    def report(self):
        """Generate performance report."""
        return {
            "name": self.name,
            "avg_duration": np.mean(self.metrics["duration"]),
            "std_duration": np.std(self.metrics["duration"]),
            "p95_duration": np.percentile(self.metrics["duration"], 95),
            "avg_memory": np.mean(self.metrics["memory_peak"]),
            "max_memory": np.max(self.metrics["memory_peak"]),
            "avg_gpu_memory": np.mean(self.metrics["gpu_memory"]),
            "throughput": len(self.metrics["duration"]) / sum(self.metrics["duration"])
        }
```

### 2. Model Performance Tests

#### EEGPT Throughput Test
```python
# tests/benchmarks/test_eegpt_performance.py

@pytest.mark.benchmark
class TestEEGPTPerformance:
    """EEGPT model performance benchmarks."""
    
    def test_single_window_inference(self, benchmark):
        """Test single window inference speed."""
        # Setup
        model = EEGPTModel.from_checkpoint(CHECKPOINT_PATH)
        model.eval()
        window = create_synthetic_eeg(duration=8.0, sfreq=256, n_channels=20)
        
        # Warmup
        for _ in range(10):
            _ = model.extract_features(window)
        
        # Benchmark
        result = benchmark(model.extract_features, window)
        
        # Assert performance targets
        assert benchmark.stats["mean"] < 0.1  # <100ms per window
        assert benchmark.stats["stddev"] < 0.02  # Low variance
    
    def test_batch_inference_throughput(self, benchmark):
        """Test batch processing throughput."""
        # Setup
        model = EEGPTModel.from_checkpoint(CHECKPOINT_PATH)
        model.eval()
        batch_sizes = [1, 8, 16, 32, 64]
        
        results = {}
        for batch_size in batch_sizes:
            windows = torch.randn(batch_size, 20, 2048)  # 20 channels, 8s @ 256Hz
            
            # Measure
            def process_batch():
                with torch.no_grad():
                    return model.extract_features_batch(windows)
            
            result = benchmark.pedantic(process_batch, rounds=10, warmup_rounds=5)
            throughput = batch_size / benchmark.stats["mean"]
            results[batch_size] = throughput
            
            # Log results
            print(f"Batch size {batch_size}: {throughput:.1f} windows/sec")
        
        # Assert minimum throughput
        assert results[32] >= 32  # At least 32 windows/sec with batch 32
    
    @pytest.mark.gpu
    def test_gpu_memory_scaling(self):
        """Test GPU memory usage scaling."""
        model = EEGPTModel.from_checkpoint(CHECKPOINT_PATH).cuda()
        model.eval()
        
        memory_usage = {}
        batch_sizes = [1, 4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            # Clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Process batch
            windows = torch.randn(batch_size, 20, 2048).cuda()
            with torch.no_grad():
                _ = model.extract_features_batch(windows)
            
            # Record memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            memory_usage[batch_size] = peak_memory
        
        # Verify linear scaling
        # Memory should scale roughly linearly with batch size
        for i in range(1, len(batch_sizes)):
            ratio = batch_sizes[i] / batch_sizes[0]
            expected = memory_usage[batch_sizes[0]] * ratio
            actual = memory_usage[batch_sizes[i]]
            assert actual < expected * 1.5  # Allow 50% overhead
```

#### AutoReject Performance Test
```python
# tests/benchmarks/test_autoreject_performance.py

@pytest.mark.benchmark
class TestAutoRejectPerformance:
    """AutoReject performance benchmarks."""
    
    def test_channel_rejection_speed(self, benchmark):
        """Test bad channel detection speed."""
        # Create test data with artifacts
        raw = create_raw_with_artifacts(
            duration=300,  # 5 minutes
            n_channels=20,
            n_bad_channels=3
        )
        
        # Setup AutoReject
        ar = AutoRejectWrapper()
        
        # Benchmark
        result = benchmark(ar.detect_bad_channels, raw)
        
        # Performance assertions
        assert benchmark.stats["mean"] < 5.0  # <5 seconds
        assert len(result) == 3  # Correctly identified bad channels
    
    def test_epoch_processing_throughput(self, benchmark):
        """Test epoch-wise processing speed."""
        # Create epoched data
        epochs = create_test_epochs(n_epochs=100, n_channels=20)
        ar = AutoRejectWrapper()
        
        # Benchmark
        result = benchmark(ar.process_epochs, epochs)
        
        # Calculate throughput
        epochs_per_second = 100 / benchmark.stats["mean"]
        
        # Assertions
        assert epochs_per_second > 20  # Process >20 epochs/second
        assert result.percentage_rejected < 20  # <20% rejection rate
```

### 3. Pipeline Integration Tests

#### End-to-End Performance Test
```python
# tests/benchmarks/test_pipeline_performance.py

@pytest.mark.integration
@pytest.mark.slow
class TestPipelinePerformance:
    """Full pipeline performance tests."""
    
    def test_20_minute_eeg_processing(self):
        """Test processing 20-minute EEG within 2 minutes."""
        # Create realistic 20-minute EEG
        eeg_file = create_realistic_eeg_file(
            duration_minutes=20,
            sampling_rate=256,
            n_channels=20
        )
        
        # Setup pipeline
        pipeline = FullPipeline(
            enable_gpu=True,
            batch_size=32,
            num_workers=4
        )
        
        # Measure processing time
        start_time = time.time()
        results = pipeline.process(eeg_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Assertions
        assert processing_time < 120  # <2 minutes
        assert results.quality_report is not None
        assert results.abnormality_score is not None
        assert results.sleep_stages is not None
        
        # Log performance metrics
        print(f"Processing time: {processing_time:.1f}s")
        print(f"Speed ratio: {20*60/processing_time:.1f}x realtime")
    
    def test_concurrent_processing(self):
        """Test handling 50 concurrent analyses."""
        import concurrent.futures
        import threading
        
        # Create test files
        test_files = [
            create_realistic_eeg_file(duration_minutes=5)
            for _ in range(50)
        ]
        
        # Setup pipeline with limited resources
        pipeline = FullPipeline(
            max_workers=10,
            max_memory_gb=40,  # 40GB total for 50 analyses
            enable_gpu_sharing=True
        )
        
        # Process concurrently
        start_time = time.time()
        completed = 0
        failed = 0
        
        def process_file(file_path):
            nonlocal completed, failed
            try:
                result = pipeline.process(file_path)
                completed += 1
                return result
            except Exception as e:
                failed += 1
                raise e
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(process_file, f) 
                for f in test_files
            ]
            
            # Wait for completion with timeout
            done, pending = concurrent.futures.wait(
                futures, 
                timeout=300,  # 5 minute timeout
                return_when=concurrent.futures.ALL_COMPLETED
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assertions
        assert completed == 50  # All completed
        assert failed == 0  # No failures
        assert total_time < 300  # Within 5 minutes
        
        # Performance metrics
        avg_time_per_file = total_time / 50
        print(f"Total time: {total_time:.1f}s")
        print(f"Average per file: {avg_time_per_file:.1f}s")
        print(f"Throughput: {50/total_time*60:.1f} files/minute")
```

### 4. Memory Profiling

#### Memory Usage Tests
```python
# tests/benchmarks/test_memory_usage.py

@pytest.mark.memory
class TestMemoryUsage:
    """Memory usage profiling tests."""
    
    @profile
    def test_model_loading_memory(self):
        """Profile memory usage during model loading."""
        # Baseline memory
        baseline = psutil.Process().memory_info().rss / 1024**3
        
        # Load models
        eegpt = EEGPTModel.from_checkpoint(EEGPT_PATH)
        peak_after_eegpt = psutil.Process().memory_info().rss / 1024**3
        
        autoreject = AutoRejectWrapper()
        peak_after_ar = psutil.Process().memory_info().rss / 1024**3
        
        yasa_model = YASAWrapper()
        peak_after_yasa = psutil.Process().memory_info().rss / 1024**3
        
        # Memory increases
        eegpt_memory = peak_after_eegpt - baseline
        ar_memory = peak_after_ar - peak_after_eegpt
        yasa_memory = peak_after_yasa - peak_after_ar
        
        # Assertions
        assert eegpt_memory < 1.5  # <1.5GB for EEGPT
        assert ar_memory < 0.5  # <500MB for AutoReject
        assert yasa_memory < 1.0  # <1GB for YASA
        
        print(f"EEGPT memory: {eegpt_memory:.2f}GB")
        print(f"AutoReject memory: {ar_memory:.2f}GB")
        print(f"YASA memory: {yasa_memory:.2f}GB")
    
    def test_data_processing_memory_leak(self):
        """Test for memory leaks during repeated processing."""
        pipeline = FullPipeline()
        
        # Process multiple files and check memory
        memory_usage = []
        
        for i in range(10):
            # Create and process file
            test_file = create_realistic_eeg_file(duration_minutes=5)
            _ = pipeline.process(test_file)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Record memory
            current_memory = psutil.Process().memory_info().rss / 1024**3
            memory_usage.append(current_memory)
            
            # Clean up
            test_file.unlink()
        
        # Check for memory leak
        # Memory should stabilize, not continuously increase
        memory_increase = memory_usage[-1] - memory_usage[2]  # Skip first 2 for warmup
        assert memory_increase < 0.5  # <500MB increase over 8 iterations
        
        # Calculate leak rate
        leak_rate = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        assert leak_rate < 0.05  # <50MB per iteration
```

### 5. API Performance Tests

#### FastAPI Endpoint Benchmarks
```python
# tests/benchmarks/test_api_performance.py
import asyncio
import aiohttp
from locust import HttpUser, task, between

@pytest.mark.asyncio
class TestAPIPerformance:
    """API endpoint performance tests."""
    
    async def test_analyze_endpoint_response_time(self):
        """Test /analyze endpoint meets <100ms target."""
        app = create_test_app()
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Prepare small test file
            small_eeg = create_test_edf_bytes(duration=10)
            
            response_times = []
            
            # Make 100 requests
            for _ in range(100):
                start = time.perf_counter()
                
                response = await client.post(
                    "/api/v1/eeg/analyze",
                    files={"file": ("test.edf", small_eeg, "application/octet-stream")},
                    data={"analysis_type": "quick"}
                )
                
                end = time.perf_counter()
                response_times.append((end - start) * 1000)  # ms
                
                assert response.status_code == 202
            
            # Calculate statistics
            avg_time = np.mean(response_times)
            p95_time = np.percentile(response_times, 95)
            p99_time = np.percentile(response_times, 99)
            
            # Assertions
            assert avg_time < 100  # Average <100ms
            assert p95_time < 150  # 95th percentile <150ms
            assert p99_time < 200  # 99th percentile <200ms
            
            print(f"Average: {avg_time:.1f}ms")
            print(f"P95: {p95_time:.1f}ms")
            print(f"P99: {p99_time:.1f}ms")
    
    async def test_concurrent_api_requests(self):
        """Test API handling concurrent requests."""
        app = create_test_app()
        
        async def make_request(client, file_data):
            """Make single analysis request."""
            start = time.time()
            response = await client.post(
                "/api/v1/eeg/analyze",
                files={"file": ("test.edf", file_data, "application/octet-stream")}
            )
            duration = time.time() - start
            return response.status_code, duration
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create test data
            test_files = [create_test_edf_bytes(duration=30) for _ in range(50)]
            
            # Make concurrent requests
            start_time = time.time()
            tasks = [make_request(client, f) for f in test_files]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Analyze results
            status_codes = [r[0] for r in results]
            response_times = [r[1] for r in results]
            
            # Assertions
            assert all(code == 202 for code in status_codes)
            assert total_time < 10  # All complete within 10 seconds
            assert max(response_times) < 5  # No request takes >5 seconds
            
            print(f"Concurrent requests: {len(results)}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Requests/second: {len(results)/total_time:.1f}")

# Locust load testing configuration
class EEGAnalysisUser(HttpUser):
    """Locust user for load testing."""
    wait_time = between(1, 3)
    
    @task
    def analyze_eeg(self):
        """Submit EEG for analysis."""
        with open("test_data/sample.edf", "rb") as f:
            self.client.post(
                "/api/v1/eeg/analyze",
                files={"file": f},
                data={"analysis_type": "full"}
            )
    
    @task
    def check_status(self):
        """Check job status."""
        job_id = "test-job-123"  # Would be dynamic in real test
        self.client.get(f"/api/v1/jobs/{job_id}")
```

### 6. Optimization Profiling

#### CPU/GPU Profiling
```python
# tests/benchmarks/test_profiling.py
import cProfile
import pstats
from torch.profiler import profile, ProfilerActivity

class TestProfiling:
    """Detailed profiling tests."""
    
    def test_cpu_profiling(self):
        """Profile CPU usage of critical paths."""
        profiler = cProfile.Profile()
        
        # Profile EEGPT feature extraction
        eeg_data = create_synthetic_eeg(duration=60, sfreq=256)
        model = EEGPTModel.from_checkpoint(CHECKPOINT_PATH)
        
        profiler.enable()
        
        # Process 1 minute of data
        windows = sliding_window(eeg_data, window_size=8, stride=4)
        features = []
        for window in windows:
            feat = model.extract_features(window)
            features.append(feat)
        
        profiler.disable()
        
        # Analyze profile
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Save detailed report
        stats.dump_stats('eegpt_cpu_profile.prof')
        
        # Extract key metrics
        total_time = stats.total_tt
        assert total_time < 30  # Process 1 minute in <30 seconds
    
    @pytest.mark.gpu
    def test_gpu_profiling(self):
        """Profile GPU usage with PyTorch profiler."""
        model = EEGPTModel.from_checkpoint(CHECKPOINT_PATH).cuda()
        model.eval()
        
        # Prepare data
        batch = torch.randn(32, 20, 2048).cuda()
        
        # Profile with PyTorch
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(100):
                    _ = model.extract_features_batch(batch)
        
        # Analyze GPU utilization
        key_metrics = prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=10
        )
        
        # Save trace
        prof.export_chrome_trace("eegpt_gpu_trace.json")
        
        # Extract GPU utilization
        cuda_time = sum(item.cuda_time_total for item in prof.key_averages())
        cpu_time = sum(item.cpu_time_total for item in prof.key_averages())
        
        gpu_utilization = cuda_time / (cuda_time + cpu_time)
        assert gpu_utilization > 0.8  # >80% GPU utilization
```

### 7. Cache Performance

#### Cache Hit Rate Testing
```python
# tests/benchmarks/test_cache_performance.py

class TestCachePerformance:
    """Cache performance benchmarks."""
    
    def test_feature_cache_hit_rate(self):
        """Test feature extraction cache performance."""
        cache = FeatureCache(max_size_gb=2)
        pipeline = FullPipeline(feature_cache=cache)
        
        # Process same file multiple times
        test_file = create_realistic_eeg_file(duration_minutes=10)
        
        # First run - cache miss
        start1 = time.time()
        result1 = pipeline.process(test_file)
        time1 = time.time() - start1
        
        # Second run - cache hit
        start2 = time.time()
        result2 = pipeline.process(test_file)
        time2 = time.time() - start2
        
        # Cache should significantly speed up processing
        speedup = time1 / time2
        assert speedup > 5  # At least 5x speedup
        assert cache.hit_rate > 0.9  # >90% hit rate
        
        print(f"First run: {time1:.1f}s")
        print(f"Cached run: {time2:.1f}s")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Hit rate: {cache.hit_rate:.2%}")
```

## Performance Monitoring in Production

### 1. Metrics Collection
```python
# src/brain_go_brrr/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
processing_time = Histogram(
    'eeg_processing_duration_seconds',
    'Time spent processing EEG files',
    ['analysis_type', 'status']
)

active_analyses = Gauge(
    'eeg_active_analyses',
    'Number of currently active analyses'
)

model_inference_time = Histogram(
    'model_inference_duration_seconds',
    'Time spent in model inference',
    ['model_name']
)

memory_usage = Gauge(
    'process_memory_usage_bytes',
    'Current memory usage in bytes'
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

cache_hits = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

# Usage example
def monitored_analysis(file_path: Path, analysis_type: str):
    """Analysis with performance monitoring."""
    active_analyses.inc()
    
    with processing_time.labels(
        analysis_type=analysis_type,
        status='success'
    ).time():
        try:
            result = process_eeg_file(file_path)
            return result
        except Exception as e:
            processing_time.labels(
                analysis_type=analysis_type,
                status='failure'
            ).observe(time.time() - start_time)
            raise
        finally:
            active_analyses.dec()
```

### 2. Performance Dashboard
```yaml
# monitoring/grafana/dashboards/performance.json
{
  "dashboard": {
    "title": "Brain-Go-Brrr Performance",
    "panels": [
      {
        "title": "Processing Time (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(eeg_processing_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Active Analyses",
        "targets": [{
          "expr": "eeg_active_analyses"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "avg(gpu_utilization_percent)"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))"
        }]
      },
      {
        "title": "Memory Usage",
        "targets": [{
          "expr": "process_memory_usage_bytes / 1024 / 1024 / 1024"
        }]
      }
    ]
  }
}
```

## Performance Optimization Strategies

### 1. Batching Optimization
```python
# src/brain_go_brrr/optimization/batching.py
class AdaptiveBatcher:
    """Adaptive batching for optimal throughput."""
    
    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        target_latency_ms: float = 100
    ):
        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.target_latency = target_latency_ms
        self.current_batch_size = min_batch_size
        
    def adjust_batch_size(self, last_latency_ms: float):
        """Adjust batch size based on latency."""
        if last_latency_ms > self.target_latency * 1.2:
            # Too slow, reduce batch
            self.current_batch_size = max(
                self.min_batch,
                int(self.current_batch_size * 0.8)
            )
        elif last_latency_ms < self.target_latency * 0.8:
            # Too fast, increase batch
            self.current_batch_size = min(
                self.max_batch,
                int(self.current_batch_size * 1.2)
            )
```

### 2. Memory Optimization
```python
# src/brain_go_brrr/optimization/memory.py
class MemoryOptimizer:
    """Memory optimization strategies."""
    
    @staticmethod
    def process_in_chunks(
        data: np.ndarray,
        process_func: Callable,
        max_memory_gb: float = 2.0
    ):
        """Process large data in memory-efficient chunks."""
        # Calculate chunk size based on memory limit
        bytes_per_sample = data.itemsize * np.prod(data.shape[1:])
        max_samples = int(max_memory_gb * 1024**3 / bytes_per_sample)
        
        results = []
        for i in range(0, len(data), max_samples):
            chunk = data[i:i + max_samples]
            result = process_func(chunk)
            results.append(result)
            
            # Force garbage collection
            del chunk
            gc.collect()
        
        return np.concatenate(results)
```

## Continuous Performance Testing

### GitHub Actions Workflow
```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  push:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance:
    runs-on: [self-hosted, gpu]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev
    
    - name: Run performance benchmarks
      run: |
        uv run pytest tests/benchmarks/ \
          --benchmark-only \
          --benchmark-json=benchmark_results.json \
          --benchmark-compare=main
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results.json
    
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const results = require('./benchmark_results.json');
          const comment = generatePerformanceReport(results);
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

## Performance Troubleshooting Guide

### Common Issues and Solutions

1. **Slow Model Loading**
   - Use model caching
   - Implement lazy loading
   - Pre-warm models on startup

2. **Memory Leaks**
   - Profile with memory_profiler
   - Check for circular references
   - Ensure proper cleanup in __del__

3. **GPU Underutilization**
   - Increase batch size
   - Use mixed precision training
   - Enable CUDA graphs

4. **Cache Misses**
   - Increase cache size
   - Implement better cache keys
   - Use distributed caching

5. **API Timeouts**
   - Implement request queuing
   - Add circuit breakers
   - Scale horizontally

## Conclusion

This comprehensive performance benchmarking guide ensures Brain-Go-Brrr meets its ambitious performance targets while maintaining accuracy and reliability. Regular benchmarking and monitoring enable continuous optimization and early detection of performance regressions.
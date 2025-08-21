# EEGPT Performance Benchmarks

This document describes the comprehensive performance benchmark suite for EEGPT inference, designed to ensure the system meets production performance requirements.

## Performance Targets

The benchmark suite validates against these performance targets:

| Metric | Target | Description |
|--------|--------|-------------|
| Single 4-second window | < 50ms | Individual window inference time |
| 20-minute recording | < 2 minutes | Full recording processing time |
| Memory usage | < 2GB | Memory consumption for typical recording |
| GPU speedup | > 2x | Minimum GPU acceleration over CPU |
| Processing throughput | > 10x real-time | Speed vs recording duration |

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
make benchmark

# Run CPU-only benchmarks (skip GPU tests)
python scripts/run_benchmarks.py --cpu-only

# Run quick benchmarks only (skip slow tests)
python scripts/run_benchmarks.py --quick

# Run with verbose output
python scripts/run_benchmarks.py --verbose
```

### Manual pytest Commands

```bash
# Run all benchmark tests
uv run pytest tests/benchmarks/ --benchmark-only -v

# Run specific benchmark categories
uv run pytest tests/benchmarks/ -m "benchmark and not slow" --benchmark-only

# Run with performance comparison
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare

# Run with JSON output for analysis
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
```

## Benchmark Categories

### 1. Single Window Benchmarks (`TestSingleWindowBenchmarks`)

Tests inference speed for individual 4-second EEG windows:

- **CPU inference speed**: Validates < 50ms target on CPU
- **GPU inference speed**: Validates < 25ms target on GPU
- **Different input sizes**: Tests various channel counts and window lengths
- **Shape validation**: Ensures correct output dimensions

### 2. Batch Processing Benchmarks (`TestBatchProcessingBenchmarks`)

Tests efficiency of batch processing:

- **Batch efficiency**: Compares batch vs individual processing
- **Batch size scaling**: Tests performance across batch sizes (1, 4, 8, 16, 32, 64)
- **Per-item performance**: Validates per-window time in batch processing

### 3. Full Recording Benchmarks (`TestFullRecordingBenchmarks`)

Tests end-to-end processing performance:

- **20-minute recording**: Validates < 2 minute processing target
- **Different durations**: Tests 1, 5, 10, 20 minute recordings
- **Throughput calculation**: Measures processing speed vs real-time
- **Batch optimization**: Uses configurable batch sizes for efficiency

### 4. Memory Benchmarks (`TestMemoryBenchmarks`)

Tests memory usage during inference:

- **Single window memory**: Validates minimal memory for individual windows
- **Full recording memory**: Validates < 2GB target for 20-minute recordings
- **GPU memory usage**: Tests CUDA memory consumption
- **Memory cleanup**: Verifies proper cleanup after processing

### 5. Performance Comparison (`TestPerformanceComparison`)

Compares different execution contexts:

- **CPU vs GPU**: Measures speedup and validates equivalent results
- **Batch vs individual**: Compares processing strategies
- **Device transfer overhead**: Measures GPU transfer costs

## Benchmark Configuration

### Performance Targets

Defined in `tests/benchmarks/benchmark_config.py`:

```python
PERFORMANCE_TARGETS = {
    "single_window_ms": 50,          # Single window target
    "twenty_min_recording_s": 120,   # Full recording target
    "memory_usage_gb": 2.0,          # Memory usage target
    "gpu_speedup_min": 2.0,          # Minimum GPU speedup
    "throughput_ratio_min": 10.0,    # Processing throughput target
}
```

### Test Data Configurations

```python
TEST_DATA_CONFIGS = {
    "single_window": {"channels": 19, "samples": 1024},
    "max_channels_window": {"channels": 58, "samples": 1024},
    "twenty_min_recording": {"channels": 19, "samples": 307200},
    "batch_sizes": [1, 4, 8, 16, 32, 64],
}
```

## Benchmark Output

### Console Output

Benchmarks display real-time performance metrics:

```
tests/benchmarks/test_eegpt_performance.py::TestSingleWindowBenchmarks::test_single_window_cpu_inference_speed
Mean: 32.45ms, Min: 31.2ms, Max: 35.1ms ✅ PASS (< 50ms target)

tests/benchmarks/test_eegpt_performance.py::TestBatchProcessingBenchmarks::test_batch_processing_efficiency
Batch processing: 8.2ms per window ✅ PASS (efficient)
```

### JSON Results

Detailed results saved to `benchmark_results/benchmark_results.json`:

```json
{
  "machine_info": {
    "python_version": "3.11.0",
    "pytorch_version": "2.2.0",
    "cuda_available": true,
    "cuda_device_name": "NVIDIA GeForce RTX 4090"
  },
  "benchmarks": [
    {
      "name": "test_single_window_cpu_inference_speed",
      "mean": 0.03245,
      "min": 0.0312,
      "max": 0.0351,
      "stddev": 0.0012,
      "rounds": 10
    }
  ]
}
```

### Report Generation

Generate formatted reports:

```python
from tests.benchmarks.benchmark_config import BenchmarkReporter

reporter = BenchmarkReporter()
# ... add results ...
markdown_report = reporter.generate_report("markdown")
```

## Dependencies

The benchmark suite requires these additional dependencies:

```toml
# Added to dev-dependencies in pyproject.toml
"pytest-benchmark>=4.0.0",
"memory-profiler>=0.61.0",
"psutil>=5.9.0",
```

Install with:

```bash
uv sync  # Installs all dev dependencies including benchmark tools
```

## Continuous Integration

### GitHub Actions Integration

Add to CI workflow:

```yaml
- name: Run Performance Benchmarks
  run: |
    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark_results.json

- name: Upload Benchmark Results
  uses: actions/upload-artifact@v3
  with:
    name: benchmark-results
    path: benchmark_results.json
```

### Performance Regression Detection

Monitor performance over time:

```bash
# Save baseline results
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline
```

## Troubleshooting

### Common Issues

1. **GPU tests skipped**: Ensure CUDA is available and PyTorch has GPU support
2. **Memory tests failing**: May need to increase memory limits or use smaller test data
3. **Slow performance**: Check if running in debug mode or with coverage enabled

### Debug Mode

Run with additional debugging:

```bash
# Enable memory profiling
uv run python -m memory_profiler scripts/run_benchmarks.py

# Run single test with profiling
uv run pytest tests/benchmarks/test_eegpt_performance.py::TestSingleWindowBenchmarks::test_single_window_cpu_inference_speed --benchmark-only -v -s
```

### Performance Optimization

If benchmarks fail to meet targets:

1. **Check model loading**: Ensure EEGPT checkpoint is optimized
2. **Verify batch sizes**: Adjust batch processing parameters
3. **GPU utilization**: Monitor GPU memory and compute usage
4. **Memory management**: Check for memory leaks or excessive allocation

## Integration with Make

The benchmark suite integrates with the project's Makefile:

```bash
make benchmark     # Run all benchmarks
make test          # Run tests (excludes benchmarks by default)
make check-all     # Run all quality checks including benchmarks
```

## Future Enhancements

Potential improvements to the benchmark suite:

- **Distributed benchmarking**: Test performance across multiple GPUs
- **Different model sizes**: Benchmark EEGPT variants (large vs xlarge)
- **Real-world data**: Use actual EEG recordings instead of synthetic data
- **Optimization benchmarks**: Test different optimization techniques
- **Deployment benchmarks**: Test containerized and serverless deployments

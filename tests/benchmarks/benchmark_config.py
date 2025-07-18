"""Configuration and utilities for EEGPT performance benchmarks."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest


# Performance targets from requirements
PERFORMANCE_TARGETS = {
    "single_window_ms": 50,          # Single 4-second window: < 50ms
    "twenty_min_recording_s": 120,   # 20-minute recording: < 2 minutes
    "memory_usage_gb": 2.0,          # Memory usage: < 2GB
    "gpu_speedup_min": 2.0,          # GPU should be at least 2x faster than CPU
    "throughput_ratio_min": 10.0,    # Should process 10x faster than real-time
}

# Default benchmark configuration
DEFAULT_BENCHMARK_CONFIG = {
    "min_rounds": 5,
    "max_time": 10.0,        # Maximum time per benchmark in seconds
    "warmup": True,
    "warmup_iterations": 1,
    "disable_gc": True,      # Disable garbage collection during timing
    "sort": "mean",
}

# Test data configurations
TEST_DATA_CONFIGS = {
    "single_window": {
        "channels": 19,
        "samples": 1024,        # 4 seconds at 256 Hz
        "dtype": "float32",
    },
    "max_channels_window": {
        "channels": 58,         # Maximum supported channels
        "samples": 1024,
        "dtype": "float32",
    },
    "twenty_min_recording": {
        "channels": 19,
        "samples": 307200,      # 20 minutes at 256 Hz
        "dtype": "float32",
    },
    "batch_sizes": [1, 4, 8, 16, 32, 64],
    "recording_durations_min": [1, 5, 10, 20],
}


class BenchmarkReporter:
    """Generate benchmark reports and validate performance targets."""
    
    def __init__(self, results_dir: Path | None = None):
        """Initialize benchmark reporter.
        
        Args:
            results_dir: Directory to save benchmark results
        """
        self.results_dir = results_dir or Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {}
        
    def add_result(self, test_name: str, benchmark_result: Any, metadata: Dict[str, Any] | None = None) -> None:
        """Add benchmark result for reporting.
        
        Args:
            test_name: Name of the benchmark test
            benchmark_result: Pytest benchmark result object
            metadata: Additional metadata for the test
        """
        result_data = {
            "timestamp": time.time(),
            "mean_time_s": benchmark_result.stats.mean,
            "mean_time_ms": benchmark_result.stats.mean * 1000,
            "min_time_s": benchmark_result.stats.min,
            "max_time_s": benchmark_result.stats.max,
            "std_time_s": benchmark_result.stats.stddev,
            "rounds": benchmark_result.stats.rounds,
            "metadata": metadata or {},
        }
        self.results[test_name] = result_data
        
    def validate_performance_target(self, test_name: str, target_key: str, value: float) -> bool:
        """Validate that a performance metric meets the target.
        
        Args:
            test_name: Name of the test
            target_key: Key in PERFORMANCE_TARGETS
            value: Measured value
            
        Returns:
            True if target is met, False otherwise
        """
        target = PERFORMANCE_TARGETS.get(target_key)
        if target is None:
            return True
            
        passed = value < target
        
        # Store validation result
        if test_name not in self.results:
            self.results[test_name] = {}
        
        self.results[test_name]["validation"] = {
            "target_key": target_key,
            "target_value": target,
            "measured_value": value,
            "passed": passed,
        }
        
        return passed
        
    def generate_report(self, format: str = "markdown") -> str:
        """Generate benchmark report.
        
        Args:
            format: Report format ("markdown", "json", "text")
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return json.dumps(self.results, indent=2)
        elif format == "markdown":
            return self._generate_markdown_report()
        else:  # text
            return self._generate_text_report()
            
    def _generate_markdown_report(self) -> str:
        """Generate markdown benchmark report."""
        lines = [
            "# EEGPT Performance Benchmark Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Targets",
            "",
        ]
        
        # Add targets table
        lines.extend([
            "| Metric | Target | Description |",
            "|--------|--------|-------------|",
        ])
        
        target_descriptions = {
            "single_window_ms": "Single 4-second window inference time",
            "twenty_min_recording_s": "20-minute recording processing time", 
            "memory_usage_gb": "Memory usage for typical recording",
            "gpu_speedup_min": "Minimum GPU speedup over CPU",
            "throughput_ratio_min": "Processing speed vs real-time ratio",
        }
        
        for key, target in PERFORMANCE_TARGETS.items():
            desc = target_descriptions.get(key, key)
            unit = "ms" if "ms" in key else "s" if "_s" in key else "GB" if "gb" in key else "x"
            lines.append(f"| {desc} | < {target} {unit} | Performance requirement |")
            
        lines.extend(["", "## Benchmark Results", ""])
        
        # Add results table
        if self.results:
            lines.extend([
                "| Test | Mean Time | Target | Status |",
                "|------|-----------|--------|--------|",
            ])
            
            for test_name, result in self.results.items():
                mean_time = result.get("mean_time_ms", 0)
                validation = result.get("validation", {})
                
                if validation:
                    target = validation["target_value"]
                    passed = validation["passed"]
                    status = "✅ PASS" if passed else "❌ FAIL"
                    target_str = f"{target} ms"
                else:
                    status = "⚪ N/A"
                    target_str = "N/A"
                    
                lines.append(f"| {test_name} | {mean_time:.1f} ms | {target_str} | {status} |")
        
        return "\n".join(lines)
        
    def _generate_text_report(self) -> str:
        """Generate plain text benchmark report."""
        lines = [
            "EEGPT Performance Benchmark Report",
            "=" * 40,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Performance Targets:",
        ]
        
        for key, target in PERFORMANCE_TARGETS.items():
            lines.append(f"  {key}: {target}")
            
        lines.extend(["", "Results:"])
        
        for test_name, result in self.results.items():
            mean_time = result.get("mean_time_ms", 0)
            lines.append(f"  {test_name}: {mean_time:.1f} ms")
            
            validation = result.get("validation")
            if validation:
                status = "PASS" if validation["passed"] else "FAIL"
                lines.append(f"    Target: {validation['target_value']} - {status}")
                
        return "\n".join(lines)
        
    def save_report(self, filename: str | None = None, format: str = "markdown") -> Path:
        """Save benchmark report to file.
        
        Args:
            filename: Output filename (auto-generated if None)
            format: Report format
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = "md" if format == "markdown" else "json" if format == "json" else "txt"
            filename = f"benchmark_report_{timestamp}.{ext}"
            
        report_path = self.results_dir / filename
        report_content = self.generate_report(format)
        
        report_path.write_text(report_content)
        return report_path


def pytest_benchmark_update_machine_info(config, machine_info):
    """Update machine info for benchmark reports."""
    import torch
    import platform
    
    machine_info.update({
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    })
    
    if torch.cuda.is_available():
        machine_info["cuda_device_name"] = torch.cuda.get_device_name(0)


# Custom benchmark fixtures
@pytest.fixture(scope="session")
def benchmark_reporter():
    """Provide benchmark reporter for collecting results."""
    return BenchmarkReporter()


def benchmark_eegpt_inference(benchmark, model, data, expected_shape=None, target_ms=None):
    """Standard benchmark wrapper for EEGPT inference.
    
    Args:
        benchmark: Pytest benchmark fixture
        model: EEGPT model instance
        data: Input EEG data
        expected_shape: Expected output shape for validation
        target_ms: Performance target in milliseconds
        
    Returns:
        Benchmark result and extracted features
    """
    def extract_features():
        return model.extract_features(data)
    
    # Run benchmark
    features = benchmark(extract_features)
    
    # Validate output shape
    if expected_shape:
        assert features.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {features.shape}"
        )
    
    # Validate performance target
    if target_ms:
        actual_ms = benchmark.stats.mean * 1000
        assert actual_ms < target_ms, (
            f"Inference took {actual_ms:.1f}ms, target was {target_ms}ms"
        )
    
    return features


def benchmark_eegpt_batch(benchmark, model, batch_data, expected_shape=None, target_ms_per_item=None):
    """Standard benchmark wrapper for EEGPT batch processing.
    
    Args:
        benchmark: Pytest benchmark fixture
        model: EEGPT model instance
        batch_data: Batch of EEG data
        expected_shape: Expected output shape for validation
        target_ms_per_item: Performance target per item in milliseconds
        
    Returns:
        Benchmark result and extracted features
    """
    def extract_features_batch():
        return model.extract_features_batch(batch_data)
    
    # Run benchmark
    features = benchmark(extract_features_batch)
    
    # Validate output shape
    if expected_shape:
        assert features.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {features.shape}"
        )
    
    # Validate performance target
    if target_ms_per_item:
        actual_ms_per_item = (benchmark.stats.mean / len(batch_data)) * 1000
        assert actual_ms_per_item < target_ms_per_item, (
            f"Batch processing took {actual_ms_per_item:.1f}ms per item, "
            f"target was {target_ms_per_item}ms"
        )
    
    return features
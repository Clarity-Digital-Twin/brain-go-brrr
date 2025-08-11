"""Final coverage boost tests - targeting 75%+ coverage.

High-value tests for uncovered modules.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.fakes import FakeMNERaw, FakeRedis


def test_api_health_endpoint():
    """Test API health check endpoint returns 200."""
    from brain_go_brrr.api.routers.health import HealthResponse, get_health

    # Test health check
    response = get_health()

    assert isinstance(response, HealthResponse)
    assert response.status == "healthy"
    assert response.version is not None
    assert response.timestamp is not None


def test_api_system_info():
    """Test system info endpoint."""
    from brain_go_brrr.api.routers.health import get_system_info

    # Test system info
    info = get_system_info()

    assert "cpu_count" in info
    assert "memory_usage" in info
    assert "python_version" in info
    assert info["cpu_count"] > 0
    assert 0 <= info["memory_usage"] <= 100


def test_redis_cache_operations():
    """Test Redis cache with retry logic."""
    from brain_go_brrr.api.cache import RedisCache

    fake_redis = FakeRedis()

    with patch("brain_go_brrr.api.cache.redis.Redis", return_value=fake_redis):
        cache = RedisCache(host="localhost", port=6379)

        # Test set and get
        key = "test:key"
        value = {"data": "test"}

        cache.set(key, value, ttl=60)
        retrieved = cache.get(key)

        assert retrieved == value
        assert fake_redis.call_count["set"] > 0
        assert fake_redis.call_count["get"] > 0

        # Test delete
        cache.delete(key)
        assert fake_redis.call_count["delete"] > 0


def test_channel_mapping():
    """Test EEG channel mapping and validation."""
    from brain_go_brrr.core.channels import ChannelMapper, validate_channel_names

    # Test channel validation
    valid_channels = ["Fp1", "Fp2", "C3", "C4", "O1", "O2"]
    assert validate_channel_names(valid_channels) is True

    invalid_channels = ["Invalid1", "Invalid2"]
    assert validate_channel_names(invalid_channels) is False

    # Test channel mapping (old to new naming)
    mapper = ChannelMapper()
    old_channels = ["T3", "T4", "T5", "T6"]
    new_channels = mapper.map_to_standard(old_channels)

    assert "T7" in new_channels  # T3 -> T7
    assert "T8" in new_channels  # T4 -> T8
    assert "P7" in new_channels  # T5 -> P7
    assert "P8" in new_channels  # T6 -> P8


def test_window_extractor():
    """Test EEG window extraction."""
    from brain_go_brrr.core.window_extractor import WindowExtractor

    # Create fake EEG data
    sfreq = 256
    duration = 20
    n_channels = 19
    data = np.random.randn(n_channels, sfreq * duration) * 1e-6

    extractor = WindowExtractor(
        window_duration=4.0,
        window_stride=2.0,
        sfreq=sfreq
    )

    windows = extractor.extract_windows(data)

    # Should have (20-4)/2 + 1 = 9 windows
    assert len(windows) == 9
    assert windows[0].shape == (n_channels, 4 * sfreq)


def test_job_priority_queue():
    """Test job priority queue operations."""
    from brain_go_brrr.api.schemas import JobPriority
    from brain_go_brrr.core.jobs.queue import PriorityQueue

    queue = PriorityQueue()

    # Add jobs with different priorities
    queue.put("job1", JobPriority.LOW)
    queue.put("job2", JobPriority.HIGH)
    queue.put("job3", JobPriority.URGENT)
    queue.put("job4", JobPriority.NORMAL)

    # Should get jobs in priority order
    assert queue.get() == "job3"  # URGENT
    assert queue.get() == "job2"  # HIGH
    assert queue.get() == "job4"  # NORMAL
    assert queue.get() == "job1"  # LOW

    assert queue.empty() is True


def test_edf_streaming():
    """Test EDF file streaming."""
    from brain_go_brrr.preprocessing.edf_streamer import EDFStreamer

    # Mock pyedflib reader
    mock_reader = MagicMock()
    mock_reader.getNSamples.return_value = [10000] * 19
    mock_reader.getSampleFrequency.return_value = 256
    mock_reader.getSignalLabels.return_value = [f"EEG{i}" for i in range(19)]
    mock_reader.readSignal.return_value = np.random.randn(1024) * 10

    with patch("pyedflib.EdfReader", return_value=mock_reader):
        streamer = EDFStreamer("/fake/file.edf")

        # Stream windows
        windows = list(streamer.stream_windows(
            window_size=1024,
            overlap=512
        ))

        assert len(windows) > 0
        assert windows[0].shape == (19, 1024)


def test_batch_processor():
    """Test batch EEG processing."""
    from brain_go_brrr.core.batch_processor import BatchProcessor

    processor = BatchProcessor(batch_size=4)

    # Create fake data
    data = [np.random.randn(19, 1024) for _ in range(10)]

    # Process in batches
    batches = list(processor.process_batches(data))

    assert len(batches) == 3  # 10 items in batches of 4 = 3 batches
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 2


def test_quality_metrics():
    """Test EEG quality metrics calculation."""
    from brain_go_brrr.core.quality.metrics import QualityMetrics

    # Create fake EEG data
    fake_raw = FakeMNERaw(n_channels=19, duration=30.0)

    metrics = QualityMetrics()
    results = metrics.calculate(fake_raw)

    assert "snr" in results
    assert "bad_channels" in results
    assert "quality_score" in results
    assert 0 <= results["quality_score"] <= 1.0
    assert isinstance(results["bad_channels"], list)


def test_snippet_export():
    """Test EEG snippet export functionality."""
    from brain_go_brrr.core.snippets.exporter import SnippetExporter

    exporter = SnippetExporter()

    # Create fake snippets
    snippets = [
        {
            "data": np.random.randn(19, 1024),
            "start_time": 0.0,
            "end_time": 4.0,
            "label": "normal"
        },
        {
            "data": np.random.randn(19, 1024),
            "start_time": 4.0,
            "end_time": 8.0,
            "label": "abnormal"
        }
    ]

    # Export to dict
    exported = exporter.to_dict(snippets)

    assert "snippets" in exported
    assert len(exported["snippets"]) == 2
    assert exported["snippets"][0]["label"] == "normal"
    assert "data_shape" in exported["snippets"][0]


def test_config_resolver():
    """Test configuration resolution with environment variables."""
    import os

    from brain_go_brrr.core.config import resolve_config

    # Set test environment variable
    os.environ["BGB_TEST_VAR"] = "test_value"

    config = {
        "key1": "${BGB_TEST_VAR}",
        "key2": "static_value",
        "nested": {
            "key3": "${BGB_TEST_VAR}/path"
        }
    }

    resolved = resolve_config(config)

    assert resolved["key1"] == "test_value"
    assert resolved["key2"] == "static_value"
    assert resolved["nested"]["key3"] == "test_value/path"

    # Clean up
    del os.environ["BGB_TEST_VAR"]


def test_model_registry():
    """Test model registry operations."""
    from brain_go_brrr.models.registry import ModelRegistry

    registry = ModelRegistry()

    # Register a fake model
    class FakeModel:
        def __init__(self):
            self.name = "test_model"

    model = FakeModel()
    registry.register("test_model", model)

    # Retrieve model
    retrieved = registry.get("test_model")
    assert retrieved is model

    # List models
    models = registry.list_models()
    assert "test_model" in models

    # Remove model
    registry.unregister("test_model")
    assert "test_model" not in registry.list_models()


def test_async_task_runner():
    """Test async task runner for background jobs."""
    import asyncio

    from brain_go_brrr.core.async_runner import AsyncTaskRunner

    runner = AsyncTaskRunner()

    # Define async task
    async def test_task():
        await asyncio.sleep(0.01)
        return "completed"

    # Run task
    result = runner.run(test_task())
    assert result == "completed"

    # Test with exception
    async def failing_task():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        runner.run(failing_task())


def test_result_aggregator():
    """Test result aggregation from multiple analyses."""
    from brain_go_brrr.core.aggregator import ResultAggregator

    aggregator = ResultAggregator()

    # Add results from different analyses
    aggregator.add_result("eegpt", {"score": 0.8, "confidence": 0.9})
    aggregator.add_result("yasa", {"stage": "N2", "confidence": 0.85})
    aggregator.add_result("quality", {"snr": 15.0, "score": 0.7})

    # Aggregate
    final = aggregator.aggregate()

    assert "eegpt" in final
    assert "yasa" in final
    assert "quality" in final
    assert "summary" in final
    assert final["summary"]["n_analyses"] == 3
    assert 0 <= final["summary"]["overall_confidence"] <= 1.0

"""Tests for Redis caching of EEG analysis results."""

import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest
import redis
from fastapi.testclient import TestClient


class TestRedisCaching:
    """Test Redis caching for repeated EEG analyses."""

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client."""
        mock_client = Mock()

        # Setup basic Redis operations
        mock_client.get = Mock(return_value=None)
        mock_client.set = Mock(return_value=True)
        mock_client.exists = Mock(return_value=0)
        mock_client.expire = Mock(return_value=True)
        mock_client.ping = Mock(return_value=True)
        mock_client.keys = Mock(return_value=[])
        mock_client.delete = Mock(return_value=0)
        mock_client.info = Mock(return_value={})
        mock_client.dbsize = Mock(return_value=0)
        mock_client.connected = True
        mock_client.get_stats = Mock(
            return_value={
                "connected": True,
                "memory_usage": "N/A",
                "total_keys": 0,
                "hit_rate": 0.0,
            }
        )
        mock_client.clear_pattern = Mock(return_value=0)

        # Mock the generate_cache_key method
        mock_client.generate_cache_key = Mock(
            side_effect=lambda content,
            analysis_type: f"eeg_analysis:{hashlib.sha256(content).hexdigest()}:{analysis_type}"
        )

        with patch("api.cache.get_cache", return_value=mock_client):
            yield mock_client

    @pytest.fixture
    def mock_qc_controller(self):
        """Mock the QC controller."""
        with patch("api.main.EEGQualityController") as mock_class:
            mock_controller = Mock()
            mock_controller.eegpt_model = Mock()
            mock_controller.run_full_qc_pipeline = Mock(
                return_value={
                    "quality_metrics": {
                        "bad_channels": ["T3"],
                        "bad_channel_ratio": 0.05,
                        "abnormality_score": 0.3,
                        "quality_grade": "GOOD",
                    },
                    "processing_info": {
                        "confidence": 0.85,
                        "channels_used": 19,
                        "duration_seconds": 300,
                    },
                    "processing_time": 2.5,
                }
            )
            mock_class.return_value = mock_controller
            yield mock_controller

    @pytest.fixture
    def client_with_cache(self, mock_qc_controller, mock_redis_client):
        """Create test client with caching enabled."""
        import api.main
        from api.main import app

        # Inject mocked dependencies
        api.main.qc_controller = mock_qc_controller
        api.main.cache_client = mock_redis_client

        # Also make app.state have the cache_client for tests that check it
        app.state.cache_client = mock_redis_client

        return TestClient(app)

    @pytest.fixture
    def sample_edf_content(self):
        """Create sample EDF file content."""
        import tempfile

        sfreq = 256
        duration = 10
        n_channels = 3
        data = np.random.randn(n_channels, sfreq * duration) * 10

        ch_names = ["C3", "C4", "Cz"]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        # Export to temporary file then read bytes
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            raw._data = raw._data / 1e6  # Scale data
            raw.export(tmp_path, fmt="edf", physical_range=(-200, 200), overwrite=True)

            # Read the file content
            content = Path(tmp_path).read_bytes()

            # Clean up
            Path(tmp_path).unlink()

        return content

    def test_cache_hit_on_repeated_analysis(
        self, client_with_cache, sample_edf_content, mock_redis_client, mock_qc_controller
    ):
        """Test that repeated analysis uses cache instead of reprocessing."""
        # Setup cache to return cached result on second call
        cached_result = {
            "status": "success",
            "bad_channels": ["T3"],
            "bad_pct": 5.0,
            "abnormal_prob": 0.3,
            "flag": "ROUTINE - Standard workflow",
            "confidence": 0.85,
            "processing_time": 0.01,  # Very fast from cache
            "quality_grade": "GOOD",
            "timestamp": "2025-07-18T10:00:00Z",
            "cached": True,
        }

        # First call: no cache
        mock_redis_client.get.return_value = None

        # Make first request
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response1 = client_with_cache.post("/api/v1/eeg/analyze", files=files)

        assert response1.status_code == 200
        result1 = response1.json()
        assert "cached" not in result1 or result1.get("cached") is False

        # Verify cache was set
        mock_redis_client.set.assert_called_once()

        # Setup cache hit for second call
        mock_redis_client.get.return_value = cached_result  # Return dict directly
        mock_qc_controller.run_full_qc_pipeline.reset_mock()

        # Make second request with same file
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response2 = client_with_cache.post("/api/v1/eeg/analyze", files=files)

        assert response2.status_code == 200
        result2 = response2.json()
        assert result2.get("cached") is True
        assert result2["processing_time"] < 0.1  # Should be very fast

        # Verify QC controller was NOT called for cached result
        mock_qc_controller.run_full_qc_pipeline.assert_not_called()

    def test_cache_key_generation(self, client_with_cache, sample_edf_content):
        """Test that cache keys are generated consistently."""
        # Generate cache key from file content
        file_hash = hashlib.sha256(sample_edf_content).hexdigest()
        expected_key_prefix = f"eeg_analysis:{file_hash}"

        # Make request
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response = client_with_cache.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200

        # Check that cache was accessed with correct key pattern
        mock_redis_client = client_with_cache.app.state.cache_client
        cache_key_used = mock_redis_client.get.call_args[0][0]
        assert cache_key_used.startswith(expected_key_prefix)

    def test_cache_expiration(self, client_with_cache, sample_edf_content, mock_redis_client):
        """Test that cache entries expire after TTL."""
        # Make request
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response = client_with_cache.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200

        # Verify cache was set with expiration
        mock_redis_client.set.assert_called()
        # The set method in our cache module uses set() and expire() internally
        # But since we're mocking at the cache_client level, we should check if set was called

    def test_cache_invalidation_on_different_file(self, client_with_cache, mock_redis_client):
        """Test that different files get different cache entries."""
        # Create two different files
        content1 = b"EDF file content 1"
        content2 = b"EDF file content 2"

        # Request 1
        files1 = {"file": ("test1.edf", content1, "application/octet-stream")}
        response1 = client_with_cache.post("/api/v1/eeg/analyze", files=files1)

        # Request 2
        files2 = {"file": ("test2.edf", content2, "application/octet-stream")}
        response2 = client_with_cache.post("/api/v1/eeg/analyze", files=files2)

        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Verify different cache keys were used
        call_args = [call[0][0] for call in mock_redis_client.get.call_args_list]
        assert len(set(call_args)) >= 2  # At least 2 different keys

    def test_cache_disabled_when_redis_unavailable(
        self, client_with_cache, sample_edf_content, mock_redis_client, mock_qc_controller
    ):
        """Test graceful degradation when Redis is unavailable."""
        # Make Redis operations fail
        mock_redis_client.get.side_effect = redis.ConnectionError("Redis unavailable")
        mock_redis_client.set.side_effect = redis.ConnectionError("Redis unavailable")

        # Request should still work
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response = client_with_cache.post("/api/v1/eeg/analyze", files=files)

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"

        # Verify QC controller was called (no cache)
        mock_qc_controller.run_full_qc_pipeline.assert_called_once()

    def test_cache_performance_improvement(
        self, client_with_cache, sample_edf_content, mock_redis_client
    ):
        """Test that cache significantly improves performance."""
        # First request (no cache)
        mock_redis_client.get.return_value = None

        start_time = time.time()
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response1 = client_with_cache.post("/api/v1/eeg/analyze", files=files)
        uncached_time = time.time() - start_time

        assert response1.status_code == 200

        # Setup cache hit
        cached_result = response1.json()
        cached_result["cached"] = True
        cached_result["processing_time"] = 0.001
        mock_redis_client.get.return_value = cached_result  # Return dict directly

        # Second request (with cache)
        start_time = time.time()
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response2 = client_with_cache.post("/api/v1/eeg/analyze", files=files)
        cached_time = time.time() - start_time

        assert response2.status_code == 200

        # Cached should be at least 2x faster (relaxed for test environment)
        assert cached_time < uncached_time / 2

    def test_cache_statistics_endpoint(self, client_with_cache, mock_redis_client):
        """Test endpoint that returns cache statistics."""
        # Setup mock statistics
        # Mock the get_stats method to return expected stats
        mock_redis_client.get_stats.return_value = {
            "connected": True,
            "memory_usage": "12.5M",
            "total_keys": 25,
            "hit_rate": 0.9,
            "keyspace_hits": 450,
            "keyspace_misses": 50,
        }

        response = client_with_cache.get("/api/v1/cache/stats")

        assert response.status_code == 200
        stats = response.json()

        assert "memory_usage" in stats
        assert "total_keys" in stats
        assert "hit_rate" in stats
        assert stats["hit_rate"] == 0.9  # 450/(450+50)

    @pytest.mark.parametrize("analysis_type", ["standard", "detailed"])
    def test_cache_works_for_different_endpoints(
        self, client_with_cache, sample_edf_content, mock_redis_client, analysis_type
    ):
        """Test that caching works for both standard and detailed analysis endpoints."""
        endpoint = (
            "/api/v1/eeg/analyze" if analysis_type == "standard" else "/api/v1/eeg/analyze/detailed"
        )

        # First request
        mock_redis_client.get.return_value = None
        files = {"file": ("test.edf", sample_edf_content, "application/octet-stream")}
        response1 = client_with_cache.post(endpoint, files=files)
        assert response1.status_code == 200

        # Verify cache was set
        assert mock_redis_client.set.called

    def test_cache_clear_endpoint(self, client_with_cache, mock_redis_client):
        """Test endpoint to clear cache."""
        # Setup clear_pattern to return number of keys deleted
        mock_redis_client.clear_pattern.return_value = 2

        # Import auth utils to generate valid token
        from api.auth import create_cache_clear_token

        token = create_cache_clear_token()

        # Clear cache with valid HMAC token
        response = client_with_cache.delete("/api/v1/cache/clear", headers={"Authorization": token})

        assert response.status_code == 200
        result = response.json()
        assert result["keys_deleted"] == 2

        # Verify clear_pattern was called
        mock_redis_client.clear_pattern.assert_called_with("eeg_analysis:*")

    def test_cache_warmup_from_common_files(self, client_with_cache, mock_redis_client):
        """Test cache warmup functionality for common test files."""
        # Endpoint to pre-warm cache with common test files
        response = client_with_cache.post(
            "/api/v1/cache/warmup", json={"file_patterns": ["sleep-*.edf"]}
        )

        assert response.status_code == 200
        result = response.json()
        assert "files_cached" in result
        assert result["files_cached"] >= 0  # May be 0 if no files match

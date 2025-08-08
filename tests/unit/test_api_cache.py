"""CLEAN tests for API cache - no bullshit mocks, test real logic."""

import json
from unittest.mock import MagicMock, patch

import pytest

from brain_go_brrr.api.cache import CACHE_VERSION, RedisCache


class TestRedisCacheClean:
    """Test RedisCache with minimal mocking - focus on REAL logic."""

    @pytest.fixture
    def mock_infra_cache(self):
        """Create a minimal mock of infra cache that behaves like real cache."""
        cache = MagicMock()
        cache.connected = True
        # Use a real dict to store values - no fake bullshit
        cache._storage = {}

        def get(key):
            return cache._storage.get(key)

        def set(key, value, expiry=None):
            cache._storage[key] = value
            return True

        def delete(key):
            if key in cache._storage:
                del cache._storage[key]
                return 1
            return 0

        cache.get = get
        cache.set = set
        cache.delete = delete
        cache.clear_pattern = MagicMock(return_value=5)
        cache.get_stats = MagicMock(return_value={"hits": 10, "misses": 2})
        cache.health_check = MagicMock(return_value={"status": "healthy"})

        return cache

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_init_connects_to_infra_cache(self, mock_get_cache, mock_infra_cache):
        """Test that RedisCache properly delegates to infra cache."""
        mock_get_cache.return_value = mock_infra_cache

        cache = RedisCache()

        assert cache.connected is True
        assert cache._cache == mock_infra_cache
        mock_get_cache.assert_called_once()

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_generate_cache_key_creates_deterministic_keys(self, mock_get_cache, mock_infra_cache):
        """Test cache key generation is deterministic and includes versioning."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        # Same content should generate same key
        content1 = b"test eeg data"
        key1 = cache.generate_cache_key(content1, "qc")
        key2 = cache.generate_cache_key(content1, "qc")

        assert key1 == key2
        assert CACHE_VERSION in key1
        assert "qc" in key1

        # Different content should generate different key
        content2 = b"different eeg data"
        key3 = cache.generate_cache_key(content2, "qc")
        assert key3 != key1

        # Different analysis type should generate different key
        key4 = cache.generate_cache_key(content1, "sleep")
        assert key4 != key1
        assert "sleep" in key4

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_set_and_get_with_metadata(self, mock_get_cache, mock_infra_cache):
        """Test setting and getting values with metadata stripping."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        # Set a value
        test_data = {"result": "abnormal", "confidence": 0.95}
        success = cache.set("test_key", test_data, ttl=300)

        assert success is True

        # Get the value back
        retrieved = cache.get("test_key")

        # Should get original data without metadata
        assert retrieved == test_data
        assert "_cache_meta" not in retrieved

        # Verify metadata was stored internally
        raw_data = json.loads(mock_infra_cache._storage["test_key"])
        assert "_cache_meta" in raw_data
        assert raw_data["_cache_meta"]["cache_version"] == CACHE_VERSION

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_get_nonexistent_key_returns_none(self, mock_get_cache, mock_infra_cache):
        """Test getting non-existent key returns None."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        result = cache.get("nonexistent_key")
        assert result is None

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_exists_checks_key_presence(self, mock_get_cache, mock_infra_cache):
        """Test exists method properly checks key presence."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        # Key doesn't exist
        assert cache.exists("test_key") is False

        # Add key
        cache.set("test_key", {"data": "value"})

        # Key exists
        assert cache.exists("test_key") is True

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_delete_removes_key(self, mock_get_cache, mock_infra_cache):
        """Test delete properly removes keys."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        # Add a key
        cache.set("test_key", {"data": "value"})
        assert cache.exists("test_key") is True

        # Delete it
        result = cache.delete("test_key")
        assert result is True
        assert cache.exists("test_key") is False

        # Delete non-existent key
        result = cache.delete("nonexistent")
        assert result is False

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_clear_pattern_delegates_correctly(self, mock_get_cache, mock_infra_cache):
        """Test clear_pattern delegates to infra cache."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        result = cache.clear_pattern("eeg_analysis:*")

        assert result == 5
        mock_infra_cache.clear_pattern.assert_called_once_with("eeg_analysis:*")

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_get_stats_returns_cache_statistics(self, mock_get_cache, mock_infra_cache):
        """Test get_stats returns proper statistics."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        stats = cache.get_stats()

        assert stats == {"hits": 10, "misses": 2}
        mock_infra_cache.get_stats.assert_called_once()

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_health_check_returns_status(self, mock_get_cache, mock_infra_cache):
        """Test health_check returns proper status."""
        mock_get_cache.return_value = mock_infra_cache
        cache = RedisCache()

        health = cache.health_check()

        assert health == {"status": "healthy"}
        mock_infra_cache.health_check.assert_called_once()

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_error_handling_in_get(self, mock_get_cache, mock_infra_cache):
        """Test get handles errors gracefully."""
        mock_get_cache.return_value = mock_infra_cache
        mock_infra_cache.get = MagicMock(side_effect=Exception("Redis error"))

        cache = RedisCache()
        result = cache.get("test_key")

        assert result is None  # Should return None on error

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_error_handling_in_set(self, mock_get_cache, mock_infra_cache):
        """Test set handles errors gracefully."""
        mock_get_cache.return_value = mock_infra_cache
        mock_infra_cache.set = MagicMock(side_effect=Exception("Redis error"))

        cache = RedisCache()
        result = cache.set("test_key", {"data": "value"})

        assert result is False  # Should return False on error

    @patch("brain_go_brrr.api.cache.get_infra_cache")
    def test_error_handling_in_exists(self, mock_get_cache, mock_infra_cache):
        """Test exists handles errors gracefully."""
        mock_get_cache.return_value = mock_infra_cache
        mock_infra_cache.get = MagicMock(side_effect=Exception("Redis error"))

        cache = RedisCache()
        result = cache.exists("test_key")

        assert result is False  # Should return False on error

"""Test Redis auto-reconnect functionality.

Following TDD and Uncle Bob's clean test principles.
"""

from unittest.mock import Mock, patch

import pytest
import redis

from api.cache import RedisCache


class TestRedisReconnect:
    """Test Redis connection resilience."""

    def test_redis_auto_reconnect_on_connection_error(self):
        """Test that Redis cache auto-reconnects when connection fails.

        Given: A Redis cache instance
        When: Connection is lost (ConnectionError)
        Then: Cache should auto-reconnect and retry operation
        """
        # Given: Mock Redis client that fails then succeeds
        mock_redis = Mock()
        mock_redis.get.side_effect = [
            redis.ConnectionError("Connection lost"),  # First call fails
            b"test_value"  # Second call succeeds after reconnect
        ]

        cache = RedisCache()

        # When: We patch the Redis client and attempt a get operation
        with patch.object(cache, '_redis', mock_redis):
            result = cache.get("test_key")

        # Then: Should have retried and succeeded
        assert result == "test_value"
        assert mock_redis.get.call_count == 2  # Failed once, retried once

    def test_redis_auto_reconnect_on_timeout(self):
        """Test that Redis cache handles timeout errors gracefully.

        Given: A Redis cache instance
        When: Operation times out
        Then: Should retry the operation
        """
        # Given: Mock Redis client that times out then succeeds
        mock_redis = Mock()
        mock_redis.set.side_effect = [
            redis.TimeoutError("Operation timed out"),  # First call times out
            True  # Second call succeeds
        ]

        cache = RedisCache()

        # When: We patch the Redis client and attempt a set operation
        with patch.object(cache, '_redis', mock_redis):
            result = cache.set("test_key", "test_value", ttl=300)

        # Then: Should have retried and succeeded
        assert result is True
        assert mock_redis.set.call_count == 2  # Timed out once, retried once

    def test_redis_gives_up_after_max_retries(self):
        """Test that Redis cache eventually gives up after max retries.

        Given: A Redis cache instance with persistent connection issues
        When: All retry attempts fail
        Then: Should raise the final exception
        """
        # Given: Mock Redis client that always fails
        mock_redis = Mock()
        mock_redis.get.side_effect = redis.ConnectionError("Persistent failure")

        cache = RedisCache()

        # When/Then: Should eventually give up and raise exception
        with patch.object(cache, '_redis', mock_redis):
            with pytest.raises(redis.ConnectionError, match="Persistent failure"):
                cache.get("test_key")

        # Should have tried multiple times before giving up
        assert mock_redis.get.call_count > 1

    def test_redis_successful_operation_no_retry(self):
        """Test that successful operations don't trigger retry logic.

        Given: A Redis cache instance with working connection
        When: Operation succeeds on first try
        Then: Should not retry
        """
        # Given: Mock Redis client that works immediately
        mock_redis = Mock()
        mock_redis.get.return_value = b"success"

        cache = RedisCache()

        # When: We perform a successful operation
        with patch.object(cache, '_redis', mock_redis):
            result = cache.get("test_key")

        # Then: Should succeed without retry
        assert result == "success"
        assert mock_redis.get.call_count == 1  # Called exactly once

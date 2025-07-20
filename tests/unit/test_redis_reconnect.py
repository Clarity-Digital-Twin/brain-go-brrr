"""Test Redis auto-reconnect functionality.

Following TDD and Uncle Bob's clean test principles.
"""

from unittest.mock import MagicMock, patch

import redis

from brain_go_brrr.api.cache import RedisCache


class TestRedisReconnect:
    """Test Redis connection resilience."""

    @patch("brain_go_brrr.api.cache.redis.Redis")
    def test_redis_auto_reconnect_on_connection_error(self, mock_redis_class):
        """Test that Redis cache auto-reconnects when connection fails.

        Given: A Redis cache instance with working initial connection
        When: Connection is lost during operation (ConnectionError)
        Then: Cache should auto-reconnect and retry operation
        """
        # Given: Mock Redis client that works for ping but fails then succeeds for get
        mock_client = MagicMock()
        mock_client.ping.return_value = True  # Initial connection works
        mock_client.get.side_effect = [
            redis.ConnectionError("Connection lost"),  # First call fails
            '{"test": "test_value"}',  # Second call succeeds after reconnect - returns JSON string
        ]
        mock_redis_class.return_value = mock_client

        cache = RedisCache()

        # When: We attempt a get operation that initially fails
        result = cache.get("test_key")

        # Then: Should have retried and succeeded
        assert result == {"test": "test_value"}
        assert mock_client.get.call_count == 2  # Failed once, retried once

    @patch("brain_go_brrr.api.cache.redis.Redis")
    def test_redis_auto_reconnect_on_timeout(self, mock_redis_class):
        """Test that Redis cache handles timeout errors gracefully.

        Given: A Redis cache instance with working connection
        When: Operation times out
        Then: Should retry the operation
        """
        # Given: Mock Redis client that works for ping but times out then succeeds for set
        mock_client = MagicMock()
        mock_client.ping.return_value = True  # Initial connection works
        mock_client.set.side_effect = [
            redis.TimeoutError("Operation timed out"),  # First call times out
            True,  # Second call succeeds
        ]
        mock_client.expire.return_value = True  # expire should also succeed
        mock_redis_class.return_value = mock_client

        cache = RedisCache()

        # When: We attempt a set operation that initially times out
        result = cache.set("test_key", {"value": "test_value"}, ttl=300)

        # Then: Should have retried and succeeded
        assert result is True
        assert mock_client.set.call_count == 2  # Timed out once, retried once

    @patch("brain_go_brrr.api.cache.redis.Redis")
    def test_redis_gives_up_after_max_retries(self, mock_redis_class):
        """Test that Redis cache eventually gives up after max retries.

        Given: A Redis cache instance with persistent connection issues
        When: All retry attempts fail
        Then: Should raise the final exception
        """
        # Given: Mock Redis client that always fails
        mock_client = MagicMock()
        mock_client.ping.return_value = True  # Initial connection works
        mock_client.get.side_effect = redis.ConnectionError("Persistent failure")
        mock_redis_class.return_value = mock_client

        cache = RedisCache()

        # When/Then: Should eventually give up and return None (graceful degradation)
        result = cache.get("test_key")

        # Should return None when all retries fail
        assert result is None
        # Should have tried multiple times before giving up
        assert mock_client.get.call_count >= 1

    @patch("brain_go_brrr.api.cache.redis.Redis")
    def test_redis_successful_operation_no_retry(self, mock_redis_class):
        """Test that successful operations don't trigger retry logic.

        Given: A Redis cache instance with working connection
        When: Operation succeeds on first try
        Then: Should not retry
        """
        # Given: Mock Redis client that works immediately
        mock_client = MagicMock()
        mock_client.ping.return_value = True  # Initial connection works
        mock_client.get.return_value = '{"result": "success"}'
        mock_redis_class.return_value = mock_client

        cache = RedisCache()

        # When: We perform a successful operation
        result = cache.get("test_key")

        # Then: Should succeed without retry
        assert result == {"result": "success"}
        assert mock_client.get.call_count == 1  # Called exactly once

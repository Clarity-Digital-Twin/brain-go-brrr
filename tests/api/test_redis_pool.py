"""Tests for Redis connection pool implementation."""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from redis.exceptions import ConnectionError, TimeoutError

from api.services.redis_pool import (
    CircuitBreaker,
    CircuitState,
    RedisConnectionPool,
    close_redis_pool,
    get_redis_pool,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Simulate failures
        for _ in range(3):
            with pytest.raises(ConnectionError):
                cb.call(Mock(side_effect=ConnectionError("Test error")))

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

        # Open the circuit
        with pytest.raises(ConnectionError):
            cb.call(Mock(side_effect=ConnectionError("Test error")))

        # Should reject without calling function
        with pytest.raises(ConnectionError, match="Circuit breaker is OPEN"):
            cb.call(Mock())

    def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker enters half-open state after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Open the circuit
        with pytest.raises(ConnectionError):
            cb.call(Mock(side_effect=ConnectionError("Test error")))

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should try again (half-open)
        successful_mock = Mock(return_value="success")
        result = cb.call(successful_mock)

        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes on successful call."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        # Some failures but not enough to open
        for _ in range(2):
            with pytest.raises(ConnectionError):
                cb.call(Mock(side_effect=ConnectionError("Test error")))

        # Successful call should reset
        result = cb.call(Mock(return_value="success"))

        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


# Module-level fixtures available to all test classes
@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch("redis.Redis") as mock:
        client = MagicMock()
        client.ping.return_value = True
        client.info.return_value = {
            "redis_version": "7.0.0",
            "connected_clients": 5,
            "used_memory_human": "1.5M",
        }
        client.info.return_value = {"keyspace_hits": 100, "keyspace_misses": 20}
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_connection_pool():
    """Mock connection pool."""
    with patch("redis.connection.ConnectionPool") as mock:
        pool = MagicMock()
        pool.max_connections = 50
        pool.created_connections = 5
        pool._available_connections = [1, 2, 3]
        pool._in_use_connections = {4: True, 5: True}
        mock.return_value = pool
        yield mock


class TestRedisConnectionPool:
    """Test Redis connection pool functionality."""

    def test_pool_initialization(self, mock_redis, mock_connection_pool):
        """Test connection pool initialization."""
        RedisConnectionPool(host="localhost", port=6379, max_connections=100)

        # Verify pool was created with correct parameters
        mock_connection_pool.assert_called_once()
        call_kwargs = mock_connection_pool.call_args.kwargs
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 6379
        assert call_kwargs["max_connections"] == 100

    def test_get_client_context_manager(self, mock_redis, mock_connection_pool):
        """Test get_client context manager."""
        pool = RedisConnectionPool()

        with pool.get_client() as client:
            assert client is not None
            # Client should be a Redis instance
            mock_redis.assert_called_with(connection_pool=pool.pool)

    def test_execute_with_retry(self, mock_redis, mock_connection_pool):
        """Test execute method with retry logic."""
        pool = RedisConnectionPool(max_retries=3)

        # Mock client that fails twice then succeeds
        client = MagicMock()
        client.get.side_effect = [
            ConnectionError("Test error 1"),
            ConnectionError("Test error 2"),
            "success",
        ]
        mock_redis.return_value = client

        # Should succeed after retries
        result = pool.execute("get", "test_key")
        assert result == "success"
        assert client.get.call_count == 3

    def test_execute_max_retries_exceeded(self, mock_redis, mock_connection_pool):
        """Test execute fails after max retries."""
        pool = RedisConnectionPool(max_retries=2)

        # Mock client that always fails
        client = MagicMock()
        client.get.side_effect = ConnectionError("Test error")
        mock_redis.return_value = client

        # Should raise after max retries
        with pytest.raises(ConnectionError):
            pool.execute("get", "test_key")

        assert client.get.call_count == 2

    def test_health_check(self, mock_redis, mock_connection_pool):
        """Test health check functionality."""
        pool = RedisConnectionPool()

        # Mock client with health data
        client = MagicMock()
        client.ping.return_value = True
        client.info.side_effect = [
            {"redis_version": "7.0.0", "connected_clients": 10, "used_memory_human": "2.5M"},
            {"keyspace_hits": 1000, "keyspace_misses": 100},
        ]
        mock_redis.return_value = client

        health = pool.health_check()

        assert health["healthy"] is True
        assert health["version"] == "7.0.0"
        assert health["connected_clients"] == 10
        assert health["used_memory_human"] == "2.5M"
        assert health["hit_rate"] == 0.909  # 1000 / (1000 + 100)
        assert health["circuit_state"] == "closed"
        assert "pool" in health

    def test_health_check_failure(self, mock_redis, mock_connection_pool):
        """Test health check when Redis is down."""
        pool = RedisConnectionPool()

        # Mock client that fails ping
        client = MagicMock()
        client.ping.side_effect = ConnectionError("Connection refused")
        mock_redis.return_value = client

        health = pool.health_check()

        assert health["healthy"] is False
        assert "error" in health
        assert health["circuit_state"] == "closed"  # First failure

    def test_get_stats(self, mock_redis, mock_connection_pool):
        """Test get_stats method."""
        pool = RedisConnectionPool()

        stats = pool.get_stats()

        assert stats["max_connections"] == 50
        assert stats["created_connections"] == 5
        assert stats["available_connections"] == 3
        assert stats["in_use_connections"] == 2
        assert stats["circuit_state"] == "closed"
        assert stats["circuit_failures"] == 0

    def test_reset_circuit_breaker(self, mock_redis, mock_connection_pool):
        """Test manual circuit breaker reset."""
        pool = RedisConnectionPool()

        # Force circuit open
        pool.circuit_breaker.state = CircuitState.OPEN
        pool.circuit_breaker.failure_count = 5

        # Reset
        pool.reset_circuit_breaker()

        assert pool.circuit_breaker.state == CircuitState.CLOSED
        assert pool.circuit_breaker.failure_count == 0

    def test_close_pool(self, mock_redis, mock_connection_pool):
        """Test closing connection pool."""
        pool = RedisConnectionPool()

        pool.close()

        # Verify disconnect was called
        pool.pool.disconnect.assert_called_once()


class TestGlobalPool:
    """Test global pool management."""

    def test_get_redis_pool_singleton(self, mock_redis, mock_connection_pool):
        """Test get_redis_pool returns singleton."""
        # Clear any existing instance
        close_redis_pool()

        pool1 = get_redis_pool()
        pool2 = get_redis_pool()

        assert pool1 is pool2

    def test_close_redis_pool(self, mock_redis, mock_connection_pool):
        """Test closing global pool."""
        # Get pool
        pool = get_redis_pool()
        assert pool is not None

        # Close it
        close_redis_pool()

        # Should create new instance next time
        new_pool = get_redis_pool()
        assert new_pool is not pool


class TestIntegration:
    """Integration tests with real Redis (if available)."""

    @pytest.fixture
    def real_pool(self):
        """Create real Redis pool for integration tests."""
        try:
            pool = RedisConnectionPool(
                host="localhost", port=6379, max_connections=10, socket_timeout=1
            )
            # Test connection
            with pool.get_client() as client:
                client.ping()
            yield pool
            pool.close()
        except (ConnectionError, TimeoutError):
            pytest.skip("Redis not available for integration tests")

    @pytest.mark.integration
    def test_real_redis_operations(self, real_pool):
        """Test real Redis operations through pool."""
        test_key = "test:pool:key"
        test_value = "test_value"

        # Set value
        result = real_pool.execute("set", test_key, test_value)
        assert result is True

        # Get value
        result = real_pool.execute("get", test_key)
        assert result == test_value

        # Delete key
        result = real_pool.execute("delete", test_key)
        assert result == 1

        # Verify deleted
        result = real_pool.execute("get", test_key)
        assert result is None

    @pytest.mark.integration
    def test_real_redis_pipeline(self, real_pool):
        """Test pipeline operations."""
        with real_pool.get_client() as client:
            pipe = client.pipeline()

            # Queue operations
            pipe.set("test:1", "value1")
            pipe.set("test:2", "value2")
            pipe.get("test:1")
            pipe.get("test:2")

            # Execute
            results = pipe.execute()

            assert results == [True, True, "value1", "value2"]

            # Cleanup
            client.delete("test:1", "test:2")

    @pytest.mark.integration
    def test_real_redis_health_check(self, real_pool):
        """Test health check with real Redis."""
        health = real_pool.health_check()

        assert health["healthy"] is True
        assert "ping_ms" in health
        assert health["ping_ms"] < 100  # Should be fast on localhost
        assert "version" in health
        assert "pool" in health

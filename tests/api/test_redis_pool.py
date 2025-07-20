"""Test Redis connection pool manager functionality.

Following best practices for Redis testing:
- Mock at the correct import level
- Use dependency injection where possible
- Test business logic separately from Redis internals
- Follow pytest-redis patterns for integration tests
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from brain_go_brrr.infra.redis import RedisConnectionPool, close_redis_pool, get_redis_pool
from brain_go_brrr.infra.redis.pool import CircuitBreaker, CircuitState


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60

    def test_failure_threshold_reached(self):
        """Test circuit opens when failure threshold is reached."""
        cb = CircuitBreaker(failure_threshold=2)

        # Mock function that fails
        failing_func = Mock(side_effect=RedisError("Connection failed"))

        # First failure
        with pytest.raises(RedisError):
            cb.call(failing_func)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

        # Second failure - should open circuit
        with pytest.raises(RedisError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 2

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        # Cause a failure
        failing_func = Mock(side_effect=RedisError("Connection failed"))
        with pytest.raises(RedisError):
            cb.call(failing_func)
        assert cb.failure_count == 1

        # Then succeed
        success_func = Mock(return_value="success")
        result = cb.call(success_func)
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_circuit_open_blocks_execution(self):
        """Test circuit blocks execution when open."""
        cb = CircuitBreaker(failure_threshold=1)

        # Open the circuit
        failing_func = Mock(side_effect=RedisError("Connection failed"))
        with pytest.raises(RedisError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Should now block execution
        success_func = Mock(return_value="success")
        with pytest.raises(ConnectionError, match="Circuit breaker is OPEN"):
            cb.call(success_func)

    def test_circuit_recovery_after_timeout(self):
        """Test circuit allows execution after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        # Open the circuit
        failing_func = Mock(side_effect=RedisError("Connection failed"))
        with pytest.raises(RedisError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # Should block immediately
        success_func = Mock(return_value="success")
        with pytest.raises(ConnectionError):
            cb.call(success_func)

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should now allow execution and recover
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_call_with_success(self):
        """Test successful call through circuit breaker."""
        cb = CircuitBreaker()

        success_func = Mock(return_value="success")
        result = cb.call(success_func)

        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


@pytest.fixture
def mock_redis():
    """Mock Redis client following best practices."""
    with patch("redis.Redis") as mock:
        client = MagicMock()
        client.ping.return_value = True
        client.info.return_value = {
            "redis_version": "7.0.0",
            "connected_clients": 5,
            "used_memory_human": "1.5M",
            "keyspace_hits": 100,
            "keyspace_misses": 20,
        }
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_connection_pool():
    """Mock ConnectionPool at the correct import path."""
    with patch("infra.redis.pool.ConnectionPool") as mock:
        pool = MagicMock()
        pool.max_connections = 50
        pool._created_connections = 5  # Direct attribute, not method
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
            {
                "redis_version": "7.0.0",
                "connected_clients": 10,
                "used_memory_human": "2.5M",
            },
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
        assert health["circuit_state"] == "closed"

    def test_get_stats(self, mock_redis, mock_connection_pool):
        """Test get_stats method."""
        pool = RedisConnectionPool()

        stats = pool.get_stats()

        assert stats["max_connections"] == 50
        assert stats["created_connections"] == 5  # Now properly mocked as attribute
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
    """Test global pool management functions."""

    def setup_method(self):
        """Clean up global pool before each test."""
        close_redis_pool()

    def teardown_method(self):
        """Clean up global pool after each test."""
        close_redis_pool()

    @patch("infra.redis.pool.RedisConnectionPool")
    def test_get_redis_pool_singleton(self, mock_pool_class):
        """Test get_redis_pool returns singleton."""
        mock_instance = MagicMock()
        mock_pool_class.return_value = mock_instance

        # First call creates instance
        pool1 = get_redis_pool()
        assert pool1 is mock_instance
        mock_pool_class.assert_called_once()

        # Second call returns same instance
        pool2 = get_redis_pool()
        assert pool2 is pool1
        assert pool2 is mock_instance
        # Should not create a new instance
        assert mock_pool_class.call_count == 1

    @patch("infra.redis.pool.RedisConnectionPool")
    def test_close_redis_pool(self, mock_pool_class):
        """Test closing global pool."""
        mock_instance = MagicMock()
        mock_pool_class.return_value = mock_instance

        # Create pool
        pool = get_redis_pool()
        assert pool is mock_instance

        # Close pool
        close_redis_pool()
        mock_instance.close.assert_called_once()

        # Next call should create new instance
        get_redis_pool()
        assert mock_pool_class.call_count == 2


@pytest.mark.integration
@pytest.mark.redis
class TestRedisIntegration:
    """Integration tests requiring a real Redis instance."""

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

            assert len(results) == 4
            assert results[0] is True  # set result
            assert results[1] is True  # set result
            assert results[2] == b"value1"  # get result
            assert results[3] == b"value2"  # get result

            # Clean up
            client.delete("test:1", "test:2")

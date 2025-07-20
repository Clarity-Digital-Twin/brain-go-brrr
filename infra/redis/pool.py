"""Redis connection pool manager for improved performance and reliability.

This module provides a connection pool wrapper that:
- Manages connection pooling for better resource utilization
- Implements circuit breaker pattern for fault tolerance
- Provides connection health monitoring
- Supports automatic reconnection with exponential backoff
"""

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import Enum
from typing import Any

import redis
from redis.connection import ConnectionPool
from redis.exceptions import ConnectionError, RedisError, TimeoutError

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for Redis connections."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[BaseException] = RedisError,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count: int = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise ConnectionError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RedisConnectionPool:
    """Managed Redis connection pool with health monitoring and circuit breaker."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        socket_keepalive: bool = True,
        socket_keepalive_options: dict[int, int] | None = None,
        health_check_interval: int = 30,
        decode_responses: bool = True,
        retry_on_timeout: bool = True,
        max_retries: int = 3,
    ):
        """Initialize Redis connection pool.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            max_connections: Maximum number of connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            socket_keepalive: Enable TCP keepalive
            socket_keepalive_options: TCP keepalive options
            health_check_interval: Seconds between health checks
            decode_responses: Decode responses to strings
            retry_on_timeout: Retry on timeout errors
            max_retries: Maximum number of retries
        """
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries

        # Default keepalive options for better connection stability
        if socket_keepalive_options is None:
            socket_keepalive_options = {
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            }

        # Create connection pool
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            socket_keepalive=socket_keepalive,
            socket_keepalive_options=socket_keepalive_options,
            health_check_interval=health_check_interval,
            decode_responses=decode_responses,
            retry_on_timeout=retry_on_timeout,
        )

        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        # Connection tracking
        self._last_health_check = 0
        self._health_check_interval = health_check_interval

        # Test initial connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test connection to Redis."""
        try:
            client = redis.Redis(connection_pool=self.pool)
            client.ping()
            logger.info(f"Connected to Redis pool at {self.host}:{self.port}")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @contextmanager
    def get_client(self) -> Generator[redis.Redis, None, None]:
        """Get Redis client from pool with automatic cleanup.

        Usage:
            with pool.get_client() as client:
                client.set('key', 'value')
        """
        client = None
        try:
            client = redis.Redis(connection_pool=self.pool)
            yield client
        finally:
            # Connection automatically returned to pool when client is garbage collected
            pass

    def execute(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Execute Redis command with circuit breaker and retry logic.

        Args:
            command: Redis command name (e.g., 'get', 'set')
            *args: Command arguments
            **kwargs: Command keyword arguments

        Returns:
            Command result
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:

                def _execute() -> Any:
                    with self.get_client() as client:
                        cmd = getattr(client, command)
                        return cmd(*args, **kwargs)

                return self.circuit_breaker.call(_execute)

            except (ConnectionError, TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"Redis {command} error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = min(2**attempt, 10)
                    time.sleep(wait_time)
                    continue

        logger.error(f"Redis {command} failed after {self.max_retries} attempts")
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Redis {command} failed with unknown error")

    def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check.

        Returns:
            Health status dictionary
        """
        try:
            start_time = time.time()

            with self.get_client() as client:
                # Ping test
                client.ping()
                ping_time = (time.time() - start_time) * 1000

                # Get server info
                info: dict[str, Any] = client.info()  # type: ignore
                stats: dict[str, Any] = client.info("stats")  # type: ignore

                # Get pool stats
                pool_stats = {
                    "created_connections": getattr(self.pool, "_created_connections", 0),
                    "available_connections": len(self.pool._available_connections),
                    "in_use_connections": len(self.pool._in_use_connections),
                }

                # Calculate hit rate
                hits = stats.get("keyspace_hits", 0)
                misses = stats.get("keyspace_misses", 0)
                total_ops = hits + misses
                hit_rate = hits / total_ops if total_ops > 0 else 0.0

                return {
                    "healthy": True,
                    "ping_ms": round(ping_time, 2),
                    "version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "N/A"),
                    "hit_rate": round(hit_rate, 3),
                    "circuit_state": self.circuit_breaker.state.value,
                    "pool": pool_stats,
                }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "circuit_state": self.circuit_breaker.state.value,
            }

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Pool statistics dictionary
        """
        return {
            "max_connections": self.pool.max_connections,
            "created_connections": getattr(self.pool, "_created_connections", 0),
            "available_connections": len(self.pool._available_connections),
            "in_use_connections": len(self.pool._in_use_connections),
            "circuit_state": self.circuit_breaker.state.value,
            "circuit_failures": self.circuit_breaker.failure_count,
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self.circuit_breaker.state = CircuitState.CLOSED
        self.circuit_breaker.failure_count = 0
        logger.info("Circuit breaker manually reset")

    def close(self) -> None:
        """Close all connections in the pool."""
        self.pool.disconnect()
        logger.info("Redis connection pool closed")


# Global pool instance
_pool_instance: RedisConnectionPool | None = None


def get_redis_pool() -> RedisConnectionPool:
    """Get or create Redis connection pool.

    Returns:
        RedisConnectionPool instance
    """
    global _pool_instance

    if _pool_instance is None:
        _pool_instance = RedisConnectionPool()

    return _pool_instance


def close_redis_pool() -> None:
    """Close the global Redis pool."""
    global _pool_instance

    if _pool_instance is not None:
        _pool_instance.close()
        _pool_instance = None

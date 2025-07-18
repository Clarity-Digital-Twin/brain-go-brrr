"""Redis caching for EEG analysis results."""

import hashlib
import json
import logging
import time
from typing import Optional, Any
import redis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache wrapper for EEG analysis results."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 decode_responses: bool = True, socket_timeout: int = 5,
                 max_retries: int = 3, retry_on_timeout: bool = True):
        """Initialize Redis client with connection parameters."""
        self.host = host
        self.port = port
        self.db = db
        self.decode_responses = decode_responses
        self.socket_timeout = socket_timeout
        self.max_retries = max_retries
        self.retry_on_timeout = retry_on_timeout
        self._last_ping_time = 0
        self._ping_interval = 30  # seconds
        
        self._connect()
    
    def _connect(self):
        """Establish connection to Redis."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=self.decode_responses,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
                retry_on_timeout=self.retry_on_timeout,
                max_connections=50,
                health_check_interval=30
            )
            # Test connection
            self.client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache disabled.")
            self.client = None
            self.connected = False
    
    def _ensure_connected(self) -> bool:
        """Ensure Redis connection is active, reconnect if needed."""
        if not self.client:
            return False
            
        # Periodic health check
        current_time = time.time()
        if current_time - self._last_ping_time > self._ping_interval:
            try:
                self.client.ping()
                self._last_ping_time = current_time
                self.connected = True
            except (ConnectionError, TimeoutError):
                logger.warning("Redis connection lost, attempting reconnect...")
                self._connect()
        
        return self.connected
    
    def health_check(self) -> dict:
        """Perform health check on Redis connection."""
        if not self.client:
            return {"healthy": False, "error": "No client initialized"}
        
        try:
            start_time = time.time()
            self.client.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            info = self.client.info()
            return {
                "healthy": True,
                "ping_ms": round(ping_time, 2),
                "version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A")
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def generate_cache_key(self, file_content: bytes, analysis_type: str = "standard") -> str:
        """Generate cache key from file content hash."""
        file_hash = hashlib.sha256(file_content).hexdigest()
        return f"eeg_analysis:{file_hash}:{analysis_type}"
    
    def get(self, key: str) -> Optional[dict]:
        """Get cached result."""
        if not self._ensure_connected():
            return None
            
        try:
            cached = self.client.get(key)
            if cached:
                logger.info(f"Cache hit for key: {key}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: dict, ttl: int = 3600) -> bool:
        """Set cache entry with TTL."""
        if not self._ensure_connected():
            return False
            
        try:
            json_value = json.dumps(value)
            self.client.set(key, json_value)
            self.client.expire(key, ttl)
            logger.info(f"Cached result for key: {key} with TTL: {ttl}s")
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._ensure_connected():
            return False
            
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        if not self._ensure_connected():
            return False
            
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self._ensure_connected():
            return 0
            
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._ensure_connected():
            return {
                "connected": False,
                "memory_usage": "N/A",
                "total_keys": 0,
                "hit_rate": 0.0
            }
            
        try:
            info = self.client.info()
            stats = self.client.info("stats")
            
            # Calculate hit rate
            hits = stats.get("keyspace_hits", 0)
            misses = stats.get("keyspace_misses", 0)
            total_ops = hits + misses
            hit_rate = hits / total_ops if total_ops > 0 else 0.0
            
            return {
                "connected": True,
                "memory_usage": info.get("used_memory_human", "N/A"),
                "total_keys": self.client.dbsize(),
                "hit_rate": round(hit_rate, 3),
                "keyspace_hits": hits,
                "keyspace_misses": misses
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {
                "connected": False,
                "error": str(e)
            }


# Global cache instance
_cache_instance = None


def get_cache() -> Optional[RedisCache]:
    """Get or create cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance if _cache_instance.connected else None
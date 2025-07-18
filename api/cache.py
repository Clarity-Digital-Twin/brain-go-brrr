"""Redis caching for EEG analysis results."""

import hashlib
import json
import logging
from typing import Optional, Any
import redis
from redis.exceptions import ConnectionError, TimeoutError

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache wrapper for EEG analysis results."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 decode_responses: bool = True, socket_timeout: int = 5):
        """Initialize Redis client with connection parameters."""
        self.host = host
        self.port = port
        self.db = db
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=decode_responses,
                socket_timeout=socket_timeout
            )
            # Test connection
            self.client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {host}:{port}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache disabled.")
            self.client = None
            self.connected = False
    
    def generate_cache_key(self, file_content: bytes, analysis_type: str = "standard") -> str:
        """Generate cache key from file content hash."""
        file_hash = hashlib.sha256(file_content).hexdigest()
        return f"eeg_analysis:{file_hash}:{analysis_type}"
    
    def get(self, key: str) -> Optional[dict]:
        """Get cached result."""
        if not self.connected or self.client is None:
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
        if not self.connected or self.client is None:
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
        if not self.connected or self.client is None:
            return False
            
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        if not self.connected or self.client is None:
            return False
            
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self.connected or self.client is None:
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
        if not self.connected or self.client is None:
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
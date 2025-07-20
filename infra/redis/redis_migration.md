# Redis Connection Pool Migration Guide

## Overview

This guide helps migrate from the existing `RedisCache` implementation to the new `RedisConnectionPool` for improved performance and reliability.

## Key Benefits

1. **Connection Pooling**: Reuses connections instead of creating new ones
2. **Circuit Breaker**: Prevents cascading failures when Redis is down
3. **Better Monitoring**: Detailed pool and connection statistics
4. **Automatic Retry**: Built-in retry logic with exponential backoff
5. **Context Manager**: Safe connection handling with automatic cleanup

## Migration Steps

### 1. Update Cache Implementation

Replace the old implementation in `api/cache.py`:

```python
# Old way
from api.cache import RedisCache
cache = RedisCache()
result = cache.get("key")

# New way
from api.services.redis_pool import get_redis_pool
pool = get_redis_pool()
result = pool.execute("get", "key")
```

### 2. Use Context Manager for Batch Operations

```python
# Old way
cache.set("key1", value1)
cache.set("key2", value2)
cache.set("key3", value3)

# New way - more efficient
with pool.get_client() as client:
    pipe = client.pipeline()
    pipe.set("key1", value1)
    pipe.set("key2", value2)
    pipe.set("key3", value3)
    pipe.execute()
```

### 3. Update Health Checks

```python
# Old way
health = cache.health_check()

# New way - more detailed
health = pool.health_check()
# Returns:
# {
#     "healthy": True,
#     "ping_ms": 0.5,
#     "version": "7.0.4",
#     "connected_clients": 10,
#     "used_memory_human": "1.5M",
#     "hit_rate": 0.85,
#     "circuit_state": "closed",
#     "pool": {
#         "created_connections": 5,
#         "available_connections": 4,
#         "in_use_connections": 1
#     }
# }
```

### 4. Implement Graceful Shutdown

```python
# In FastAPI shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    from api.services.redis_pool import close_redis_pool
    close_redis_pool()
```

### 5. Update Error Handling

```python
# Old way
try:
    result = cache.get("key")
except Exception as e:
    logger.error(f"Cache error: {e}")
    result = None

# New way - circuit breaker handles failures
try:
    result = pool.execute("get", "key")
except ConnectionError as e:
    # Circuit breaker is open
    logger.warning(f"Redis unavailable: {e}")
    result = None
```

## Configuration

Environment variables for the new pool:

```bash
# Connection settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password  # Optional

# Pool settings
REDIS_MAX_CONNECTIONS=50
REDIS_SOCKET_TIMEOUT=5
REDIS_HEALTH_CHECK_INTERVAL=30

# Circuit breaker
REDIS_FAILURE_THRESHOLD=5
REDIS_RECOVERY_TIMEOUT=60
```

## Monitoring

The new pool provides better monitoring:

```python
# Get pool statistics
stats = pool.get_stats()
print(f"Connections in use: {stats['in_use_connections']}/{stats['max_connections']}")
print(f"Circuit state: {stats['circuit_state']}")

# Monitor in API endpoint
@app.get("/health/redis")
async def redis_health():
    pool = get_redis_pool()
    return pool.health_check()
```

## Performance Tips

1. **Use pipelines for batch operations**:

   ```python
   with pool.get_client() as client:
       pipe = client.pipeline()
       for key in keys:
           pipe.get(key)
       results = pipe.execute()
   ```

2. **Implement local caching for hot data**:

   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def get_cached_value(key: str):
       return pool.execute("get", key)
   ```

3. **Use appropriate TTLs**:

   ```python
   # Short TTL for frequently changing data
   pool.execute("setex", "user_session", 300, session_data)  # 5 minutes

   # Longer TTL for stable data
   pool.execute("setex", "analysis_result", 3600, result)  # 1 hour
   ```

## Testing

Test the migration with these scenarios:

1. **Normal operation**: Verify basic get/set works
2. **Connection failure**: Stop Redis and verify circuit breaker opens
3. **Recovery**: Start Redis and verify circuit breaker closes
4. **Load test**: Verify pool handles concurrent requests
5. **Memory test**: Monitor connection count under load

## Rollback Plan

If issues arise, keep the old implementation available:

```python
# Temporary compatibility wrapper
class RedisCache:
    def __init__(self):
        self.pool = get_redis_pool()

    def get(self, key):
        return self.pool.execute("get", key)

    def set(self, key, value, ttl=3600):
        return self.pool.execute("setex", key, ttl, value)
```

## Timeline

1. **Week 1**: Update development environment
2. **Week 2**: Update staging with monitoring
3. **Week 3**: Gradual production rollout
4. **Week 4**: Remove old implementation

# Enhance Health Check Endpoint

## Problem
Current health endpoint only returns basic status. Need comprehensive health checks for production monitoring.

## Requirements
Add the following health checks:
1. **Model Status** - Verify EEGPT model is loaded and responsive
2. **Redis Status** - Check Redis connection and response time
3. **Disk Space** - Ensure adequate space for EDF processing
4. **Memory Usage** - Monitor Python process memory
5. **GPU Status** - If available, check GPU memory

## Implementation
Update `/api/v1/health` endpoint to return:
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2025-01-20T13:44:00Z",
  "version": "0.1.0",
  "checks": {
    "model": {
      "status": "healthy",
      "loaded": true,
      "response_time_ms": 45
    },
    "redis": {
      "status": "healthy",
      "connected": true,
      "response_time_ms": 2
    },
    "disk": {
      "status": "healthy",
      "free_gb": 250,
      "required_gb": 10
    },
    "memory": {
      "status": "healthy",
      "used_mb": 1024,
      "available_mb": 8192
    }
  }
}
```

## Acceptance Criteria
- [ ] All health checks implemented
- [ ] Proper error handling for each check
- [ ] Response time < 100ms
- [ ] Status degrades gracefully (not all-or-nothing)

@clod please work on this autonomously

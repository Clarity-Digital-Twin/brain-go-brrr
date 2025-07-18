# Wire Real Redis in CI with Docker Compose

## Current Status
Our Redis caching functionality is implemented and tested locally, but CI tests run without a real Redis instance. This limits our ability to test the full caching behavior.

## Tasks
- [ ] Create `docker-compose.test.yml` for CI environment
- [ ] Add Redis service to docker-compose
- [ ] Update GitHub Actions workflow to use docker-compose
- [ ] Configure test environment to connect to Redis container
- [ ] Add integration tests for Redis caching
- [ ] Test auto-reconnect behavior in CI

## Implementation Plan
1. Create docker-compose configuration:
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

2. Update GitHub Actions to start services before tests
3. Set REDIS_URL environment variable in CI
4. Add wait-for-redis logic in tests

## Priority
Medium - Important for comprehensive testing but not blocking core functionality

@claude Please add Redis to our CI pipeline using docker-compose so we can test the full caching functionality in GitHub Actions.

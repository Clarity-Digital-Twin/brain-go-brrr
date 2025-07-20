# Fix Redis Cache Serialization Issues

## Problem
5 test failures remain due to Redis cache serialization issues with JobData dataclass.

## Error Details
```
TypeError: Object of type JobData is not JSON serializable
```

## Root Cause
The cache layer is trying to JSON serialize JobData dataclass instances directly, but they contain datetime objects and other non-serializable fields.

## Solution
1. Add custom JSON encoder for JobData in `src/brain_go_brrr/infra/cache.py`
2. Handle datetime serialization properly
3. Consider using pickle for complex objects or implement `to_dict()` method

## Acceptance Criteria
- [ ] All 5 failing tests pass
- [ ] Cache properly stores and retrieves JobData objects
- [ ] No performance regression

## Files to Modify
- `src/brain_go_brrr/infra/cache.py`
- `src/brain_go_brrr/api/schemas.py` (add serialization methods)

@clod please work on this autonomously

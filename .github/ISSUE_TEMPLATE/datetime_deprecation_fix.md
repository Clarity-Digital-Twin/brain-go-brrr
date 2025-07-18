---
name: Fix datetime.utcnow() Deprecation Warnings
about: Update deprecated datetime usage to timezone-aware objects
title: 'fix: Replace deprecated datetime.utcnow() with timezone-aware datetime'
labels: bug, maintenance
assignees: ''
---

## Problem Statement
Multiple deprecation warnings appear in tests due to using `datetime.utcnow()`, which is deprecated in Python 3.12+. We need to update to timezone-aware datetime objects.

## Files to Update
- `api/main.py`: Lines 105, 191, 211, 368
- Any other files using `datetime.utcnow()`

## Requirements
1. Replace all instances of `datetime.utcnow()` with `datetime.now(datetime.UTC)`
2. Import `datetime.UTC` or use `timezone.utc` for older Python compatibility
3. Ensure all datetime objects are timezone-aware
4. Update any datetime comparisons to handle timezone awareness

## Implementation Details
```python
# Old (deprecated)
timestamp = datetime.utcnow().isoformat()

# New (timezone-aware)
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).isoformat()

# Or for Python 3.11+ compatibility
timestamp = datetime.now(datetime.UTC).isoformat()
```

## Acceptance Criteria
- [ ] Zero datetime deprecation warnings in test output
- [ ] All timestamps include timezone information
- [ ] Backward compatibility maintained for API responses
- [ ] Tests pass on Python 3.11, 3.12, and 3.13

@claude please fix all datetime deprecation warnings by updating to timezone-aware datetime objects. Ensure compatibility with Python 3.11+ and update any related tests.

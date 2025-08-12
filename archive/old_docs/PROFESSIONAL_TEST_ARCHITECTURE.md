# Professional Test Architecture - Zero Skips Strategy

## The Problem (First Principles Analysis)

When you run tests without proper filters, pytest COLLECTS integration tests then SKIPS them. This creates noise and confusion. Professional teams solve this with:

1. **Deselection over Skipping** - Don't collect what you won't run
2. **Consistent Filtering** - Every entry point uses same filters
3. **Clear Boundaries** - Unit vs Integration vs E2E

## The Solution We Implemented

### 1. Deselection in conftest.py
```python
def pytest_collection_modifyitems(config, items):
    """Deselect integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration", default=False):
        return
    
    # Deselect instead of skip for cleaner output
    drop = [it for it in items if "integration" in it.keywords]
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = [it for it in items if it not in drop]
```

### 2. Default Filters in pytest.ini
```ini
addopts =
    -m "not integration and not slow and not gpu and not external"
```

### 3. Consistent Makefile Targets
Every test target now includes proper filters:
- `make test` → excludes integration
- `make test-fast` → excludes integration  
- `make test-cov` → excludes integration
- `make test-all-cov` → excludes integration

### 4. Explicit Integration Testing
```bash
# Run integration tests explicitly
make test-integration
# Or with specific resources
pytest --run-integration -m integration
```

## Test Categories

### Unit Tests (620 tests)
- **No external dependencies**
- **No file I/O**
- **No network calls**
- **Use fixtures/fakes/DI**
- Run in <90 seconds

### Integration Tests (139 tests)
- **Require real resources**
- **Test component interactions**
- **May use real models/data**
- **Explicitly gated with flags**
- Run with `--run-integration`

## The Numbers

### Before Fix
```
pytest tests  # Would show 139 SKIPPED
```

### After Fix
```
pytest tests  # Shows 0 SKIPPED, 199 deselected
make test     # Shows 620 passed, 0 skipped
```

## Professional Patterns Applied

### 1. Dependency Injection
```python
class EEGPTWrapper:
    def __init__(self, model=None):
        self.model = model or load_real_model()
```

### 2. Synthetic Fixtures
```python
@pytest.fixture
def synthetic_sleep_raw():
    # 5 minutes of realistic EEG
    return create_synthetic_eeg()
```

### 3. Service Fakes
```python
@pytest.fixture
def redis_client():
    if has_redis():
        return real_redis()
    return fakeredis.FakeRedis()
```

### 4. Deterministic Seeds
```python
def pytest_sessionstart(session):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
```

## How Google/Meta Would Do It

### Test Pyramid
```
         /\
        /E2E\      (5%)  - Full system, real services
       /------\
      /Integr. \   (15%) - Component boundaries  
     /----------\
    /   Unit     \ (80%) - Fast, isolated, deterministic
   /--------------\
```

### Test Selection
```python
# Bazel/Buck style test selection
size = ["small", "medium", "large"]
tags = ["unit", "integration", "e2e", "gpu", "flaky"]

# Run only small tests
bazel test //... --test_size_filters=small

# Our equivalent
pytest -m "not integration and not slow"
```

### CI Strategy
```yaml
# Fast path (every commit)
- run: make test  # <5 minutes

# Full suite (pre-merge)
- run: make test-integration  # <30 minutes

# Nightly (comprehensive)
- run: make test-all --run-integration --with-gpu
```

## Enforcement

### Pre-Push Hook
```bash
#!/bin/bash
set -e
make lint
make type-check  
make test  # Fast tests only
```

### Coverage Gates
```python
# pytest.ini
[coverage:run]
fail_under = 70

# Makefile
test-cov:
    pytest --cov-fail-under=70
```

## The Result

**ZERO SKIPPED TESTS** in normal operation:
- Unit tests: 620 passing, 0 skipped
- Integration: 139 deselected (not skipped!)
- Clear separation of concerns
- Fast feedback loop
- Professional quality

## Maintenance Rules

1. **Never use `pytest.skip()` in test bodies** - Use markers
2. **Never use MagicMock for numerical code** - Use DI
3. **Always mark integration tests** - `@pytest.mark.integration`
4. **Keep unit tests under 100ms each** - Use fixtures
5. **Seed all randomness** - Deterministic tests

## Commands

```bash
# Daily development
make test          # 620 tests, <90s, no skips

# Pre-push verification  
make check-all     # Lint + Type + Test

# Full integration
make test-integration  # When you have resources

# Coverage analysis
make coverage      # With 70% threshold
```

This is how professional teams achieve ZERO SKIPS while maintaining clear test boundaries.
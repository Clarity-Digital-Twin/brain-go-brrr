# Test Coverage Status

## Current Situation

**Problem**: Standard pytest/coverage tools hang due to MNE imports at module level.

**Solution**: Created clean, focused tests with proper mocking.

## Coverage Estimate

Based on existing test files and module analysis:

### ✅ Well Tested (~80%+ coverage)
- `services.yasa_adapter` - Channel aliasing, sleep staging
- `core.config` - Configuration management  
- `models.linear_probe` - Sleep/abnormality classification
- `core.exceptions` - Error hierarchy (NEW)
- `core.edf_loader` - EDF file handling (NEW)

### ⚠️ Partially Tested (~40-60% coverage)
- `api.routers.*` - API endpoints have tests but not complete
- `core.preprocessing` - Some test files exist
- `utils.time` - Basic tests exist

### ❌ Needs Testing (~0-20% coverage)
- `models.eegpt_model` - Core model, critical path
- `data.tuab_dataset` - Dataset handling
- `data.tuab_cached_dataset` - Caching logic
- `core.channels` - Channel mapping
- `core.features.extractor` - Feature extraction

## Priority Testing Plan

### Phase 1: Critical Path (NO BULLSHIT)
1. ✅ `core.exceptions` - DONE
2. ✅ `core.edf_loader` - DONE  
3. 🔄 `models.eegpt_model` - In progress
4. ⏳ `data.tuab_dataset` - Next

### Phase 2: API Coverage
- Complete API endpoint tests
- Add integration tests for full workflows

### Phase 3: Utils and Helpers
- Channel mapping utilities
- Feature extraction helpers
- Preprocessing pipelines

## Testing Principles

1. **NO OVER-MOCKING**: Mock only external dependencies (MNE, torch models)
2. **CLEAN TESTS**: Each test should be readable and focused
3. **REAL LOGIC**: Test actual business logic, not just imports
4. **FAST EXECUTION**: Use mocks to avoid slow operations

## Estimated Total Coverage

**Current**: ~45% (based on file count and test analysis)
**Target**: 80% for critical paths, 60% overall
**Timeline**: 
- Phase 1: Today
- Phase 2: This week
- Phase 3: Ongoing

## How to Run Tests (Without Hanging)

```bash
# Run specific test files with mocking
uv run pytest tests/unit/test_core_exceptions.py -v --no-cov -o addopts=''

# Run our custom test files
uv run python tests/unit/test_core_exceptions.py  # If written as standalone

# Avoid these (they hang):
# - make test-cov
# - pytest with coverage enabled
# - Any test importing MNE at module level
```

## Next Steps

1. Create `test_eegpt_model.py` with proper torch mocking
2. Create `test_tuab_dataset.py` with data mocking
3. Fix pytest hanging issue (requires refactoring imports)
4. Set up CI to run tests without coverage initially
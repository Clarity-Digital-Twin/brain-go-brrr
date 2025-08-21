# 🔴 CRITICAL: Post-Refactoring Issues Found

## Executive Summary
The recent Clean Architecture refactoring (migrating from `core.*` to DDD layers) and EEGPT Unification broke **41 integration tests** that were never run after the refactoring. These tests have been failing silently because CI only runs them on `main` branch, not during development.

## 🐛 Issues Discovered

### 1. Import Path Issues (FIXED ✅)
- **Issue**: Test using old `core.quality.controller` path instead of new `brain_go_brrr.domain.quality.controller`
- **Files Affected**: `tests/api/test_api_sleep_edf.py`
- **Status**: Fixed in this session

### 2. EEGPT Model Implementation Broken (🔴 CRITICAL)
Multiple fundamental issues with the EEGPT model wrapper:

#### Missing Attributes
- `EEGPTModel` missing `config` attribute
- `EEGPTModel` missing `n_summary_tokens` attribute  
- `EEGPTModel` missing `_get_cached_channel_ids` method

#### Wrong Output Shapes
- Summary tokens returning shape `(1, 512)` instead of expected `(4, 512)`
- Feature extraction not producing expected 4 summary tokens
- Window scoring returning empty results

#### Poor Feature Discrimination
- Alpha/Beta patterns too similar (0.993 correlation, should be <0.95)
- Different frequency patterns too similar (0.996 correlation)
- Model not discriminating between different EEG patterns

### 3. Benchmark Tests Missing Plugin (FIXED ✅)
- **Issue**: `benchmark` fixture not available because pytest-benchmark plugin not loaded
- **Fix**: Added `-p benchmark` to Makefile test-benchmarks target
- **Status**: Fixed in this session

### 4. Integration Test Categories
**41 failures across these categories:**

#### API Tests (7 failures)
- Sleep-EDF processing endpoints
- Concurrent processing
- CLI streaming functionality

#### EEGPT Core (10 failures)  
- Model architecture tests
- Feature extraction pipeline
- Abnormality prediction
- Channel adaptation
- Summary token generation

#### Accuracy Tests (8 failures)
- Sensitivity below 80% requirement
- AUROC below 0.85 requirement  
- Cross-validation inconsistent
- Confidence calibration poor

#### Processing Pipeline (5 failures)
- Parallel processing broken
- TUAB file loading with autoreject
- Preprocessing not preserving patterns

## 📊 Test Results Summary

```
Integration Tests: 41 failed, 51 passed, 12 skipped
Benchmarks: Working when plugin loaded
Unit Tests: ~812 passing (from earlier runs)
```

## 🔧 Fixes Required

### Immediate (Blocking CI)
1. ✅ Fix import paths from `core.*` to new DDD structure
2. ✅ Add benchmark plugin to Makefile
3. ❌ Fix EEGPT model wrapper implementation
4. ❌ Add missing model attributes and methods

### High Priority  
1. Fix summary token generation (should return 4 tokens)
2. Fix feature discrimination between patterns
3. Fix abnormality detection accuracy
4. Fix parallel processing pipeline

### Medium Priority
1. Improve confidence calibration
2. Fix CLI streaming tests
3. Update deprecated method calls

## 🎯 Root Causes

1. **No Integration Testing During Refactoring**: The refactoring was done without running integration tests, allowing breaking changes to slip through.

2. **CI Only on Main**: Integration tests only run on `main` branch, not during development or staging, so issues weren't caught early.

3. **Incomplete Migration**: The EEGPT unification (replacing 3 deprecated models) wasn't fully completed - the new unified model is missing critical functionality.

4. **Plugin Autoload Disabled**: With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, plugins like pytest-benchmark must be explicitly loaded, which wasn't done.

## 📝 Recommendations

1. **Run integration tests on ALL branches** during development, not just main
2. **Add pre-push hooks** to run at least smoke integration tests
3. **Complete the EEGPT unification** properly with all required methods
4. **Add integration test gates** to PRs before merging
5. **Document the new architecture** so tests can be updated correctly

## 🚨 Current Status

- **Development Branch**: Core tests passing, integration not tested
- **Staging Branch**: Core tests passing, integration not tested  
- **Main Branch**: Core tests passing, integration/benchmarks failing

The codebase is **NOT stable for production** until these EEGPT model issues are fixed. The refactoring is incomplete and has broken critical functionality.

## Next Steps

1. Fix EEGPT model implementation to match test expectations
2. Run full integration test suite locally after each fix
3. Update CI to run integration tests on all branches
4. Add regression tests for all fixed issues
5. Document the new architecture properly
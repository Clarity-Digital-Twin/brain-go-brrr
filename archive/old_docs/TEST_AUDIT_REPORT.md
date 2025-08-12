# Test Suite Audit Report - Brain-Go-Brrr

## Executive Summary

**Current State**: ✅ All 690 tests passing (199 deselected for integration/benchmarks)
**Coverage**: ~70% (meets minimum target)
**Quality Issues**: Multiple anti-patterns identified that reduce test maintainability

## Critical Findings

### 1. ❌ Overly Permissive Assertions (`hasattr` everywhere)
**Count**: 30+ occurrences across test suite
**Impact**: Tests pass even when implementation is broken
**Files Most Affected**:
- `test_coverage_boost.py` (8 instances)
- `test_models_linear_probe.py` (4 instances)
- `test_edf_validator.py` (4 instances)

**Example**:
```python
# BAD - tests internals, not behavior
assert hasattr(probe, "dropout")
assert hasattr(probe, "classifier")

# GOOD - tests actual behavior
x = torch.randn(4, 20, 1024)
y = probe(x)
assert y.shape == (4, 2)
```

### 2. ❌ Global Class Patches
**Count**: 4 critical instances
**Files**: `test_coverage_boost.py`
**Impact**: Can leak across tests, brittle to refactoring

**Example**:
```python
# BAD - patches at import path
with patch("brain_go_brrr.core.config.Path.exists", return_value=True)

# GOOD - patch where used or inject dependency
with patch.object(detector_module.Path, "exists", return_value=True)
```

### 3. ❌ Mocking torch.load with Huge State Dicts
**Location**: `test_coverage_boost.py:117-138`
**Issue**: 20+ line mock state dict coupled to classifier internals
**Impact**: Breaks on any classifier architecture change

**Solution**: Add `classifier` parameter to constructor for test injection

### 4. ⚠️ Redis Test Without Behavior Verification
**Location**: `test_coverage_boost.py:145-159`
**Issue**: Only checks object creation, not actual Redis operations
**Impact**: Could pass with completely broken Redis client

### 5. ❌ Two-Layer Probe Test Regression
**Location**: `test_coverage_boost.py:206-215`
**Original**: Checked `n_classes`
**Current**: Only checks `hasattr` for internals
**Impact**: Lost actual functionality verification

### 6. ⚠️ Benchmark Test Logic Confusion
**Location**: `test_eegpt_performance.py`
**Issue**: Mixed `allclose` checks make CPU/GPU drift unclear
**Impact**: Can mask real performance regressions

## Dependency Injection Opportunities

### 1. ParallelEEGPipeline
```python
# Current (hardcoded dependencies)
def __init__(self, eegpt_model_path=None, device="cpu"):
    self.eegpt_extractor = EEGPTFeatureExtractor(...)
    self.sleep_analyzer = SleepAnalyzer()

# Improved (injectable)
def __init__(self, eegpt_model_path=None, device="cpu", 
             extractor=None, sleep_analyzer=None):
    self.eegpt_extractor = extractor or EEGPTFeatureExtractor(...)
    self.sleep_analyzer = sleep_analyzer or SleepAnalyzer()
```

### 2. AbnormalityDetector
```python
# Add optional classifier injection
def __init__(self, model_path, ..., classifier=None):
    ...
    if classifier:
        self.classifier = classifier
    else:
        self._init_classification_head()
```

### 3. EDFStreamer
```python
# Add reader factory for testing
def __init__(self, file_path, reader_factory=None):
    self.reader_factory = reader_factory or pyedflib.EdfReader
```

## Test Deselection Analysis

**Total**: 199 tests deselected
**Categories**:
- Integration tests: ~40 tests (marked with `@pytest.mark.integration`)
- Benchmark tests: ~20 tests (marked with `@pytest.mark.benchmark`)
- GPU tests: ~10 tests (marked with `@pytest.mark.gpu`)
- Slow tests: ~129 tests (various performance/load tests)

**All deselected tests PASS when run** - no hidden failures

## Anti-Pattern Distribution

| Anti-Pattern | Count | Severity | Files |
|-------------|-------|----------|-------|
| `hasattr` assertions | 30+ | High | 12 |
| Global patches | 4 | Critical | 1 |
| Deep mocking | 3 | High | 2 |
| Missing behavior tests | 15+ | Medium | 5 |
| Unclear assertions | 8 | Low | 3 |

## Recommended Fixes (Priority Order)

### Phase 1: Create Test Infrastructure (1 hour)
1. Create `tests/fakes.py` with proper test doubles
2. Add DI hooks to 3 main constructors
3. Create behavior-focused test utilities

### Phase 2: Fix Critical Tests (2 hours)
1. Replace `test_coverage_boost.py` tests with behavior assertions
2. Fix benchmark test assertions for clarity
3. Add proper Redis behavior verification

### Phase 3: Systematic Cleanup (3 hours)
1. Replace all `hasattr` with behavior tests
2. Remove global patches in favor of local mocks
3. Add missing edge case coverage

## Quick Win Improvements

### 1. Create `tests/fakes.py`:
```python
class FakeEEGPTBackbone:
    def extract_features(self, x):
        return np.zeros((x.shape[0], 2048))

class FakeClassifierHead(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 2)

class FakeRedis:
    def __init__(self):
        self.storage = {}
    def get(self, key):
        return self.storage.get(key)
    def set(self, key, value):
        self.storage[key] = value
        return True

class FakeEdfReader:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def getNSamples(self):
        return [1000] * 19
    def getSampleFrequency(self):
        return [256.0] * 19
```

### 2. Example Refactored Test:
```python
# BEFORE
def test_abnormal_detector_init():
    with patch("Path.exists"), patch("torch.load", huge_dict):
        detector = AbnormalityDetector(fake_path)
        assert hasattr(detector, "detect_abnormality")

# AFTER
def test_abnormal_detector_behavior():
    detector = AbnormalityDetector(
        model_path=fake_path,
        classifier=FakeClassifierHead()
    )
    # Test actual behavior
    result = detector.detect_abnormality(test_eeg_data)
    assert result.prediction in ["normal", "abnormal"]
    assert 0 <= result.confidence <= 1
```

## Coverage Analysis

**Current**: ~70% (minimum threshold)
**Target**: 85% for production readiness

**Lowest Coverage Modules**:
1. `brain_go_brrr.infra.*` - Redis/caching code
2. `brain_go_brrr.core.pipeline.*` - Parallel processing
3. `brain_go_brrr.data.tuab_*` - Dataset loaders
4. Error handling paths in all modules

## Test Execution Commands

```bash
# Quick unit tests (2s)
pytest -m "not integration and not benchmark" -q

# With coverage (10s)
make test-all-cov

# Integration tests (30s)
pytest tests/integration --run-integration -q

# GPU tests (requires CUDA)
CUDA_VISIBLE_DEVICES=0 pytest -m gpu -q

# All tests including deselected (80s)
pytest tests -q
```

## Conclusion

The test suite is **functionally complete** (all tests pass) but has **significant maintainability issues**. The anti-patterns identified make tests brittle and reduce their value as documentation and regression guards. The recommended fixes can be implemented incrementally without breaking existing functionality.

**Next Steps**:
1. Implement fakes.py (30 min)
2. Add DI hooks to 3 constructors (30 min)
3. Refactor test_coverage_boost.py (1 hour)
4. Systematically replace hasattr assertions (2 hours)

**Risk**: Low - all changes are test-only or backward-compatible API additions
**Impact**: High - significantly improved test maintainability and clarity
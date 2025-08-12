# Skipped Tests Analysis

## Summary
- **Total Tests**: 747 (583 passing, 164 skipped)
- **Skip Rate**: 22% (acceptable for a project with optional integration tests)

## Categories of Skipped Tests

### 1. ✅ APPROPRIATELY SKIPPED (138 tests - 84%)
These tests require external resources or heavy models and should remain skipped by default:

#### Integration Tests (require --run-integration flag)
- **124 tests** marked as integration tests
- These test real model loading, external data, or end-to-end pipelines
- **Correct behavior**: Skip by default, run with `--run-integration` flag
- Examples:
  - `test_eegpt_integration.py` - Real EEGPT model loading
  - `test_yasa_integration.py` - Full sleep analysis pipeline
  - `test_autoreject_pipeline.py` - Heavy preprocessing

#### Data Requirements
- **4 tests** require Sleep-EDF dataset (not downloaded)
- **1 test** requires TUH abnormal dataset
- **1 test** YASA requires >=5 min data (test has 10s)
- **1 test** Autoreject needs more data for cross-validation
- **Correct behavior**: Skip when data unavailable

#### GPU Tests
- **4 tests** require CUDA (GPU not available in CI)
- **Correct behavior**: Skip on CPU-only systems

#### Optional Dependencies
- **2 tests** require psutil for memory monitoring
- **Correct behavior**: Skip when optional deps missing

### 2. ❌ SHOULD BE FIXED (26 tests - 16%)
These tests are skipped due to API changes or missing implementations:

#### API Changes That Need Updates (20 tests)
```python
# EEGPTConfig API changed (9 tests)
- test_models_eegpt_model.py - Config uses dataclasses now
- Old: EEGPTConfig(n_channels=20)
- New: Uses @dataclass with different attrs

# Preprocessing API changed (9 tests)  
- test_core_preprocessing.py - Function signatures changed
- Need to update to match new preprocessing pipeline

# EEGPTWrapper methods (2 tests)
- forward() and normalize() signatures changed
```

#### Not Implemented Features (6 tests)
```python
# Missing implementations:
- TwoLayerProbe (2 tests) - Feature not implemented
- validate_edf_path (2 tests) - Function stub exists but not implemented
- Stream command (1 test) - CLI command not implemented
- minmax normalization (1 test) - Normalization type not added
```

#### Import Issues (1 test)
```python
# MNE import causes test collection errors
- test_data_tuab_cached_dataset.py
- Likely circular import or missing mock
```

## Action Plan

### Phase 1: Quick Fixes (Can fix today)
1. **Fix EEGPTConfig tests** (9 tests)
   - Update to match new dataclass-based config
   - Simple attribute name changes

2. **Fix preprocessing tests** (9 tests)
   - Update function signatures
   - Adjust expected parameters

3. **Fix import issue** (1 test)
   - Add proper mocking for MNE imports

### Phase 2: Implementation (This week)
1. **Implement validate_edf_path** (2 tests)
   - Simple path validation function
   - Check file exists and has .edf extension

2. **Update EEGPTWrapper** (2 tests)
   - Match new forward/normalize signatures
   - Update test expectations

### Phase 3: New Features (Later)
1. **TwoLayerProbe** - New model architecture
2. **Stream command** - CLI streaming feature
3. **minmax normalization** - Additional preprocessing option

## Running Integration Tests

To run the appropriately skipped integration tests:
```bash
# Run ALL tests including integration (slow)
pytest --run-integration

# Run specific integration test
pytest tests/integration/test_yasa_integration.py --run-integration

# Check if integration tests would pass
pytest tests/integration/ --run-integration --collect-only
```

## Recommendations

### ✅ Keep Skipped (Good Practice)
- Integration tests that require real models/data
- GPU-specific tests
- Tests requiring large datasets

### 🔧 Fix Now (Low Effort, High Value)
- API change updates (20 tests)
- Simple missing implementations (2-3 tests)
- Import issues (1 test)

### 📋 Track as Tech Debt
- TwoLayerProbe implementation
- Stream command feature
- Additional normalization methods

## Coverage Impact

Fixing the 26 "should be fixed" tests would:
- Add ~300-400 lines of coverage
- Increase total coverage by ~2-3%
- Improve confidence in core models

## Bottom Line

- **84% of skipped tests are appropriately skipped** (integration/data/GPU)
- **16% need fixing** due to API drift or missing implementations
- **Priority**: Fix the 20 API-related tests first (easy wins)
- **Keep skipped**: Integration tests that would slow down CI
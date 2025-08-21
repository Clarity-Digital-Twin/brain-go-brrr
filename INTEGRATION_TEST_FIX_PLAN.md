# Integration Test Fix Plan
**Date**: August 21, 2025  
**Goal**: Systematically fix all 34 integration test failures

## Root Cause Summary

After investigating from first principles, we found:

1. **Incomplete Refactoring**: The EEGPT unification created a compatibility wrapper but didn't fully implement the expected interface
2. **Import Path Mismatches**: Tests still reference old module paths that don't exist
3. **Shape Inconsistencies**: Model returns wrong tensor shapes for summary tokens
4. **Missing Mock Patches**: Test mocks aren't patching the correct locations

## Fix Priority Order

### Priority 1: Core EEGPT Model Issues (10 tests)
These are fundamental and block other fixes.

#### Issue 1.1: Import Path Errors
**Tests affected**: All in `test_eegpt_model_loading.py`
**Problem**: Tests patch `brain_go_brrr.models.eegpt_model` but should patch `brain_go_brrr.infra.ml_models`
**Fix**: Update all patch decorators to correct paths

#### Issue 1.2: Summary Token Shape
**Tests affected**: `test_eegpt_summary_tokens.py`
**Problem**: Model returns (1, 512) but tests expect (4, 512)
**Fix**: Implement proper summary token extraction in `eegpt_compat.py`

#### Issue 1.3: Feature Discrimination
**Tests affected**: `test_features_discriminate_between_patterns`
**Problem**: 99.6% correlation between different patterns (should be <95%)
**Fix**: Don't duplicate tokens, actually extract 4 distinct summary tokens

### Priority 2: Sleep-EDF API Failures (7 tests)
These affect the sleep analysis endpoints.

#### Issue 2.1: Real Data Processing
**Tests affected**: All Sleep-EDF integration tests
**Problem**: Mock vs real data handling inconsistency
**Fix**: Update mocks to match new model structure

### Priority 3: Accuracy Requirements (8 tests)
These are performance metrics that fail due to the token duplication.

#### Issue 3.1: Model Performance
**Tests affected**: AUROC, sensitivity, cross-validation tests
**Problem**: Duplicated tokens reduce model discriminative power
**Fix**: Will be resolved when Issue 1.2 is fixed

### Priority 4: Parallel Processing (1 test)
Low priority but affects scalability.

#### Issue 4.1: Pipeline Coordination
**Tests affected**: `test_parallel_pipeline.py`
**Problem**: Components work individually but not together
**Fix**: Review pipeline orchestration logic

## Implementation Steps

### Step 1: Fix Import Paths (Quick Win)
```bash
# Files to update:
- tests/unit/test_eegpt_model_loading.py
- tests/unit/test_eegpt_integration.py
- tests/unit/test_eegpt_summary_tokens.py

# Change all:
FROM: patch("brain_go_brrr.models.eegpt_model.XXX")
TO:   patch("brain_go_brrr.infra.ml_models.eegpt_wrapper.XXX")
```

### Step 2: Implement Proper Summary Tokens
```python
# In eegpt_compat.py, update extract_features():
# Instead of duplicating, extract the 4 CLS tokens properly
# The model should have 4 special tokens at the beginning
```

### Step 3: Fix Mock Configurations
```python
# Update all mock encoders to return correct shapes:
mock_encoder.extract_features.return_value = torch.randn(batch, 4, 512)
# Not: torch.randn(batch, 1, 512) or torch.randn(batch, 304, 512)
```

### Step 4: Verify Each Fix
```bash
# After each change:
pytest tests/unit/test_eegpt_model_loading.py -xvs --run-integration
# Verify no regression:
make test
```

## Success Metrics

- [ ] All 34 integration test failures resolved
- [ ] No regression in passing tests (800+ unit tests still pass)
- [ ] CI/CD green on all branches
- [ ] Benchmark tests produce valid JSON

## Time Estimate

Based on the investigation:
- Step 1 (Import paths): 30 minutes
- Step 2 (Summary tokens): 1 hour
- Step 3 (Mock configs): 30 minutes
- Step 4 (Verification): 30 minutes
- **Total**: ~2.5 hours

## Commands for Testing

```bash
# Run specific integration tests
pytest tests/unit/test_eegpt_model_loading.py -m integration --run-integration -xvs

# Run all integration tests
make test-integration

# Check CI locally
make check-all
```

## Next Actions

1. Start with import path fixes (lowest risk, highest impact)
2. Test each fix in isolation
3. Document any new issues discovered
4. Update this plan as we progress
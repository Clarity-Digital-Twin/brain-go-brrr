# Integration Test Audit - Why These Tests Exist

## THE PROBLEM
We have 164 skipped tests taking up space. Either they should WORK or be DELETED.

## CATEGORIES OF SKIPPED TESTS

### 1. INTEGRATION TESTS (52 tests marked @pytest.mark.integration)
**WHY THEY EXIST**: Test real model loading, external data, end-to-end pipelines
**WHY THEY'RE SKIPPED**: Prevent CI from taking 30+ minutes
**WHEN TO RUN**: 
- Before releases
- After major model changes
- Nightly builds
**CURRENT STATUS**: UNKNOWN - Need to verify if they pass

### 2. REQUIRES EXTERNAL DATA (7 tests)
**WHY THEY EXIST**: Test on real EEG datasets (Sleep-EDF, TUH)
**WHY THEY'RE SKIPPED**: Data not downloaded (gigabytes of EEG files)
**WHEN TO RUN**: After downloading datasets with `make download-data`
**EXAMPLES**:
```python
pytest.skip("Sleep-EDF data not available")
pytest.skip("TUH abnormal dataset not available")
```

### 3. GPU TESTS (4 tests)
**WHY THEY EXIST**: Test CUDA-accelerated model inference
**WHY THEY'RE SKIPPED**: CI runners don't have GPUs
**WHEN TO RUN**: On GPU-enabled machines
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
```

### 4. API DRIFT (20 tests)
**WHY THEY EXIST**: Were testing valid functionality
**WHY THEY'RE SKIPPED**: Code changed, tests weren't updated
**STATUS**: BROKEN - Need fixing
```python
# EEGPTConfig API changed (9 tests)
# Preprocessing API changed (9 tests)  
# EEGPTWrapper methods changed (2 tests)
```

### 5. UNIMPLEMENTED FEATURES (6 tests)
**WHY THEY EXIST**: Test-first development (TDD)
**WHY THEY'RE SKIPPED**: Features not built yet
**STATUS**: Placeholder tests for future work
```python
- TwoLayerProbe (2 tests)
- validate_edf_path (2 tests)
- Stream command (1 test)
- minmax normalization (1 test)
```

## THE REAL QUESTIONS

### Do Integration Tests Pass?
Let's find out RIGHT NOW:

```bash
# Test a simple one
pytest tests/unit/test_sleep_analysis.py --run-integration -k "test_sleep_stager_initialization"

# If that works, test all integration
pytest tests/ --run-integration --timeout=300
```

### Should They Be Deleted?
- **DELETE** if they've been broken >6 months
- **DELETE** if the feature they test is deprecated
- **FIX** if they test critical paths
- **KEEP SKIPPED** if they test optional features

## ACTION PLAN

### Step 1: Verify Integration Tests Work
```bash
# Run with strict timeout to prevent hangs
timeout 300 pytest tests/integration/ --run-integration -x

# Check specific categories
pytest -m integration --run-integration --collect-only  # Just collect, don't run
```

### Step 2: Document Each Skip Category
For EVERY skipped test, add a comment explaining:
1. WHY it exists
2. WHEN to run it
3. WHAT it requires (data/GPU/etc)

Example:
```python
@pytest.mark.skip(
    reason="integration test - requires EEGPT model (2GB). "
           "Run with --run-integration for full validation. "
           "Expected runtime: 30s"
)
def test_eegpt_full_pipeline():
    """Validates complete EEGPT inference pipeline with real model."""
    pass
```

### Step 3: Fix or Delete Broken Tests

#### FIX THESE (Low effort, high value):
- EEGPTConfig tests (9) - Just update to new API
- Preprocessing tests (9) - Update function signatures

#### DELETE THESE (If confirmed broken):
- Tests that have been skipped >6 months
- Tests for deprecated features
- Tests that duplicate other coverage

#### KEEP SKIPPED (But document):
- GPU tests (need hardware)
- Large dataset tests (need data)
- Slow integration tests (but they should PASS when run)

## VERIFICATION CHECKLIST

- [ ] Run ALL integration tests with timeout
- [ ] Document pass/fail status for each category
- [ ] Add clear skip reasons with requirements
- [ ] Delete tests that have been broken >6 months
- [ ] Fix tests that are simple API updates
- [ ] Add CI job that runs integration tests weekly

## WHY THIS MATTERS

**GOOD REASONS TO SKIP**:
- Requires expensive resources (GPU, large models)
- Takes >30 seconds (slows CI)
- Requires external data not in repo

**BAD REASONS TO SKIP**:
- "It's broken" - FIX IT OR DELETE IT
- "API changed" - UPDATE THE TEST
- "Not sure if it works" - FIND OUT

## BOTTOM LINE

Every skipped test should either:
1. **PASS** when its requirements are met (GPU/data/time)
2. **BE DELETED** if it's broken/obsolete

No test should exist in a permanent "skipped because broken" state.
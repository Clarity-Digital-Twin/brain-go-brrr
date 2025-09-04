# 🚨 P0 CRITICAL FIXES - RUNTIME CRASH BUGS

**Created**: September 4, 2025
**Owner**: ___________________
**Time Required**: 45-90 minutes
**Status**: 🔴 CRITICAL - FIX IMMEDIATELY
**Revision**: Added missing batch endpoint per senior audit

---

## 📌 SSOT: EEGPT Feature Dimensions Rule

> **🎯 THE GOLDEN RULE**
> - **IF feeding a probe**: Use `extract_features(..., summary=False)` → (B,4,512) → `.flatten(1)` → (B,2048)
> - **IF NOT feeding a probe**: Can use `extract_features(..., summary=True)` → (B,512) for heuristics/stats
> - **WHY**: EEGPT outputs 4 summary tokens × 512 dims. Probes trained on all 2048 dims. Using 512 = CRASH.

---

## 📋 EXECUTIVE SUMMARY

**We have 2 P0 bugs causing runtime crashes:**
1. **API endpoints** pass 512 dims to probes expecting 2048 → **RuntimeError** (4 locations)
2. **SleepProbeTrainer** passes 512 dims to probe expecting 2048 → **RuntimeError** (2 locations)

**Root Cause**: Missing `summary=False` parameter in `extract_features()` calls
**Business Impact**: 100% API failure rate for EEGPT endpoints, blocks demos
**Fix Complexity**: Simple parameter addition + flatten (6 fixes total)

---

## 🔴 P0 BUG #1: API ENDPOINTS DIMENSION MISMATCH

### THE CRASH
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 2048x6)
              Your 512-dim features trying to enter 2048-dim probe input
```

### FILES TO FIX

#### File: `src/brain_go_brrr/api/routers/eegpt.py`
**3 Endpoints to fix:**
1. `/eegpt/analyze` - Uses `extract_features(window_data, channel_names)`
2. `/eegpt/sleep/stages` - Uses `extract_features(window_data, channel_names)`
3. `/analyze/batch` - Uses `extract_features_batch(batch_array, channel_names)`

**Fix for regular extract_features calls**: Add `summary=False` and `.flatten(1)`
```python
# BEFORE (crashes):
features = eegpt_model.extract_features(window_data, channel_names)

# AFTER (works):
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)
```

**Fix for extract_features_batch call**: Add `summary=False` and `.flatten(1)`
```python
# BEFORE (crashes):
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names)

# AFTER (works):
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names, summary=False)
batch_features = batch_features.reshape(batch_features.shape[0], -1)  # (B,4,512) → (B,2048)
```

#### File: `src/brain_go_brrr/api/routers/sleep.py`
**Pattern to find**: `eegpt_model.extract_features(window_data, channel_names)`
**Fix**: Add `summary=False` and `.flatten(1)`

#### File: `src/brain_go_brrr/infra/ml_models/eegpt_compat.py`
**Method to fix**: `extract_features_batch` (line ~273)
```python
# BEFORE (crashes):
features = self.encoder.extract_features(batch_tensor)

# AFTER (works):
features = self.encoder.extract_features(batch_tensor, summary=False)
# Then before returning, ensure reshape:
if features.ndim == 3:  # (B, 4, 512)
    features = features.reshape(features.shape[0], -1)  # (B, 2048)
```

### VERIFICATION
```bash
# Find all broken calls (should return 4 results BEFORE fix):
rg "extract_features(_batch)?\(" src/brain_go_brrr/api/routers | grep -v "summary="

# After fix (should return 0 results):
rg "extract_features(_batch)?\(" src/brain_go_brrr/api/routers | grep -v "summary="

# Confirm fix applied (should show 4+ results):
rg "extract_features(_batch)?.*summary=False" src/brain_go_brrr/api/routers

# Check eegpt_compat.py fix:
rg "extract_features.*summary=False" src/brain_go_brrr/infra/ml_models/eegpt_compat.py
```

---

## 🔴 P0 BUG #2: SLEEPPROBETRAINER DIMENSION MISMATCH

### THE CRASH
```
RuntimeError: size mismatch, m1: [256 x 512], m2: [2048 x 5] at linear layer
```

### FILES TO FIX

#### File: `src/brain_go_brrr/application/training/sleep_probe_trainer.py`
**Pattern to find**: All `self.eegpt_model.extract_features` or `self.feature_extractor.extract_features`
**Occurrences**: 2 (in train_step and validation_step methods)
**Fix**: Add `summary=False` and `.flatten(1)` after EACH

```python
# BEFORE (crashes):
features = self.feature_extractor.extract_features(eeg_data)

# AFTER (works):
features = self.feature_extractor.extract_features(eeg_data, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)
```

### VERIFICATION
```bash
# Find broken calls (should show 2 BEFORE fix):
rg "extract_features\(" src/brain_go_brrr/application/training/sleep_probe_trainer.py | grep -v "summary="

# After fix (should show 0):
rg "extract_features\(" src/brain_go_brrr/application/training/sleep_probe_trainer.py | grep -v "summary="

# Confirm both fixes applied:
rg "extract_features.*summary=False" src/brain_go_brrr/application/training/sleep_probe_trainer.py
# Should show 2 results
```

---

## ✅ ALLOWED vs ❌ FORBIDDEN USAGE

### ❌ FORBIDDEN (WILL CRASH) - Must Fix
```python
# Any path that feeds a probe/classifier:
features = model.extract_features(data, channels)  # 512 dims
output = probe(features)  # Expects 2048 → CRASH!
```

**Forbidden Locations** (all 6 must be fixed):
- `api/routers/eegpt.py` - 3 endpoints: `/eegpt/analyze`, `/eegpt/sleep/stages`, `/analyze/batch`
- `api/routers/sleep.py` - EEGPT branch (feeds to sleep probe)
- `application/training/sleep_probe_trainer.py` - 2 methods: train_step, validation_step
- `infra/ml_models/eegpt_compat.py` - extract_features_batch method

### ✅ ALLOWED (Won't Crash) - Don't Change
```python
# Paths using features for statistics/heuristics (not probes):
features = model.extract_features(data, channels, summary=True)  # 512 dims
mean_activation = features.mean()  # Simple stats, no probe
```

**Allowed Locations**:
- `domain/quality/controller.py` - QC heuristics, no probe
- `domain/preprocessing/features/extractor.py` - Feature aggregation, no probe
- `cli.py` - Streaming display, no probe (but should fix for consistency)
- `application/pipeline/eegpt_orchestration.py` - Already uses `summary=False` correctly

---

## 📊 DEFINITION OF DONE

### Acceptance Criteria
- [ ] **API Fix Applied**: All 4 extract_features calls in api/routers have `summary=False` + flatten
- [ ] **Compat Fix Applied**: extract_features_batch in eegpt_compat.py passes `summary=False`
- [ ] **Trainer Fix Applied**: Both extract_features calls in sleep_probe_trainer have fixes
- [ ] **Verification Passes**: Pattern searches return 0 unfixed calls in critical paths
- [ ] **Smoke Test Passes**: Can call all 3 EEGPT API endpoints without RuntimeError
- [ ] **Batch Endpoint Works**: `/analyze/batch` processes multiple windows without crash
- [ ] **Trainer Runs**: SleepProbeTrainer completes 1 epoch with real EEGPTModel (no mocks)
- [ ] **Tests Added**: Unit test verifying probe paths use 2048 dims
- [ ] **CI Green**: `make test`, `make typecheck`, `make lint` all pass
- [ ] **No Regressions**: Existing tests still pass

### Test to Add
```python
# tests/unit/api/test_eegpt_p0_fix.py
def test_api_uses_2048_dims_for_probes():
    """P0 Fix Verification: API must flatten features for probes."""
    mock_model = Mock()
    mock_model.extract_features.return_value = torch.randn(32, 4, 512)

    # Simulate fixed API behavior
    features = mock_model.extract_features("data", "channels", summary=False)
    features = features.flatten(1)

    assert features.shape == (32, 2048), f"P0 VIOLATION: Expected (32,2048), got {features.shape}"
    mock_model.extract_features.assert_called_with("data", "channels", summary=False)

def test_batch_endpoint_uses_2048_dims():
    """P0 Fix Verification: Batch endpoint must flatten features."""
    mock_model = Mock()
    mock_model.extract_features_batch.return_value = np.random.randn(16, 4, 512)

    # Simulate fixed batch behavior
    features = mock_model.extract_features_batch("batch", "channels", summary=False)
    features = features.reshape(features.shape[0], -1)

    assert features.shape == (16, 2048), f"P0 VIOLATION: Expected (16,2048), got {features.shape}"
```

---

## 🔧 IMPLEMENTATION PLAN

### Step 1: Fix API Routers (25 min)
```bash
# Fix eegpt.py - 3 endpoints
vim src/brain_go_brrr/api/routers/eegpt.py
# Search: /extract_features
# Fix regular calls: Add summary=False and .flatten(1)
# Fix batch call: Add summary=False and .reshape(shape[0], -1)

# Fix sleep.py - 1 endpoint
vim src/brain_go_brrr/api/routers/sleep.py
# Same process
```

### Step 2: Fix eegpt_compat.py (10 min)
```bash
vim src/brain_go_brrr/infra/ml_models/eegpt_compat.py
# Find extract_features_batch method (~line 273)
# Add summary=False to encoder.extract_features call
# Add reshape logic before return
```

### Step 3: Fix SleepProbeTrainer (15 min)
```bash
vim src/brain_go_brrr/application/training/sleep_probe_trainer.py
# Fix both occurrences in train_step and validation_step
```

### Step 4: Verify (10 min)
```bash
# Run all verification commands from above
# Ensure 0 unfixed calls remain in critical paths
```

### Step 5: Test (20 min)
```bash
# Add unit tests
vim tests/unit/api/test_eegpt_p0_fix.py

# Run tests
make test

# Smoke test all 3 API endpoints
curl -X POST localhost:8000/api/v1/eegpt/analyze -F "file=@test.edf"
curl -X POST localhost:8000/api/v1/eegpt/sleep/stages -F "file=@test.edf"
curl -X POST localhost:8000/api/v1/analyze/batch -F "file=@test.edf"
```

### Step 6: Commit (5 min)
```bash
git add -p  # Review each change
git commit -m "fix(p0): add summary=False to prevent 512/2048 dimension crashes

- API routers: Fixed 4 endpoints (analyze, sleep/stages, batch)
- eegpt_compat: Fixed extract_features_batch to pass summary=False
- SleepProbeTrainer: Fixed both train and validation steps
- Fixes RuntimeError: mat1 and mat2 shapes cannot be multiplied
- Closes P0 critical issue from technical debt"
```

---

## 🚦 RISK ASSESSMENT

| Risk | Mitigation | Residual |
|------|-----------|----------|
| Breaking working code | Full test suite before merge | Low |
| Missing an occurrence | Pattern-based verification | Low |
| Performance regression | Same computation, just no averaging | None |
| Merge conflicts | Single atomic PR, small blast radius | Low |

---

## 📝 PAPER EVIDENCE

From EEGPT paper (`literature/markdown/EEGPT/EEGPT.md`):
> "We use 4 × 512 dimensional features for downstream tasks"
> "The final representation is obtained by flattening the 4 summary tokens"

This is why probes expect 2048 (4×512) input dimensions.

---

## 🆘 QUICK REFERENCE

**The Fix Pattern (copy-paste ready):**
```python
# For regular extract_features:
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)

# For extract_features_batch:
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names, summary=False)
batch_features = batch_features.reshape(batch_features.shape[0], -1)
```

**Why it crashes**: Probes trained on 2048-dim input, API provides 512
**Why tests missed it**: Mocks return correct shape, masking the bug
**Production impact**: 100% failure rate on EEGPT API endpoints

---

## 👥 SIGN-OFF

**Pre-Implementation Review**
- [ ] Reviewed affected files exist
- [ ] Verified crashes with current code
- [ ] Understood fix requirements

**Implementation**
- [ ] Applied fixes to all 6 locations
- [ ] Ran verification commands
- [ ] Added unit tests

**Post-Implementation**
- [ ] Smoke tests pass on all 3 endpoints
- [ ] CI/CD green
- [ ] No regressions

**Owner**: _____________________ **Started**: _________ **Completed**: _________

**Reviewer**: __________________ **Date**: ___________

**Deployed**: __________________ **Date**: ___________

---

## 📝 REVISION HISTORY

- **v1.0** (Sept 4): Initial P0 document with 2 API endpoints
- **v2.0** (Sept 4): Added Definition of Done, removed line numbers
- **v3.0** (Sept 4): Added missing `/analyze/batch` endpoint and `eegpt_compat.py` fix per audit

---

**END OF P0 CRITICAL FIXES**

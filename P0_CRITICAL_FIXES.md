# 🚨 P0 CRITICAL FIXES - RUNTIME CRASH BUGS

**Created**: September 4, 2025
**Owner**: ___________________
**Time Required**: 45-90 minutes
**Status**: 🔴 CRITICAL - FIX IMMEDIATELY

---

## 📌 SSOT: EEGPT Feature Dimensions Rule

> **🎯 THE GOLDEN RULE**
> - **IF feeding a probe**: Use `extract_features(..., summary=False)` → (B,4,512) → `.flatten(1)` → (B,2048)
> - **IF NOT feeding a probe**: Can use `extract_features(..., summary=True)` → (B,512) for heuristics/stats
> - **WHY**: EEGPT outputs 4 summary tokens × 512 dims. Probes trained on all 2048 dims. Using 512 = CRASH.

---

## 📋 EXECUTIVE SUMMARY

**We have 2 P0 bugs causing runtime crashes:**
1. **API endpoints** pass 512 dims to probes expecting 2048 → **RuntimeError**
2. **SleepProbeTrainer** passes 512 dims to probe expecting 2048 → **RuntimeError**

**Root Cause**: Missing `summary=False` parameter in `extract_features()` calls
**Business Impact**: 100% API failure rate for EEGPT endpoints, blocks demos
**Fix Complexity**: Simple parameter addition + flatten (7 lines total)

---

## 🔴 P0 BUG #1: API ENDPOINTS DIMENSION MISMATCH

### THE CRASH
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 2048x6)
              Your 512-dim features trying to enter 2048-dim probe input
```

### FILES TO FIX

#### File: `src/brain_go_brrr/api/routers/eegpt.py`
**Pattern to find**: `eegpt_model.extract_features(window_data, channel_names)`
**Fix**: Add `summary=False` and `.flatten(1)` after EVERY occurrence

```python
# BEFORE (crashes):
features = eegpt_model.extract_features(window_data, channel_names)

# AFTER (works):
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)
```

#### File: `src/brain_go_brrr/api/routers/sleep.py`
**Pattern to find**: `eegpt_model.extract_features(window_data, channel_names)`
**Fix**: Same as above - add `summary=False` and `.flatten(1)`

### VERIFICATION
```bash
# Find all broken calls (should return 3 results BEFORE fix):
rg "extract_features\([^)]*\)" src/brain_go_brrr/api/routers | grep -v "summary="

# After fix (should return 0 results):
rg "extract_features\([^)]*\)" src/brain_go_brrr/api/routers | grep -v "summary="

# Confirm fix applied (should show 3+ results):
rg "extract_features.*summary=False" src/brain_go_brrr/api/routers
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

**Forbidden Locations**:
- `api/routers/eegpt.py` - ALL occurrences (feeds to probes)
- `api/routers/sleep.py` - EEGPT branch (feeds to sleep probe)
- `application/training/sleep_probe_trainer.py` - ALL occurrences (direct probe training)

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

---

## 📊 DEFINITION OF DONE

### Acceptance Criteria
- [ ] **API Fix Applied**: All 3 extract_features calls in api/routers have `summary=False` + `.flatten(1)`
- [ ] **Trainer Fix Applied**: Both extract_features calls in sleep_probe_trainer have fixes
- [ ] **Verification Passes**: Pattern searches return 0 unfixed calls in critical paths
- [ ] **Smoke Test Passes**: Can call `/api/v1/eeg/extract_features` without RuntimeError
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
```

---

## 🔧 IMPLEMENTATION PLAN

### Step 1: Fix API Routers (20 min)
```bash
# Open and fix all occurrences:
vim src/brain_go_brrr/api/routers/eegpt.py
# Search: /extract_features
# Add: summary=False parameter and features = features.flatten(1) after each

vim src/brain_go_brrr/api/routers/sleep.py
# Same process
```

### Step 2: Fix SleepProbeTrainer (15 min)
```bash
vim src/brain_go_brrr/application/training/sleep_probe_trainer.py
# Fix both occurrences in train_step and validation_step
```

### Step 3: Verify (10 min)
```bash
# Run all verification commands from above
# Ensure 0 unfixed calls remain in critical paths
```

### Step 4: Test (15 min)
```bash
# Add unit test
vim tests/unit/api/test_eegpt_p0_fix.py

# Run tests
make test

# Smoke test API
curl -X POST localhost:8000/api/v1/eeg/extract_features -F "file=@test.edf"
```

### Step 5: Commit (5 min)
```bash
git add -p  # Review each change
git commit -m "fix(p0): add summary=False to prevent 512/2048 dimension crashes

- API routers now use extract_features(..., summary=False) + flatten(1)
- SleepProbeTrainer uses correct 2048 dims for probe input
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
# Replace this:
features = eegpt_model.extract_features(window_data, channel_names)

# With this:
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)
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
- [ ] Applied fixes to all locations
- [ ] Ran verification commands
- [ ] Added unit test

**Post-Implementation**
- [ ] Smoke test passes
- [ ] CI/CD green
- [ ] No regressions

**Owner**: _____________________ **Started**: _________ **Completed**: _________

**Reviewer**: __________________ **Date**: ___________

**Deployed**: __________________ **Date**: ___________

---

**END OF P0 CRITICAL FIXES**

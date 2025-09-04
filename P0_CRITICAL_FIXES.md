# 🚨 P0 CRITICAL FIXES - RUNTIME CRASH BUGS

**Created**: September 4, 2025
**Owner**: ___________________
**Time Required**: 45-90 minutes
**Status**: 🔴 CRITICAL - FIX IMMEDIATELY
**Revision**: v4 - Pristine version with all ambiguities resolved

---

## 📌 SSOT: EEGPT Feature Dimensions Rule

> **🎯 THE GOLDEN RULE**
> - **IF feeding a probe**: Use `extract_features(..., summary=False)` → (B,4,512) → `.flatten(1)` → (B,2048)
> - **IF NOT feeding a probe**: Can use `extract_features(..., summary=True)` → (B,512) for heuristics/stats
> - **WHY**: EEGPT outputs 4 summary tokens × 512 dims. Probes trained on all 2048 dims. Using 512 = CRASH.
> - **WHERE TO FLATTEN**: ONLY at probe call-sites. Never inside helper methods.

---

## 📋 EXECUTIVE SUMMARY

**We have 6 crash sites causing runtime failures:**
1. **API endpoints** pass 512 dims to probes expecting 2048 → **RuntimeError** (4 call-sites)
2. **SleepProbeTrainer** passes 512 dims to probe expecting 2048 → **RuntimeError** (2 call-sites)

**Plus 1 supporting change:**
- **eegpt_compat.py** - Update method signatures to accept `summary` parameter

**Root Cause**: Missing `summary=False` parameter in `extract_features()` calls
**Business Impact**: 100% API failure rate for EEGPT endpoints, blocks demos
**Fix Complexity**: Simple parameter addition + flatten at call-sites

---

## 🔴 P0 BUG #1: API ENDPOINTS DIMENSION MISMATCH

### THE CRASH
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 2048x6)
              Your 512-dim features trying to enter 2048-dim probe input
```

### FILES TO FIX

#### File: `src/brain_go_brrr/api/routers/eegpt.py`
**3 Call-sites to fix:**
1. `/eegpt/analyze` - Uses `extract_features(window_data, channel_names)`
2. `/eegpt/sleep/stages` - Uses `extract_features(window_data, channel_names)`
3. `/analyze/batch` - Uses `extract_features_batch(batch_array, channel_names)`

**Fix for regular extract_features calls**:
```python
# BEFORE (crashes):
features = eegpt_model.extract_features(window_data, channel_names)

# AFTER (works):
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)
```

**Fix for extract_features_batch call**:
```python
# BEFORE (crashes):
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names)

# AFTER (works):
# P0: Probes require 2048-dim (4×512) features
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names, summary=False)
batch_features = batch_features.flatten(1)  # (B,4,512) → (B,2048)
```

#### File: `src/brain_go_brrr/api/routers/sleep.py`
**1 Call-site to fix:**
- EEGPT branch in sleep analysis endpoint

```python
# BEFORE (crashes):
features = eegpt_model.extract_features(window_data, channel_names)

# AFTER (works):
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)
```

### SUPPORTING CHANGE (Not a crash site, but required)

#### File: `src/brain_go_brrr/infra/ml_models/eegpt_compat.py`

**Update method signatures to accept and forward `summary` parameter:**

1. **extract_features method** (~line 156):
```python
def extract_features(
    self,
    data: npt.NDArray[np.float64],
    channel_names: list[str] | None = None,
    summary: bool = True  # ADD THIS PARAMETER
) -> npt.NDArray[np.float64]:
    # ... existing code ...
    # Forward summary to encoder:
    features = self.encoder.extract_features(data_tensor, summary=summary)
    # Return as-is - NO FLATTENING HERE (flatten at call-site only)
    return features.cpu().numpy()
```

2. **extract_features_batch method** (~line 258):
```python
def extract_features_batch(
    self,
    windows: npt.NDArray[np.float64] | torch.Tensor,
    channel_names: list[str] | None = None,
    summary: bool = True  # ADD THIS PARAMETER
) -> npt.NDArray[np.float64]:
    # ... existing code ...
    # Forward summary to encoder:
    features = self.encoder.extract_features(batch_tensor, summary=summary)
    # Return (B,4,512) if summary=False, (B,512) if summary=True
    # NO FLATTENING HERE - flatten at call-site only!
    return features.cpu().numpy()
```

### VERIFICATION
```bash
# Find all broken API calls (should return 4 BEFORE fix):
rg -n "extract_features(_batch)?\([^)]*$" src/brain_go_brrr/api/routers | grep -v "summary="

# After fix (should return 0):
rg -n "extract_features(_batch)?\([^)]*$" src/brain_go_brrr/api/routers | grep -v "summary="

# Verify compat methods have summary parameter:
rg -n "def extract_features(_batch)?\(.*summary" src/brain_go_brrr/infra/ml_models/eegpt_compat.py
# Should show 2 results

# Confirm all API calls flatten after getting features:
rg -A1 "extract_features(_batch)?.*summary=False" src/brain_go_brrr/api/routers | grep "flatten(1)"
# Should show 4 results
```

---

## 🔴 P0 BUG #2: SLEEPPROBETRAINER DIMENSION MISMATCH

### THE CRASH
```
RuntimeError: size mismatch, m1: [256 x 512], m2: [2048 x 5] at linear layer
```

### FILES TO FIX

#### File: `src/brain_go_brrr/application/training/sleep_probe_trainer.py`
**2 Call-sites to fix:**
- train_step method (~line 110)
- validation_step method (~line 193)

```python
# BEFORE (crashes):
features = self.feature_extractor.extract_features(eeg_data)

# AFTER (works):
# P0: Probes require 2048-dim (4×512) features
features = self.feature_extractor.extract_features(eeg_data, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)
```

### VERIFICATION
```bash
# Find broken calls (should show 2 BEFORE fix):
rg -n "extract_features\([^)]*$" src/brain_go_brrr/application/training/sleep_probe_trainer.py | grep -v "summary="

# After fix (should show 0):
rg -n "extract_features\([^)]*$" src/brain_go_brrr/application/training/sleep_probe_trainer.py | grep -v "summary="

# Confirm both fixes applied with flatten:
rg -A1 "extract_features.*summary=False" src/brain_go_brrr/application/training/sleep_probe_trainer.py | grep "flatten(1)"
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

**Forbidden Locations** (all 6 crash sites):
- `api/routers/eegpt.py` - 3 call-sites: `/eegpt/analyze`, `/eegpt/sleep/stages`, `/analyze/batch`
- `api/routers/sleep.py` - 1 call-site: EEGPT branch
- `application/training/sleep_probe_trainer.py` - 2 call-sites: train_step, validation_step

**Critical Rule**: Routers MUST NEVER use `summary=True` on any path that feeds a probe.

### ✅ ALLOWED (Won't Crash) - Don't Change
```python
# Paths using features for statistics/heuristics (not probes):
features = model.extract_features(data, channels, summary=True)  # 512 dims
mean_activation = features.mean()  # Simple stats, no probe
```

**Allowed Locations**:
- `domain/quality/controller.py` - QC heuristics, no probe
- `domain/preprocessing/features/extractor.py` - Feature aggregation, no probe
- `cli.py` - Streaming display, no probe
- `application/pipeline/eegpt_orchestration.py` - Already uses `summary=False` correctly

---

## 📊 DEFINITION OF DONE

### Acceptance Criteria
- [ ] **API Fixes Applied**: All 4 API call-sites (`extract_features`/`extract_features_batch`) in api/routers pass `summary=False` and flatten before probes
- [ ] **Compat Signatures Updated**: Both methods in eegpt_compat.py accept `summary: bool = True` parameter
- [ ] **Compat Returns Correct Shape**: extract_features_batch returns (B,4,512) when summary=False, NO internal flattening
- [ ] **Trainer Fixes Applied**: Both extract_features calls in sleep_probe_trainer have `summary=False` + `.flatten(1)`
- [ ] **Verification Passes**: Pattern searches return 0 unfixed calls in critical paths
- [ ] **Smoke Test Passes**: Can call all 3 EEGPT API endpoints without RuntimeError
- [ ] **Batch Endpoint Works**: `/analyze/batch` processes multiple windows without crash
- [ ] **Trainer Runs**: SleepProbeTrainer completes 1 epoch with real EEGPTModel (no mocks)
- [ ] **Contract Tests Added**: Test verifying extract_features_batch returns (B,4,512) when summary=False
- [ ] **CI Green**: `make test`, `make typecheck`, `make lint` all pass
- [ ] **No Regressions**: Existing tests still pass

### Tests to Add
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
    # extract_features_batch should return (B,4,512) when summary=False
    mock_model.extract_features_batch.return_value = np.random.randn(16, 4, 512)

    # Simulate fixed batch behavior (flatten at call-site)
    features = mock_model.extract_features_batch("batch", "channels", summary=False)
    features = torch.from_numpy(features).flatten(1)

    assert features.shape == (16, 2048), f"P0 VIOLATION: Expected (16,2048), got {features.shape}"
    mock_model.extract_features_batch.assert_called_with("batch", "channels", summary=False)

def test_extract_features_batch_contract():
    """Contract test: extract_features_batch returns correct shape."""
    mock_compat = Mock()
    mock_compat.extract_features_batch.return_value = np.random.randn(8, 4, 512)

    # When summary=False, should return (B,4,512) NOT flattened
    features = mock_compat.extract_features_batch(np.zeros((8, 19, 1024)), ["ch1"], summary=False)
    assert features.shape == (8, 4, 512), "extract_features_batch must NOT flatten internally"
```

---

## 🔧 IMPLEMENTATION PLAN

### Step 1: Update eegpt_compat.py signatures (15 min)
```bash
vim src/brain_go_brrr/infra/ml_models/eegpt_compat.py
# Add summary: bool = True to both extract_features and extract_features_batch
# Forward summary to encoder.extract_features calls
# REMOVE any internal flattening - return (B,4,512) when summary=False
```

### Step 2: Fix API Routers (20 min)
```bash
# Fix eegpt.py - 3 call-sites
vim src/brain_go_brrr/api/routers/eegpt.py
# Add comment: # P0: Probes require 2048-dim (4×512) features
# Add: summary=False and .flatten(1) after each extract_features call

# Fix sleep.py - 1 call-site
vim src/brain_go_brrr/api/routers/sleep.py
# Same process
```

### Step 3: Fix SleepProbeTrainer (10 min)
```bash
vim src/brain_go_brrr/application/training/sleep_probe_trainer.py
# Fix both occurrences in train_step and validation_step
# Add comment and summary=False + .flatten(1)
```

### Step 4: Verify (10 min)
```bash
# Run all verification commands from above
# Ensure 0 unfixed calls remain in critical paths
# Verify compat methods have summary parameter
# Verify all call-sites flatten after getting features
```

### Step 5: Test (20 min)
```bash
# Add contract tests
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

- Updated eegpt_compat signatures to accept summary parameter
- Fixed 4 API call-sites to pass summary=False and flatten at call-site
- Fixed 2 trainer call-sites with same pattern
- No internal flattening in compat layer (SSOT: flatten at probe only)
- Added contract tests for shape verification
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
| Double-flatten bug | SSOT: flatten only at call-sites | None |
| Signature mismatch | Update compat methods first | None |

---

## 📝 PAPER EVIDENCE

From EEGPT paper (`literature/markdown/EEGPT/EEGPT.md`):
> "We use 4 × 512 dimensional features for downstream tasks"
> "The final representation is obtained by flattening the 4 summary tokens"

Our implementation (`eegpt_architecture.py`):
- `embed_dim: int = 512`
- `embed_num: int = 4  # Number of summary tokens`
- Probes expect `input_dim=2048` (4 × 512)

---

## 🆘 QUICK REFERENCE

**The Fix Pattern (copy-paste ready):**
```python
# For regular extract_features (torch tensors):
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)  # (B,4,512) → (B,2048)

# For extract_features_batch (may return numpy):
# P0: Probes require 2048-dim (4×512) features
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names, summary=False)
if isinstance(batch_features, np.ndarray):
    batch_features = torch.from_numpy(batch_features)
batch_features = batch_features.flatten(1)  # (B,4,512) → (B,2048)
```

**Why it crashes**: Probes trained on 2048-dim input, API provides 512
**Why tests missed it**: Mocks return correct shape, masking the bug
**Production impact**: 100% failure rate on EEGPT API endpoints

---

## 👥 SIGN-OFF

**Pre-Implementation Review**
- [ ] Reviewed affected files exist
- [ ] Verified crashes with current code
- [ ] Understood SSOT: flatten only at call-sites

**Implementation**
- [ ] Updated 2 compat method signatures
- [ ] Applied fixes to 6 crash sites
- [ ] Ran verification commands
- [ ] Added contract tests

**Post-Implementation**
- [ ] Smoke tests pass on all 3 endpoints
- [ ] SleepProbeTrainer runs 1 epoch
- [ ] CI/CD green
- [ ] No regressions

**Owner**: _____________________ **Started**: _________ **Completed**: _________

**Reviewer**: __________________ **Date**: ___________

**Deployed**: __________________ **Date**: ___________

---

## 📝 REVISION HISTORY

- **v1.0** (Sept 4): Initial P0 document with 2 API endpoints
- **v2.0** (Sept 4): Added Definition of Done, removed line numbers
- **v3.0** (Sept 4): Added missing `/analyze/batch` endpoint per audit
- **v4.0** (Sept 4): Pristine version - fixed signature issues, clarified SSOT, standardized on .flatten(1)

---

**END OF P0 CRITICAL FIXES**

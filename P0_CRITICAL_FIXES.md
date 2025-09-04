# 🚨 P0 CRITICAL FIXES - RUNTIME CRASH BUGS

**Created**: September 4, 2025  
**Owner**: ___________________  
**Time Required**: 90 minutes (30 min tests, 30 min fixes, 30 min refactor)  
**Status**: 🔴 CRITICAL - FIX IMMEDIATELY  
**Revision**: v5 - TDD methodology integrated (single source of truth)  
**Approach**: RED → GREEN → REFACTOR (Test-Driven Development)

---

## 📌 SSOT: EEGPT Feature Dimensions Rule

> **🎯 THE GOLDEN RULE**  
> - **IF feeding a probe**: Use `extract_features(..., summary=False)` → (B,4,512) → `.flatten(1)` → (B,2048)
> - **IF NOT feeding a probe**: Can use `extract_features(..., summary=True)` → (B,512) for heuristics/stats
> - **WHY**: EEGPT outputs 4 summary tokens × 512 dims. Probes trained on all 2048 dims. Using 512 = CRASH.
> - **WHERE TO FLATTEN**: ONLY at probe call-sites. Never inside helper methods.
> - **THE INVARIANT**: Probes MUST receive 2048-dim features (4×512 flattened)

---

## 📋 EXECUTIVE SUMMARY

**We have 6 crash sites causing runtime failures:**
1. **API endpoints** pass 512 dims to probes expecting 2048 → **RuntimeError** (4 call-sites)
2. **SleepProbeTrainer** passes 512 dims to probe expecting 2048 → **RuntimeError** (2 call-sites)

**Plus 1 supporting change:**
- **eegpt_compat.py** - Update method signatures to accept `summary` parameter

**Root Cause**: Missing `summary=False` parameter in `extract_features()` calls  
**Business Impact**: 100% API failure rate for EEGPT endpoints, blocks demos  
**Fix Strategy**: TDD - Write failing tests FIRST, then minimal fixes, then refactor  

---

## 🔴 PHASE 1: RED - Write Failing Tests First (30 min)

### Why Start with Tests?
Per Clean Code/TDD principles, we write tests that PROVE the bug exists before fixing it. These tests MUST fail on current code.

### Step 1.1: Create Test Files
```bash
# Create P0-specific test files
touch tests/unit/api/routers/test_eegpt_p0_dimensions.py
touch tests/unit/application/training/test_sleep_probe_p0_dimensions.py
touch tests/unit/infra/ml_models/test_eegpt_compat_p0_contract.py
touch tests/unit/utils/test_probe_utils.py
```

### Step 1.2: Write Boundary Tests

#### Test 1: API Endpoints Must Flatten to 2048
```python
# tests/unit/api/routers/test_eegpt_p0_dimensions.py
"""P0: Test that API endpoints flatten features to 2048 dims before probes."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, ANY

class TestP0APIDimensions:
    """Prove API endpoints crash with 512 dims, work with 2048."""
    
    def test_analyze_endpoint_flattens_to_2048(self):
        """P0: /eegpt/analyze must flatten (B,4,512) to (B,2048)."""
        with patch('brain_go_brrr.api.routers.eegpt.get_eegpt_model') as mock_get:
            # Model returns (B,4,512) when summary=False
            mock_model = Mock()
            mock_model.extract_features.return_value = torch.randn(1, 4, 512)
            mock_get.return_value = mock_model
            
            # Probe expects 2048 dims
            mock_probe = Mock()
            mock_probe.predict_proba.side_effect = lambda x: (
                torch.randn(x.shape[0], 2) if x.shape[-1] == 2048 
                else pytest.fail(f"Probe got {x.shape[-1]} dims, expected 2048")
            )
            
            # THIS TEST MUST FAIL ON CURRENT CODE
            # After fix, it should pass
            
    def test_batch_endpoint_flattens_to_2048(self):
        """P0: /analyze/batch must flatten (B,4,512) to (B,2048)."""
        # Similar test for batch endpoint
        
    def test_sleep_stages_endpoint_flattens_to_2048(self):
        """P0: /eegpt/sleep/stages must flatten to 2048."""
        # Similar test for sleep endpoint
```

#### Test 2: Trainer Must Flatten to 2048
```python
# tests/unit/application/training/test_sleep_probe_p0_dimensions.py
"""P0: Test that SleepProbeTrainer flattens to 2048 before probe."""

class TestP0TrainerDimensions:
    def test_train_step_flattens_to_2048(self):
        """P0: train_step must flatten (B,4,512) to (B,2048)."""
        # Mock extractor returns (B,4,512) when summary=False
        # Probe validates 2048 dims
        # THIS TEST MUST FAIL ON CURRENT CODE
        
    def test_validation_step_flattens_to_2048(self):
        """P0: validation_step must flatten to 2048."""
        # Similar test
```

#### Test 3: Compat Contract (No Internal Flattening)
```python
# tests/unit/infra/ml_models/test_eegpt_compat_p0_contract.py
"""P0: Test eegpt_compat contract - NO internal flattening."""

class TestP0CompatContract:
    def test_extract_features_batch_returns_unflattened(self):
        """P0: extract_features_batch must return (B,4,512) when summary=False."""
        # Must NOT flatten internally (SSOT: flatten at call-sites only)
        # THIS TEST MUST FAIL ON CURRENT CODE
        
    def test_extract_features_forwards_summary_parameter(self):
        """P0: extract_features must forward summary parameter to encoder."""
        # Verify summary is passed through
```

### Step 1.3: Run Tests to Confirm They FAIL
```bash
# These MUST all fail on current code - that proves the bug exists
pytest tests/unit/api/routers/test_eegpt_p0_dimensions.py -xvs
pytest tests/unit/application/training/test_sleep_probe_p0_dimensions.py -xvs
pytest tests/unit/infra/ml_models/test_eegpt_compat_p0_contract.py -xvs

# Expected: All tests FAIL with dimension mismatches
```

---

## 🟢 PHASE 2: GREEN - Minimal Fixes to Pass Tests (30 min)

Now we make the SMALLEST changes to make tests pass. No extras, no refactoring yet.

### FILES TO FIX

#### Fix 1: `src/brain_go_brrr/infra/ml_models/eegpt_compat.py` (Supporting Change)

**Update method signatures to accept and forward `summary` parameter:**

```python
def extract_features(
    self,
    data: npt.NDArray[np.float64],
    channel_names: list[str] | None = None,
    summary: bool = True  # ADD THIS PARAMETER
) -> npt.NDArray[np.float64]:
    # Forward summary to encoder:
    features = self.encoder.extract_features(data_tensor, summary=summary)
    # Return as-is - NO FLATTENING HERE (flatten at call-site only)
    return features.cpu().numpy()

def extract_features_batch(
    self,
    windows: npt.NDArray[np.float64] | torch.Tensor,
    channel_names: list[str] | None = None,
    summary: bool = True  # ADD THIS PARAMETER
) -> npt.NDArray[np.float64]:
    # Forward summary to encoder:
    features = self.encoder.extract_features(batch_tensor, summary=summary)
    # NO FLATTENING HERE - return (B,4,512) if summary=False
    return features.cpu().numpy()
```

#### Fix 2: `src/brain_go_brrr/api/routers/eegpt.py` (3 Call-sites)

```python
# Fix /eegpt/analyze endpoint:
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(window_data, channel_names, summary=False)  # numpy
features_tensor = torch.as_tensor(features, dtype=torch.float32)  # Convert numpy→torch
features_tensor = features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass features_tensor to probe

# Fix /eegpt/sleep/stages endpoint:
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(window_data, channel_names, summary=False)  # numpy
features_tensor = torch.as_tensor(features, dtype=torch.float32)
features_tensor = features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass features_tensor to probe

# Fix /analyze/batch endpoint:
# P0: Probes require 2048-dim (4×512) features
batch_features = eegpt_model.extract_features_batch(batch_array, channel_names, summary=False)
batch_features_tensor = torch.as_tensor(batch_features, dtype=torch.float32)
batch_features_tensor = batch_features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass batch_features_tensor to probe
```

#### Fix 3: `src/brain_go_brrr/api/routers/sleep.py` (1 Call-site)

```python
# EEGPT branch in sleep analysis:
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(window_data, channel_names, summary=False)  # numpy
features_tensor = torch.as_tensor(features, dtype=torch.float32)
features_tensor = features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass features_tensor to probe
```

#### Fix 4: `src/brain_go_brrr/application/training/sleep_probe_trainer.py` (2 Call-sites)

```python
# In train_step:
# P0: Probes require 2048-dim (4×512) features
features = self.feature_extractor.extract_features(eeg_data, summary=False)  # numpy
features_tensor = torch.as_tensor(features, dtype=torch.float32)
features_tensor = features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass features_tensor to probe

# In evaluate_probe method:
# P0: Probes require 2048-dim (4×512) features
features = self.feature_extractor.extract_features(eeg_data, summary=False)  # numpy
features_tensor = torch.as_tensor(features, dtype=torch.float32)
features_tensor = features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass features_tensor to probe
```

### Run Tests Again - They Should PASS
```bash
# All tests should now pass
pytest tests/unit/api/routers/test_eegpt_p0_dimensions.py -xvs
pytest tests/unit/application/training/test_sleep_probe_p0_dimensions.py -xvs
pytest tests/unit/infra/ml_models/test_eegpt_compat_p0_contract.py -xvs
```

---

## 🔧 PHASE 3: REFACTOR - DRY with Adapter Pattern (30 min)

Now that tests pass, we refactor to remove duplication while keeping tests green.

### Step 3.1: Create Adapter Helper
```python
# src/brain_go_brrr/utils/probe_utils.py
"""Utilities for probe preparation following EEGPT contract."""
import torch
import numpy as np
from typing import Union

def prepare_probe_features(
    features: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """
    Adapter: Convert EEGPT features to probe-ready format.
    
    Accepts: (B,4,512) from extract_features with summary=False
    Returns: (B,2048) ready for probe input
    
    This is the SINGLE SOURCE OF TRUTH for probe preparation.
    """
    # Convert numpy to torch if needed (use as_tensor for zero-copy when possible)
    if isinstance(features, np.ndarray):
        features = torch.as_tensor(features, dtype=torch.float32)
    
    # Flatten if needed (idempotent - safe to call multiple times)
    if features.ndim == 3 and features.shape[1] == 4 and features.shape[2] == 512:
        features = features.flatten(1)  # (B,4,512) → (B,2048)
    
    # Safety assertion
    assert_probe_ready(features)
    return features

def assert_probe_ready(features: torch.Tensor) -> None:
    """Safety net: Ensure features are probe-ready."""
    if features.shape[-1] != 2048:
        raise AssertionError(
            f"Probe expects 2048 dims, got {tuple(features.shape)}. "
            f"Did you forget summary=False or prepare_probe_features()?"
        )
```

### Step 3.2: Write Test for Adapter
```python
# tests/unit/utils/test_probe_utils.py
"""Test probe preparation utilities."""

def test_prepare_probe_features_flattens():
    """Adapter correctly flattens (B,4,512) to (B,2048)."""
    from brain_go_brrr.utils.probe_utils import prepare_probe_features
    
    features = torch.randn(32, 4, 512)
    prepared = prepare_probe_features(features)
    assert prepared.shape == (32, 2048)

def test_assert_probe_ready_catches_wrong_dims():
    """Safety net catches dimension errors."""
    from brain_go_brrr.utils.probe_utils import assert_probe_ready
    
    good = torch.randn(32, 2048)
    assert_probe_ready(good)  # No error
    
    bad = torch.randn(32, 512)
    with pytest.raises(AssertionError, match="Probe expects 2048 dims"):
        assert_probe_ready(bad)
```

### Step 3.3: Refactor All Call-sites to Use Adapter
```python
# All endpoints and trainer now use:
from brain_go_brrr.utils.probe_utils import prepare_probe_features

# ❌ WRONG - numpy doesn't support .flatten(1):
features = extract_features(..., summary=False)  # numpy (B,4,512)
features = features.flatten(1)  # TypeError!

# ✅ CORRECT - Use adapter (handles numpy→torch):
features = extract_features(..., summary=False)  # numpy (B,4,512)
features = prepare_probe_features(features)  # torch (B,2048)
```

### Step 3.4: Add End-to-End Integration Test
```python
# tests/integration/test_p0_end_to_end.py
"""End-to-end test that would have caught the P0 bug."""

def test_full_api_to_probe_path():
    """Integration: API endpoint → EEGPT → probe with correct dims."""
    # Use real EDF file, verify no dimension errors
    pass
```

---

## ✅ VERIFICATION & TESTING

### Pattern-Based Verification Commands
```bash
# Find all broken extract_features calls (single and batch) missing summary:
rg -n "extract_features(_batch)?\(" src/brain_go_brrr/api/routers | rg -v "summary="

# Check extract_features_batch calls specifically:
rg -n "extract_features_batch\([^)]*$" src/brain_go_brrr | rg -v "summary="

# Check trainer calls missing summary:
rg -n "extract_features\(" src/brain_go_brrr/application/training/sleep_probe_trainer.py | rg -v "summary="

# Verify compat methods have summary parameter:
rg -n "def extract_features(_batch)?\(.*summary" src/brain_go_brrr/infra/ml_models/eegpt_compat.py

# Confirm all call-sites use prepare_probe_features (after refactor):
rg "prepare_probe_features" src/brain_go_brrr/api/routers src/brain_go_brrr/application/training

# Run P0-specific tests:
pytest tests/unit -k "p0" -xvs

# Run full test suite:
make test

# Check coverage:
make coverage
```

---

## ❌ ALLOWED vs ❌ FORBIDDEN USAGE

### ❌ FORBIDDEN (WILL CRASH) - Must Fix
```python
# Any path that feeds a probe/classifier:
features = model.extract_features(data, channels)  # 512 dims
output = probe(features)  # Expects 2048 → CRASH!
```

**Forbidden Locations** (all 6 crash sites):
- `api/routers/eegpt.py` - 3 call-sites: `/eegpt/analyze`, `/eegpt/sleep/stages`, `/analyze/batch`
- `api/routers/sleep.py` - 1 call-site: EEGPT branch
- `application/training/sleep_probe_trainer.py` - 2 call-sites: train_step, evaluate_probe

**Critical Rule**: Routers MUST NEVER use `summary=True` on any path that feeds a probe.

### ✅ ALLOWED (Won't Crash) - Don't Change
```python
# Paths using features for statistics/heuristics (not probes):
features = model.extract_features(data, channels, summary=True)  # 512 dims
mean_activation = features.mean()  # Simple stats, no probe
```

---

## 📊 DEFINITION OF DONE

### TDD Checklist
- [ ] **Red Phase**: Write failing tests for all 6 crash sites
- [ ] **Red Phase**: Verify all tests FAIL (proves bug exists)
- [ ] **Green Phase**: Update compat signatures to accept `summary`
- [ ] **Green Phase**: Fix 4 API call-sites with `summary=False` + flatten
- [ ] **Green Phase**: Fix 2 trainer call-sites with `summary=False` + flatten
- [ ] **Green Phase**: All P0 tests PASS
- [ ] **Refactor Phase**: Create `prepare_probe_features` adapter
- [ ] **Refactor Phase**: Replace all flatten calls with adapter
- [ ] **Refactor Phase**: Add safety net `assert_probe_ready`
- [ ] **Integration**: End-to-end test passes

### Acceptance Criteria
- [ ] Pattern searches return 0 unfixed calls in critical paths
- [ ] Can call all 3 EEGPT API endpoints without RuntimeError
- [ ] SleepProbeTrainer completes 1 epoch with real EEGPTModel
- [ ] CI/CD green: `make test`, `make typecheck`, `make lint`
- [ ] No regressions in existing tests

---

## 🎯 COMMIT STRATEGY (Small, Atomic, TDD)

```bash
# Commit 1: Tests that prove the bug
git add tests/unit/api/routers/test_eegpt_p0_dimensions.py
git add tests/unit/application/training/test_sleep_probe_p0_dimensions.py
git add tests/unit/infra/ml_models/test_eegpt_compat_p0_contract.py
git commit -m "test(p0): add failing tests for 2048-dim probe contract

- API endpoints must flatten to 2048 before probes
- Trainer must flatten to 2048 before probes  
- Compat must NOT flatten internally (SSOT)
- Tests currently fail, proving P0 bug exists"

# Commit 2: Minimal fix to make tests pass
git add src/brain_go_brrr/infra/ml_models/eegpt_compat.py
git add src/brain_go_brrr/api/routers/eegpt.py
git add src/brain_go_brrr/api/routers/sleep.py
git add src/brain_go_brrr/application/training/sleep_probe_trainer.py
git commit -m "fix(p0): enforce 2048-dim probe inputs at call-sites

- Updated compat to accept/forward summary parameter
- Fixed 4 API call-sites to use summary=False + flatten
- Fixed 2 trainer call-sites with same pattern
- All P0 tests now pass"

# Commit 3: Refactor for DRY
git add src/brain_go_brrr/utils/probe_utils.py
git add tests/unit/utils/test_probe_utils.py
git add -p  # Stage refactored endpoints/trainer
git commit -m "refactor: introduce prepare_probe_features adapter

- Single source of truth for probe preparation
- Removes duplicate flatten logic
- Adds safety assertion for 2048 dims
- Follows GoF Adapter pattern"
```

---

## 🚦 RISK ASSESSMENT

| Risk | Mitigation | Residual |
|------|-----------|----------|
| Tests don't catch real bug | Use actual tensor shapes | Low |
| Fix breaks existing code | Full test suite after each change | Low |
| Performance regression | Flatten is O(1) view operation | None |
| Double-flatten bug | SSOT: flatten only at call-sites | None |
| Future violations | assert_probe_ready guard | None |

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

## 👥 SIGN-OFF

**Pre-Implementation Review**
- [ ] Reviewed TDD approach (red-green-refactor)
- [ ] Verified tests will prove bug exists
- [ ] Understood SSOT: flatten only at call-sites

**Red Phase Complete**
- [ ] All P0 tests written
- [ ] All tests FAIL on current code
- [ ] Bug proven to exist

**Green Phase Complete**  
- [ ] Minimal fixes applied
- [ ] All P0 tests PASS
- [ ] No extra changes made

**Refactor Phase Complete**
- [ ] Adapter pattern implemented
- [ ] Duplicate code removed
- [ ] Safety net in place

**Post-Implementation**
- [ ] Integration test passes
- [ ] Full test suite green
- [ ] No regressions

**Owner**: _____________________ **Started**: _________ **Completed**: _________

**Reviewer**: __________________ **Date**: ___________

**Deployed**: __________________ **Date**: ___________

---

## 📝 REVISION HISTORY

- **v1.0** (Sept 4): Initial P0 document with 2 API endpoints
- **v2.0** (Sept 4): Added Definition of Done, removed line numbers  
- **v3.0** (Sept 4): Added missing `/analyze/batch` endpoint per audit
- **v4.0** (Sept 4): Fixed signature issues, clarified SSOT, standardized on .flatten(1)
- **v5.0** (Sept 4): Integrated TDD methodology - single source of truth document
- **v6.0** (Sept 4): Fixed critical numpy→torch conversion bug in all examples:
  - numpy arrays don't support `.flatten(1)` - must convert to torch first
  - Changed "validation_step" to correct "evaluate_probe" method name
  - Added expanded verification patterns for extract_features_batch
  - Made shape contract explicit: numpy in, torch out

---

**END OF P0 CRITICAL FIXES WITH TDD IMPLEMENTATION**
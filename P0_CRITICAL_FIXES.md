# 🚨 P0 CRITICAL FIXES - RUNTIME CRASH BUGS

**Created**: September 4, 2025  
**Owner**: ___________________  
**Time Required**: 90 minutes (30 min tests, 30 min fixes, 30 min refactor)  
**Status**: 🔴 CRITICAL - FIX IMMEDIATELY  
**Revision**: v8.0 - TDD + Adapter boundary (final perfect version)  
**Approach**: RED → GREEN → REFACTOR (Test-Driven Development)

---

## 📌 SSOT: EEGPT Feature Dimensions Rule

> **🎯 THE GOLDEN RULE**  
> - **IF feeding a probe**: Use `extract_features(..., summary=False)` → numpy (B,4,512) → torch → `.flatten(1)` → (B,2048)
> - **IF NOT feeding a probe**: Can use `extract_features(..., summary=True)` → (B,512) for heuristics/stats
> - **WHY**: EEGPT outputs 4 summary tokens × 512 dims. Probes trained on all 2048 dims. Using 512 = CRASH.
> - **WHERE TO FLATTEN**: At probe boundaries (routers/trainers). During **Green** phase, inline `.flatten(1)` is acceptable to make tests pass. After **Refactor** phase, use the `prepare_probe_features` adapter exclusively. Never flatten inside model/encoder/compat internals.
> - **THE INVARIANT**: Probes MUST receive 2048-dim features (4×512 flattened)
>
> **🔍 Shape Contract:**
> - `summary=False` returns **numpy** array: (B,4,512) for batch, or (4,512) for single window
> - Single windows should be expanded to (1,4,512) before flattening
> - Probes expect **torch** tensor shape (B,2048) where B ≥ 1
> - **Must convert**: numpy → torch via `torch.as_tensor()`, then `.flatten(1)`
> - **Critical Rule**: API routers must NEVER use summary=True for probe paths

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
# Create P0-specific test files (not probe_utils yet - that's for refactor phase)
touch tests/unit/api/routers/test_eegpt_p0_dimensions.py
touch tests/unit/application/training/test_sleep_probe_p0_dimensions.py
touch tests/unit/infra/ml_models/test_eegpt_compat_p0_contract.py
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
    
    def test_analyze_endpoint_flattens_to_2048(self, monkeypatch, tmp_path):
        """P0: /eegpt/analyze must flatten (4,512) to (1,2048)."""
        from fastapi.testclient import TestClient
        from brain_go_brrr.api.app import app
        from brain_go_brrr.api.routers import eegpt as eegpt_router
        
        mock_model = Mock()
        # Single-window contract: (4,512) numpy on summary=False
        mock_model.extract_features.return_value = np.random.randn(4, 512).astype(np.float32)
        monkeypatch.setattr(eegpt_router, "get_eegpt_model", lambda: mock_model)
        
        # Probe that asserts 2048 at call-time
        def _probe_predict(x: torch.Tensor):
            if x.ndim != 2 or x.shape[-1] != 2048:
                pytest.fail(f"Probe got {tuple(x.shape)} expected (B,2048)")
            return torch.randn(x.shape[0], 2)
        
        mock_probe = Mock(predict_proba=_probe_predict)
        monkeypatch.setattr(eegpt_router, "get_probe", lambda: mock_probe, raising=False)
        
        # Hit the real route - THIS TEST MUST FAIL ON CURRENT CODE
        client = TestClient(app)
        files = {"file": ("fake.edf", b"x", "application/octet-stream")}
        response = client.post("/api/v1/eegpt/analyze", files=files)
        # Test will fail with dimension error before fix
            
    def test_batch_endpoint_flattens_to_2048(self):
        """P0: /analyze/batch must flatten (B,4,512) to (B,2048)."""
        # Mock returns numpy (B,4,512)
        mock_model.extract_features_batch.return_value = np.random.randn(32, 4, 512).astype(np.float32)
        
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
        # Mock self.eegpt_model.extract_features returns (B,4,512) when summary=False
        # Probe validates 2048 dims
        # THIS TEST MUST FAIL ON CURRENT CODE
        
    def test_evaluate_probe_flattens_to_2048(self):
        """P0: evaluate_probe must flatten to 2048."""
        # Mock self.eegpt_model.extract_features returns (B,4,512)
        # Verify probe receives (B,2048)
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
import numpy as np
import numpy.typing as npt

def extract_features(
    self,
    data: npt.NDArray[np.float32],  # float32 matches runtime
    channel_names: list[str] | None = None,
    summary: bool = True  # ADD THIS PARAMETER
) -> npt.NDArray[np.float32]:  # Returns float32
    # Forward summary to encoder:
    features = self.encoder.extract_features(data_tensor, summary=summary)
    # Return as-is - NO FLATTENING HERE (flatten at call-site only)
    return features.cpu().numpy().astype(np.float32, copy=False)

def extract_features_batch(
    self,
    windows: npt.NDArray[np.float32] | torch.Tensor,  # float32
    channel_names: list[str] | None = None,
    summary: bool = True  # ADD THIS PARAMETER
) -> npt.NDArray[np.float32]:  # Returns float32
    # Forward summary to encoder:
    features = self.encoder.extract_features(batch_tensor, summary=summary)
    # NO FLATTENING HERE - return (B,4,512) if summary=False
    return features.cpu().numpy().astype(np.float32, copy=False)
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
# In train_step (class method):
# P0: Probes require 2048-dim (4×512) features
features = self.eegpt_model.extract_features(eeg_data, summary=False)  # numpy (4,512) or (B,4,512)
features_tensor = torch.as_tensor(features, dtype=torch.float32)
if features_tensor.ndim == 2:  # Single window (4,512)
    features_tensor = features_tensor.unsqueeze(0)  # (1,4,512)
features_tensor = features_tensor.flatten(1)  # (B,4,512) → (B,2048)
# Pass features_tensor to probe

# In evaluate_probe function (module-level, no self):
# P0: Probes require 2048-dim (4×512) features
features = eegpt_model.extract_features(eeg_data, summary=False)  # numpy
features_tensor = torch.as_tensor(features, dtype=torch.float32)
if features_tensor.ndim == 2:  # Single window
    features_tensor = features_tensor.unsqueeze(0)
features_tensor = features_tensor.flatten(1)  # → (B,2048)
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
    
    Accepts: (4,512) single window or (B,4,512) batch from extract_features with summary=False
    Returns: (B,2048) ready for probe input, where B >= 1
    
    This is the SINGLE SOURCE OF TRUTH for probe preparation.
    """
    # Convert numpy to torch if needed (use as_tensor for zero-copy when possible)
    if isinstance(features, np.ndarray):
        features = torch.as_tensor(features, dtype=torch.float32)
    
    # Handle single window: (4,512) → (1,4,512)
    if features.ndim == 2 and features.shape == torch.Size([4, 512]):
        features = features.unsqueeze(0)  # Add batch dimension
    
    # Flatten batch: (B,4,512) → (B,2048)
    if features.ndim == 3 and features.shape[1] == 4 and features.shape[2] == 512:
        features = features.flatten(1)
    
    # Safety assertion
    assert_probe_ready(features)
    return features

def assert_probe_ready(features: torch.Tensor) -> None:
    """Verify features are probe-ready."""
    if features.ndim != 2 or features.shape[-1] != 2048:
        raise AssertionError(
            f"Probe expects (B,2048); got {tuple(features.shape)}. "
            f"Use summary=False and prepare_probe_features()."
        )
```

### Step 3.2: Create and Test Adapter
```bash
# NOW create the probe_utils test file
touch tests/unit/utils/test_probe_utils.py
```

```python
# tests/unit/utils/test_probe_utils.py
"""Test probe preparation utilities."""
import numpy as np
import torch
import pytest

def test_prepare_probe_features_batch():
    """Adapter correctly flattens (B,4,512) to (B,2048)."""
    from brain_go_brrr.utils.probe_utils import prepare_probe_features
    
    features = np.random.randn(32, 4, 512).astype(np.float32)
    prepared = prepare_probe_features(features)
    assert prepared.shape == (32, 2048)
    assert isinstance(prepared, torch.Tensor)

def test_prepare_probe_features_single_window():
    """Adapter handles single window (4,512) to (1,2048)."""
    from brain_go_brrr.utils.probe_utils import prepare_probe_features
    
    features = np.random.randn(4, 512).astype(np.float32)  # Single window
    prepared = prepare_probe_features(features)
    assert prepared.shape == (1, 2048)  # Batch dim added
    assert isinstance(prepared, torch.Tensor)

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

# Check multi-line calls that might be missing summary:
rg -nP 'extract_features(?:_batch)?\([^)]*\)' src/brain_go_brrr | rg -v 'summary\s*='

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
- **v7.0** (Sept 4): Final pristine version with all feedback integrated:
  - Fixed trainer attribute: `self.eegpt_model` not `self.feature_extractor`
  - Clarified compat is "supporting change" not a 7th crash site
  - Added explicit single-window shape handling (4,512) → (1,4,512)
  - Added shape contract unit test for compat verification
  - Added critical rule: API routers must never use summary=True for probe paths
- **v8.0** (Sept 4): Perfect implementation-ready version:
  - Fixed SSOT wording: flatten at boundaries via adapter, not "never in helpers"
  - Updated prepare_probe_features to handle single-window (4,512) case
  - Fixed test mocks to return realistic numpy arrays not torch tensors
  - Fixed header typo: "✅ ALLOWED vs ❌ FORBIDDEN"
  - Moved probe_utils test creation to refactor phase (not RED phase)
  - Added explicit single-window test case for adapter
- **v9.0** (Sept 4): Final singularity perfection version:
  - Fixed header to match v8.0 revision
  - Clarified SSOT: inline flatten OK in Green phase, adapter in Refactor
  - Fixed all trainer references to use correct attributes (self.eegpt_model / eegpt_model)
  - Updated compat type hints to float32 (matches runtime)
  - Made RED tests actually invoke endpoints to ensure they fail
  - Added single-window handling to all Phase 2 endpoint examples
  - Added multi-line grep pattern for verification

---

**END OF P0 CRITICAL FIXES WITH TDD IMPLEMENTATION**
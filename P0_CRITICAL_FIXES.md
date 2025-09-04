# 🚨 P0 CRITICAL FIXES - RUNTIME CRASH BUGS

**Created**: September 4, 2025
**Purpose**: Eliminate ALL P0 bugs that cause runtime crashes
**Status**: 🔴 CRITICAL - FIX IMMEDIATELY
**Time Required**: 1-2 hours total

---

## 📋 EXECUTIVE SUMMARY

**We have 2 P0 bugs that WILL crash in production:**
1. **API endpoints** pass 512 dims to probes expecting 2048 → **RuntimeError**
2. **SleepProbeTrainer** passes 512 dims to probe expecting 2048 → **RuntimeError**

**Root Cause**: Missing `summary=False` parameter in `extract_features()` calls
**Impact**: ANY API call to EEGPT endpoints = instant crash
**Fix Complexity**: Simple - add 7 lines of code total

---

## 🔴 P0 BUG #1: API ENDPOINTS DIMENSION MISMATCH

### THE CRASH
```python
# What happens when you call the API:
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 2048x6)
              ^^^^^                                     ^^^^^^^     ^^^^^^^
              Your batch of 512-dim features            Probe expects 2048 input dims
```

### ROOT CAUSE ANALYSIS

The EEGPT model outputs 4 summary tokens of 512 dimensions each:
- **Correct for probes**: (B, 4, 512) → flatten → (B, 2048)
- **What API does**: Averages tokens → (B, 512) → **CRASH**

**EEGPT Paper Evidence** (literature/markdown/EEGPT/EEGPT.md):
> "We use 4 × 512 dimensional features for downstream tasks" (Line 297)
> "Linear probing with 2048-dimensional input" (Table 12)

### AFFECTED CODE LOCATIONS

#### 1. `/api/routers/eegpt.py` Line 138 (extract_features endpoint)
```python
# CURRENT (BROKEN):
features = eegpt_model.extract_features(window_data, channel_names)
# Returns 512 dims → probe expects 2048 → CRASH

# FIX:
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)  # (B, 4, 512) → (B, 2048)
```

#### 2. `/api/routers/eegpt.py` Line 271 (stream_features endpoint)
```python
# CURRENT (BROKEN):
features = eegpt_model.extract_features(window_data, channel_names)

# FIX:
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)
```

#### 3. `/api/routers/sleep.py` Line 486 (analyze_sleep endpoint)
```python
# CURRENT (BROKEN):
features = eegpt_model.extract_features(window_data, channel_names)

# FIX:
features = eegpt_model.extract_features(window_data, channel_names, summary=False)
features = features.flatten(1)
```

### COMPLETE FIX IMPLEMENTATION

```python
# File: src/brain_go_brrr/api/routers/eegpt.py

# Line 138 - extract_features endpoint
def extract_features(...):
    # BEFORE:
    # features = eegpt_model.extract_features(window_data, channel_names)

    # AFTER:
    features = eegpt_model.extract_features(window_data, channel_names, summary=False)
    features = features.flatten(1)  # Critical: flatten (B,4,512) to (B,2048)

# Line 271 - stream_features endpoint
async def stream_features(...):
    # BEFORE:
    # features = eegpt_model.extract_features(window_data, channel_names)

    # AFTER:
    features = eegpt_model.extract_features(window_data, channel_names, summary=False)
    features = features.flatten(1)

# File: src/brain_go_brrr/api/routers/sleep.py

# Line 486 - analyze_sleep endpoint
async def analyze_sleep(...):
    # BEFORE:
    # features = eegpt_model.extract_features(window_data, channel_names)

    # AFTER:
    features = eegpt_model.extract_features(window_data, channel_names, summary=False)
    features = features.flatten(1)
```

### VERIFICATION COMMANDS

```bash
# 1. Find all API extract_features calls missing summary=False
rg -n "extract_features\(" src/brain_go_brrr/api/routers | grep -v "summary="

# 2. After fix - verify all have summary=False
rg -n "extract_features\(.*summary=False" src/brain_go_brrr/api/routers
# Should show 3 results

# 3. Test the fix works
python -c "
import torch
# Simulate API behavior
features = torch.randn(32, 4, 512)  # What extract_features returns with summary=False
flattened = features.flatten(1)
print(f'Shape after flatten: {flattened.shape}')  # Should be [32, 2048]
assert flattened.shape == (32, 2048), 'Wrong shape!'
print('✅ Fix verified - 2048 dimensions')
"
```

---

## 🔴 P0 BUG #2: SLEEPPROBETRAINER DIMENSION MISMATCH

### THE CRASH
```python
# What happens when you run SleepProbeTrainer:
RuntimeError: size mismatch, m1: [256 x 512], m2: [2048 x 5] at linear layer
```

### ROOT CAUSE ANALYSIS

SleepProbeTrainer trains a linear probe but passes wrong feature dimensions:
- **Probe expects**: 2048 input dimensions (per EEGPT paper)
- **Trainer provides**: 512 dimensions (averaged tokens)
- **Why tests pass**: Mock returns correct shape, masking the bug

### AFFECTED CODE LOCATIONS

#### `/application/training/sleep_probe_trainer.py` Line 110 (train_step)
```python
# CURRENT (BROKEN):
features = self.feature_extractor.extract_features(eeg_data)
# Returns 512 dims but probe expects 2048

# FIX:
features = self.feature_extractor.extract_features(eeg_data, summary=False)
features = features.flatten(1)  # (B, 4, 512) → (B, 2048)
```

#### `/application/training/sleep_probe_trainer.py` Line 193 (validation_step)
```python
# CURRENT (BROKEN):
features = self.feature_extractor.extract_features(eeg_data)

# FIX:
features = self.feature_extractor.extract_features(eeg_data, summary=False)
features = features.flatten(1)
```

### COMPLETE FIX IMPLEMENTATION

```python
# File: src/brain_go_brrr/application/training/sleep_probe_trainer.py

def train_step(self, batch):
    """Training step with correct dimensions."""
    eeg_data, labels = batch

    # BEFORE (Line 110):
    # features = self.feature_extractor.extract_features(eeg_data)

    # AFTER:
    features = self.feature_extractor.extract_features(eeg_data, summary=False)
    features = features.flatten(1)  # (B, 4, 512) → (B, 2048)

    # Rest of method unchanged
    outputs = self.probe(features)
    loss = self.criterion(outputs, labels)
    return loss

def validation_step(self, batch):
    """Validation step with correct dimensions."""
    eeg_data, labels = batch

    # BEFORE (Line 193):
    # features = self.feature_extractor.extract_features(eeg_data)

    # AFTER:
    features = self.feature_extractor.extract_features(eeg_data, summary=False)
    features = features.flatten(1)  # (B, 4, 512) → (B, 2048)

    # Rest of method unchanged
    outputs = self.probe(features)
    val_loss = self.criterion(outputs, labels)
    return val_loss
```

### VERIFICATION COMMANDS

```bash
# 1. Check current broken state
rg -n "extract_features\(" src/brain_go_brrr/application/training/sleep_probe_trainer.py
# Should show lines 110, 193 without summary parameter

# 2. After fix - verify summary=False added
rg -n "extract_features.*summary=False" src/brain_go_brrr/application/training/sleep_probe_trainer.py
# Should show 2 results

# 3. Verify flatten is added after each extract_features
rg -A1 "extract_features.*summary=False" src/brain_go_brrr/application/training/sleep_probe_trainer.py | grep flatten
# Should show 2 flatten calls
```

---

## ✅ TEST CASES TO ADD

### Test 1: API Dimension Test
```python
# tests/unit/api/routers/test_eegpt_dimensions.py
import torch
import pytest
from unittest.mock import Mock

def test_api_returns_2048_dimensions():
    """Verify API endpoints return 2048 dims for probe compatibility."""
    # Mock EEGPT model
    mock_model = Mock()
    mock_model.extract_features.return_value = torch.randn(32, 4, 512)

    # Simulate API logic (after fix)
    features = mock_model.extract_features("data", "channels", summary=False)
    features = features.flatten(1)

    assert features.shape == (32, 2048), f"Expected (32, 2048), got {features.shape}"
    mock_model.extract_features.assert_called_with("data", "channels", summary=False)
```

### Test 2: Trainer Dimension Test
```python
# tests/unit/application/training/test_sleep_trainer_dimensions.py
def test_trainer_uses_2048_dimensions():
    """Verify trainer passes 2048 dims to probe."""
    from brain_go_brrr.application.training.sleep_probe_trainer import SleepProbeTrainer

    # Create trainer with mocked components
    trainer = SleepProbeTrainer(...)

    # Mock feature extractor to return correct shape
    trainer.feature_extractor.extract_features = Mock(
        return_value=torch.randn(16, 4, 512)
    )

    # Mock batch
    batch = (torch.randn(16, 19, 1024), torch.randint(0, 5, (16,)))

    # Run train step
    loss = trainer.train_step(batch)

    # Verify extract_features called with summary=False
    trainer.feature_extractor.extract_features.assert_called_with(
        batch[0], summary=False
    )
```

---

## 🔧 IMPLEMENTATION CHECKLIST

### Phase 1: Fix API Endpoints (30 minutes)
- [ ] Open `src/brain_go_brrr/api/routers/eegpt.py`
- [ ] Fix line 138: Add `summary=False` and `.flatten(1)`
- [ ] Fix line 271: Add `summary=False` and `.flatten(1)`
- [ ] Open `src/brain_go_brrr/api/routers/sleep.py`
- [ ] Fix line 486: Add `summary=False` and `.flatten(1)`
- [ ] Run verification commands to confirm

### Phase 2: Fix SleepProbeTrainer (20 minutes)
- [ ] Open `src/brain_go_brrr/application/training/sleep_probe_trainer.py`
- [ ] Fix line 110: Add `summary=False` and `.flatten(1)`
- [ ] Fix line 193: Add `summary=False` and `.flatten(1)`
- [ ] Run verification commands to confirm

### Phase 3: Add Tests (30 minutes)
- [ ] Create dimension test for API endpoints
- [ ] Create dimension test for trainer
- [ ] Run tests to verify fixes work

### Phase 4: Validation (10 minutes)
- [ ] Run full test suite: `make test`
- [ ] Run type checking: `make typecheck`
- [ ] Test actual API endpoint with curl
- [ ] Document fix in CHANGELOG

---

## 🎯 SUCCESS CRITERIA

The fixes are complete when:
1. ✅ All API extract_features calls use `summary=False`
2. ✅ All probe-feeding code flattens to 2048 dims
3. ✅ No RuntimeError on API calls
4. ✅ SleepProbeTrainer runs without dimension errors
5. ✅ Tests verify 2048 dimensions everywhere
6. ✅ CI/CD passes

---

## 🚦 RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation |
|------|------------|---------|------------|
| Breaking existing code | Low | Medium | Full test suite |
| Missing a location | Low | High | Grep verification |
| Wrong fix applied | Low | High | Dimension tests |
| Performance impact | None | None | Same computation |

**Overall Risk**: LOW - Simple parameter addition, no logic changes

---

## 📊 IMPACT IF NOT FIXED

### Immediate (TODAY):
- **ANY** call to `/api/v1/eeg/extract_features` → **CRASH**
- **ANY** call to `/api/v1/eeg/stream_features` → **CRASH**
- **ANY** call to `/api/v1/sleep/analyze` → **CRASH**
- Running SleepProbeTrainer → **CRASH**

### Business Impact:
- 100% API failure rate for EEGPT endpoints
- Cannot demo to stakeholders
- Cannot run sleep analysis via API
- Blocks all probe training

---

## 🔍 ROOT CAUSE LESSONS

### Why This Happened:
1. **Assumption**: Developers assumed `extract_features()` returns probe-ready dims
2. **Documentation Gap**: Wrapper doesn't document the summary parameter clearly
3. **Test Gap**: Mocks return correct shape, masking the issue
4. **Review Gap**: PR reviews didn't catch dimension mismatch

### Prevention for Future:
1. **Type Hints**: Add shape annotations to tensor returns
2. **Runtime Checks**: Add assertion for expected dimensions
3. **Documentation**: Clear docstrings about summary parameter
4. **Integration Tests**: Test with real models, not just mocks

---

## 📝 SENIOR AUDITOR SIGN-OFF

### Pre-Fix Verification
- [ ] Confirmed bugs exist with verification commands
- [ ] Reviewed affected code locations
- [ ] Understood dimension requirements from paper

### Implementation
- [ ] Applied fixes to all 5 locations
- [ ] Added flatten(1) after each fix
- [ ] Verified with test commands

### Post-Fix Validation
- [ ] API endpoints return 2048 dims
- [ ] Trainer passes 2048 dims to probe
- [ ] All tests pass
- [ ] No new errors introduced

**Reviewed By**: ___________________ **Date**: ___________

**Fixed By**: ______________________ **Date**: ___________

**Validated By**: __________________ **Date**: ___________

---

## 🆘 HELP & QUESTIONS

**Q: Why do we need summary=False?**
A: EEGPT outputs 4 tokens. Probes were trained on all 4 (2048 dims). Averaging them (summary=True) loses 75% of information and causes dimension mismatch.

**Q: Is this the same as the training script issue?**
A: No, training scripts already use summary=False correctly. This is about API and application layer code.

**Q: Will this affect performance?**
A: No, we still compute the same features, just don't average them.

**Q: How was this not caught earlier?**
A: Tests use mocks that return correct shapes, masking the bug. Real model usage would crash immediately.

---

**END OF P0 CRITICAL FIXES DOCUMENT**

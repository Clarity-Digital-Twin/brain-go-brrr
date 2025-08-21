# 🔍 PHASE 1 DEEP INVESTIGATION REPORT

## Executive Summary
After deep investigation from first principles, Phase 1 involves 3 critical files with specific API compatibility issues that need careful handling.

## 🚨 CRITICAL FINDINGS

### 1. API Incompatibility Issues Found

#### Issue #1: `EEGPTWrapper` lacks `load_model()` method
- **File**: `infra/adapters/model_adapter.py:29`
- **Problem**: Calls `self.model.load_model()` but `EEGPTWrapper` doesn't have this method
- **Solution**: Remove the `load_model()` call - EEGPTWrapper loads automatically in `__init__`

#### Issue #2: `EEGPTWrapper` lacks `device` parameter
- **File**: `infra/adapters/model_adapter.py:28`
- **Problem**: Passes `device` to constructor but `create_normalized_eegpt` doesn't accept it
- **Solution**: Add `.to(device)` after creation or modify wrapper to accept device

#### Issue #3: `extract_features` signature mismatch
- **File**: `infra/adapters/model_adapter.py:49`
- **Problem**: Calls with `(eeg_data, channel_names)` but wrapper expects `(x, chan_ids, return_all_temporal)`
- **Solution**: Adapter needs to transform the call properly

## 📊 DETAILED FILE ANALYSIS

### File 1: `domain/quality/controller.py`

**Current Code (Lines 107-116)**:
```python
if model is None and eegpt_model_path is not None:
    from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
    try:
        model = EEGPTModel(eegpt_model_path)  # type: ignore[assignment]
        self.model = model
    except Exception:
        pass
```

**What It Needs**:
- Model with `extract_features(eeg_array)` method ✅ (EEGPTWrapper has this)

**Replacement**:
```python
if model is None and eegpt_model_path is not None:
    from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
    try:
        model = create_normalized_eegpt(checkpoint_path=eegpt_model_path)
        self.model = model
    except Exception:
        pass
```

**Call Sites**: Used by 6 factories and 6 test files

### File 2: `infra/adapters/model_adapter.py`

**Current Code (Lines 27-29)**:
```python
self.model = EEGPTModel(checkpoint_path=model_path, device=device)
self.model.load_model()
```

**Current extract_features (Lines 45-50)**:
```python
channel_names = [f"CH_{i}" for i in range(eeg_data.shape[0])]
eeg_data_64 = eeg_data.astype(np.float64)
features = self.model.extract_features(eeg_data_64, channel_names)
return features.astype(np.float32)
```

**Problems**:
1. No `device` parameter in `create_normalized_eegpt`
2. No `load_model()` method in `EEGPTWrapper`
3. `extract_features` expects different parameters

**COMPLEX FIX REQUIRED** - Need to either:
- Option A: Modify `EEGPTWrapper` to be fully compatible
- Option B: Create adapter layer in `model_adapter.py`
- Option C: Change all callers to use new API

**Call Sites**: Used by 4 factories and 1 test file

### File 3: `application/use_cases/tasks/enhanced_abnormality_detection.py`

**Current Code (Lines 91-93)**:
```python
probe = EEGPTTwoLayerProbe(
    backbone_dim=768, n_input_channels=n_channels, n_classes=n_classes
)
```

**New API Signature**:
```python
EEGPTProbe(
    checkpoint_path=None,
    n_classes=2,
    n_input_channels=20,
    architecture='two_layer',  # KEY DIFFERENCE
    hidden_dim=128,  # Different param name
    # No backbone_dim parameter!
)
```

**Problem**: Parameter mismatch
- Old: `backbone_dim=768`
- New: No equivalent parameter (might be hardcoded)

**Replacement**:
```python
probe = EEGPTProbe(
    architecture='two_layer',
    n_input_channels=n_channels,
    n_classes=n_classes,
    # Note: backbone_dim is not configurable in new API
)
```

**Call Sites**: Only used internally in this file

## 🔴 BLOCKING ISSUES

### Must Resolve Before Phase 1:

1. **Device Support**: How to handle device parameter?
   - Add device support to wrapper?
   - Use `.to(device)` after creation?

2. **Channel Names**: How to convert channel names to channel IDs?
   - EEGPTWrapper expects channel IDs (integers?)
   - model_adapter provides channel names (strings)

3. **Backbone Dimension**: Is 768 hardcoded in new probe?
   - Old code explicitly sets `backbone_dim=768`
   - New API doesn't have this parameter

## 🛠️ PROPOSED SOLUTIONS

### Solution A: Minimal Changes (RISKY)
Just update imports and hope for the best:
```python
# domain/quality/controller.py
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
model = create_normalized_eegpt(checkpoint_path=eegpt_model_path)

# enhanced_abnormality_detection.py
from brain_go_brrr.infra.ml_models.eegpt_probe_unified import EEGPTProbe
probe = EEGPTProbe(architecture='two_layer', n_input_channels=n_channels, n_classes=n_classes)
```

### Solution B: Adapter Pattern (SAFER)
Create compatibility layer in model_adapter:
```python
class EEGPTModelAdapter(EEGModelPort):
    def __init__(self, model_path: str, device: str = "cpu"):
        # Create model without device
        self.model = create_normalized_eegpt(checkpoint_path=model_path)
        # Move to device
        self.model = self.model.to(device)
        # No load_model() needed

    def extract_features(self, eeg_data, sampling_rate=256):
        # Convert to torch tensor
        import torch
        x = torch.from_numpy(eeg_data).unsqueeze(0)  # Add batch dimension
        # Call without channel names
        features = self.model.extract_features(x)
        # Convert back to numpy
        return features.squeeze(0).detach().cpu().numpy()
```

### Solution C: Full Compatibility Wrapper (SAFEST)
Create a compatibility wrapper that matches old API exactly:
```python
class EEGPTModelCompat:
    """Compatibility wrapper for old EEGPTModel API."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.wrapper = create_normalized_eegpt(checkpoint_path)
        self.device = device
        self.wrapper = self.wrapper.to(device)

    def load_model(self):
        """No-op for compatibility."""
        pass

    def extract_features(self, eeg_data, channel_names=None):
        """Match old API signature."""
        # Implementation to match old behavior
```

## 📋 VERIFICATION NEEDED

Before proceeding, we need to verify:

1. **Test Device Handling**:
```python
model = create_normalized_eegpt(checkpoint_path="path.ckpt")
model = model.to("cuda")  # Does this work?
```

2. **Test Feature Extraction**:
```python
import numpy as np
import torch
eeg_data = np.random.randn(20, 1024).astype(np.float32)
x = torch.from_numpy(eeg_data).unsqueeze(0)
features = model.extract_features(x)
print(features.shape)  # What shape?
```

3. **Test Probe Compatibility**:
```python
probe = EEGPTProbe(architecture='two_layer', n_classes=2)
# Does it expect 768-dim input or something else?
```

## 🚦 GO/NO-GO DECISION

### ⚠️ HOLD - Need Clarification

**Cannot proceed until we resolve**:
1. How to handle device parameter
2. How to handle channel names vs IDs
3. Whether backbone_dim=768 is critical

**Recommendation**:
1. First, test the new APIs manually to understand their behavior
2. Then choose between Solution B (Adapter) or Solution C (Compatibility Wrapper)
3. Only then proceed with Phase 1 changes

## 📝 UPDATED PHASE 1 PLAN

### Step 0: Verification (DO FIRST)
```bash
# Test new APIs
uv run python -c "
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
import torch
import numpy as np

# Test 1: Device handling
model = create_normalized_eegpt()
print('Can move to device?', hasattr(model, 'to'))

# Test 2: Feature extraction
x = torch.randn(1, 20, 1024)
try:
    features = model.extract_features(x)
    print('Feature shape:', features.shape)
except Exception as e:
    print('Extract failed:', e)
"
```

### Step 1: Choose Solution
Based on verification results, choose:
- **Solution B** if minor incompatibilities
- **Solution C** if major incompatibilities

### Step 2: Implement Chosen Solution
- Update the 3 files
- Run tests after each file
- Commit if tests pass

## ❌ DO NOT PROCEED YET

We need to resolve the blocking issues first. The investigation revealed that the APIs are NOT drop-in replacements as we hoped.

---

*Investigation Complete: Phase 1 is more complex than initially thought. Need verification before proceeding.*

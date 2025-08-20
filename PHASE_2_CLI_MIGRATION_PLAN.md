# 📋 PHASE 2: CLI Migration Plan

## Executive Summary
Phase 2 involves updating the CLI module to use the new EEGPT wrapper API. The CLI already has a **partial hybrid implementation** that we need to clean up.

## 🔍 Current State Analysis

### File: `src/brain_go_brrr/cli.py`

#### Lines 117-134 (stream command)
**Current Implementation (HYBRID MESS)**:
```python
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel  # Line 117

# Initialize model
model = EEGPTModel(config={"device": "cpu"}, auto_load=False)  # Line 126

# Use mock model for now
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt  # Line 129

model.encoder = create_normalized_eegpt(checkpoint_path=None)  # Line 131
if model.encoder is not None:
    model.encoder.to(model.device)
model.is_loaded = True
```

#### Line 151 (feature extraction)
```python
features = model.extract_features(data_window, ch_names)
```

## 🛠️ Required Changes

### Option A: Direct Wrapper Usage (CLEAN)
Replace the hybrid mess with direct wrapper usage:

```python
# Line 117: Remove old import
# from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel  # DELETE

# Line 125-134: Replace with clean implementation
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
import torch

# Initialize model
model = create_normalized_eegpt(checkpoint_path=None)
if model is not None:
    model = model.to("cpu")

# Line 151: Update feature extraction
# Convert to tensor and add batch dimension
data_tensor = torch.from_numpy(data_window).unsqueeze(0).float()
features_tensor = model.extract_features(data_tensor)
features = features_tensor.squeeze(0).detach().cpu().numpy()
```

### Option B: Compatibility Wrapper (SAFER)
Create a thin compatibility wrapper to maintain the same interface:

```python
# New class in cli.py or separate file
class CLIModelWrapper:
    """Wrapper to maintain CLI interface compatibility."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder = create_normalized_eegpt(checkpoint_path=None)
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
    
    def extract_features(self, data_window, channel_names):
        """Extract features with old API signature."""
        import torch
        # Convert numpy to tensor
        data_tensor = torch.from_numpy(data_window).unsqueeze(0).float()
        # Extract features
        features_tensor = self.encoder.extract_features(data_tensor)
        # Convert back to numpy
        return features_tensor.squeeze(0).detach().cpu().numpy()

# Then in stream command:
model = CLIModelWrapper(device="cpu")
# Line 151 stays the same:
features = model.extract_features(data_window, ch_names)
```

## 🚨 Critical Considerations

### API Differences
1. **Input Format**:
   - Old: `extract_features(numpy_array, channel_names)`
   - New: `extract_features(torch_tensor)` (no channel names)

2. **Output Format**:
   - Old: numpy array directly
   - New: torch tensor (needs conversion)

3. **Device Handling**:
   - Old: `EEGPTModel(config={"device": "cpu"})`
   - New: `model.to("cpu")` after creation

### Testing Requirements
The CLI stream command is tested in:
- `tests/unit/test_cli.py`
- `tests/integration/test_cli_streaming.py` (if exists)

## 📊 Impact Analysis

### Files Affected
1. `src/brain_go_brrr/cli.py` - 1 command affected (`stream`)

### Dependencies
- No other files depend on the CLI's internal model usage
- The stream command outputs JSON, so format must stay consistent

### Risk Assessment
- **Low Risk**: CLI is relatively isolated
- **Testing**: Has unit tests that mock the model anyway
- **User Impact**: None if output format stays the same

## ✅ Recommended Approach

**Use Option B (Compatibility Wrapper)** because:
1. Maintains exact same interface
2. Localizes changes to one place
3. Easy to test in isolation
4. Can be removed later when fully migrated

## 📝 Implementation Steps

1. **Add CLIModelWrapper class** (lines 100-120)
2. **Remove old import** (line 117)
3. **Replace model initialization** (lines 126-134)
4. **Test stream command** with real EDF file
5. **Verify JSON output format** unchanged

## 🧪 Verification Commands

```bash
# Test the stream command
uv run brain-go-brrr stream tests/data/sample.edf --format json

# Run CLI tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit/test_cli.py -xvs

# Check for any remaining old imports
grep -n "from brain_go_brrr.infra.ml_models.eegpt_model import" src/brain_go_brrr/cli.py
```

## ⚠️ Rollback Plan

If issues arise:
1. The old EEGPTModel still exists (with deprecation warning)
2. Can revert to hybrid approach temporarily
3. No database or state changes involved

## 📈 Success Criteria

- [ ] Stream command works with real EDF files
- [ ] JSON output format unchanged
- [ ] No deprecation warnings from CLI
- [ ] All CLI tests pass
- [ ] Performance similar or better

---

**Estimated Time**: 30 minutes
**Complexity**: Low
**Risk**: Low
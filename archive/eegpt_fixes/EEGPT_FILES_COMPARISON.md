# EEGPT Files Detailed Comparison

## The Core Question: Why Do We Have So Many EEGPT Files?

### File Size Comparison
- `eegpt_architecture.py`: 20KB - Core implementation
- `eegpt_model.py`: 26KB - BIGGEST file (red flag!)
- `eegpt_wrapper.py`: 6KB - Clean wrapper
- `eegpt_linear_probe.py`: 6KB
- `eegpt_linear_probe_robust.py`: 11KB
- `eegpt_two_layer_probe.py`: 9KB

## Functionality Breakdown

### `eegpt_architecture.py` ✅ ESSENTIAL
**What it does**:
- Core Vision Transformer implementation
- Attention mechanisms, patch embedding
- The actual EEGPT model

**Unique features**:
- `EEGTransformer` class
- `return_all_temporal` flag (we just added)
- Rotary position embeddings

**Verdict**: **KEEP** - This is the foundation

### `eegpt_wrapper.py` ✅ CLEAN WRAPPER
**What it does**:
- Wraps architecture with normalization
- Simple, focused on one job

**Unique features**:
- Normalization handling
- Clean `extract_features()` method
- `create_normalized_eegpt()` factory

**Verdict**: **KEEP** - Does one thing well

### `eegpt_model.py` 🚨 THE MONSTER
**What it does**: TOO MUCH!
- Another wrapper around eegpt_wrapper (!)
- Config management
- Preprocessing functions
- Window extraction
- Batch processing
- MNE integration
- Abnormality prediction
- Analysis orchestration

**Unique features** (that maybe shouldn't be here):
- `EEGPTConfig` dataclass
- `predict_abnormality()` - full pipeline
- `analyze()` - orchestration
- `process_recording()` - batch processing
- `extract_windows()` - preprocessing
- `preprocess_for_eegpt()` - standalone function
- `extract_features_from_raw()` - standalone function

**Problems**:
1. **Wraps a wrapper**: Uses `eegpt_wrapper.py` internally!
2. **Kitchen sink**: Mixes model, preprocessing, analysis
3. **Redundant config**: Has its own config system
4. **API confusion**: Different interface than wrapper

**Verdict**: **REFACTOR** - Split into appropriate modules

## The Probe Mess

### Why 4 Different Probes?

1. **`linear_probe.py`** (Generic)
   - Generic `LinearProbeHead` class
   - Works with any features
   - Has pooling options (mean, max, cls)

2. **`eegpt_linear_probe.py`** (EEGPT-specific)
   - Channel adaptation layer
   - Uses `LinearWithConstraint`
   - Two-layer classifier

3. **`eegpt_linear_probe_robust.py`** (Paranoid version)
   - Everything from #2 PLUS:
   - Input clipping
   - NaN checks everywhere
   - Gradient-friendly ops
   - **NEVER IMPORTED DIRECTLY!**

4. **`eegpt_two_layer_probe.py`** (Another variant)
   - Similar to #2 but different architecture
   - Different initialization

**The Problem**: Each developer created their own probe instead of configuring one!

## Dependency Hell

```
eegpt_architecture.py (CORE)
    ↑
eegpt_wrapper.py (wraps core)
    ↑
eegpt_model.py (wraps wrapper! 🤦)
    ↑
CLI, API routes (use the double-wrapped version)
```

This is **inception-level wrapping**!

## What Each File is ACTUALLY Used For

### Used Correctly ✅
- Training scripts → `eegpt_wrapper.py`
- Our fixes → `eegpt_architecture.py` + `eegpt_wrapper.py`

### Used Incorrectly ❌
- CLI → `eegpt_model.py` (should use wrapper)
- API routes → `eegpt_model.py` (should use wrapper)
- Tasks → Random probe variants

## The Real Problem: `eegpt_model.py`

This file is trying to be:
1. A model wrapper (redundant with `eegpt_wrapper.py`)
2. A preprocessing module (should be separate)
3. An analysis pipeline (should be in application layer)
4. A config manager (should use central config)
5. A batch processor (should be separate utility)

## Recommended Refactoring

### Step 1: Extract from `eegpt_model.py`
```python
# Move to preprocessing module:
- preprocess_for_eegpt()
- extract_windows()

# Move to analysis/pipeline:
- predict_abnormality()
- analyze()
- process_recording()

# Move to data utilities:
- extract_features_batch()

# Keep only if needed:
- EEGPTConfig (or use central config)
```

### Step 2: Unify Probes
```python
# Single configurable probe
class EEGPTProbe(nn.Module):
    def __init__(self,
                 architecture="linear",  # or "two_layer"
                 robust_mode=False,      # NaN handling
                 channel_adapter=True,   # Channel adaptation
                 pooling="none"):        # For compatibility
        ...
```

### Step 3: Update Imports
```python
# OLD: from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
# NEW: from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper

# OLD: model = EEGPTModel(checkpoint_path)
# NEW: model = create_normalized_eegpt(checkpoint_path)
```

## Files to Delete After Refactoring

1. **Immediately deletable**:
   - `eegpt_linear_probe_robust.py` - Never imported
   - `cache_port.py` - Useless abstraction

2. **After migration**:
   - `eegpt_model.py` - Move useful parts elsewhere
   - One of the probe variants - Keep unified version

## Migration Risk Assessment

### High Risk Areas
- **API routes**: Currently depend on `EEGPTModel` class
- **CLI**: Uses `EEGPTModel` for commands

### Safe to Change
- **Training scripts**: Already use correct wrapper
- **Unused files**: Can delete immediately

## The Cache Situation

Less critical but still messy:
- `cache.py` - Actual implementation ✅
- `cache_factory.py` - Unnecessary factory pattern
- `cache_port.py` - Empty protocol

Just use `cache.py` directly!

## Action Items

1. **Document API surface** of `EEGPTModel` that's actually used
2. **Create migration shim** for backward compatibility
3. **Extract functions** to appropriate modules
4. **Unify probe implementations**
5. **Update imports** gradually
6. **Delete redundant files**

## Bottom Line

We have a **26KB monster file** (`eegpt_model.py`) that wraps a 6KB wrapper (`eegpt_wrapper.py`) that wraps the actual model. This is architectural debt that needs cleaning!

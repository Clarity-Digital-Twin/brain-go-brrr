# Infrastructure ML Models Investigation Report

## 🚨 Executive Summary

The `infra/ml_models` directory is a **MESS** with:
- **6 different EEGPT files** with overlapping functionality
- **3 different probe implementations** doing similar things
- **3 cache files** with unclear relationships
- Multiple unused/redundant implementations

## 📊 File Analysis

### EEGPT Files (6 total - TOO MANY!)

#### 1. `eegpt_architecture.py` (20KB) ✅ CORE
- **Purpose**: Core Vision Transformer implementation
- **Status**: ACTIVELY USED - We just fixed this for temporal features
- **Key classes**: `EEGTransformer`, `Block`, `Attention`, `PatchEmbed`
- **Verdict**: **KEEP** - This is the foundation

#### 2. `eegpt_wrapper.py` (6KB) ✅ ACTIVELY USED
- **Purpose**: Wraps architecture with normalization
- **Status**: ACTIVELY USED - We just updated this
- **Key classes**: `EEGPTWrapper`, `create_normalized_eegpt()`
- **Used by**: Training scripts, many other files
- **Verdict**: **KEEP** - Essential wrapper

#### 3. `eegpt_model.py` (26KB) ⚠️ REDUNDANT MONSTER
- **Purpose**: Another wrapper with TOO MUCH stuff
- **Status**: WIDELY IMPORTED but probably redundant
- **Used by**: CLI, API routes, adapters
- **Key classes**: `EEGPTModel`, `EEGPTConfig`, preprocessing functions
- **Problems**:
  - Duplicates functionality from `eegpt_wrapper.py`
  - Has its own config system
  - Mixes model with preprocessing
- **Verdict**: **REFACTOR/REMOVE** - Merge useful parts into wrapper

#### 4. `eegpt_linear_probe.py` (6KB) ❓ UNCLEAR
- **Purpose**: Linear probe with channel adaptation
- **Used by**: `abnormality_detection.py` task
- **Features**: Channel adapter, two-layer classifier
- **Verdict**: **MAYBE KEEP** - Check if actively used

#### 5. `eegpt_linear_probe_robust.py` (11KB) ❓ REDUNDANT
- **Purpose**: "Robust" version with NaN handling
- **Used by**: Nothing directly!
- **Features**: Input clipping, validation, NaN prevention
- **Problems**: Duplicates linear_probe.py with extra checks
- **Verdict**: **MERGE OR REMOVE** - Combine with regular probe

#### 6. `eegpt_two_layer_probe.py` (9KB) ❓ ANOTHER PROBE
- **Purpose**: Yet another probe variant
- **Used by**: `enhanced_abnormality_detection.py`
- **Features**: Two-layer architecture
- **Verdict**: **REDUNDANT** - Why do we need 3 probes?

### Linear Probe Confusion

We have **FOUR** different probe implementations:
1. `linear_probe.py` - Generic linear probe head
2. `eegpt_linear_probe.py` - EEGPT-specific probe
3. `eegpt_linear_probe_robust.py` - "Robust" EEGPT probe
4. `eegpt_two_layer_probe.py` - Two-layer EEGPT probe

**This is insane!** Should be ONE configurable class.

### Cache Files (3 total)

#### 1. `cache.py` (6KB)
- **Purpose**: Main cache implementation
- **Key classes**: `InMemoryCache`
- **Features**: TTL, size limits, async support

#### 2. `cache_factory.py` (3KB)
- **Purpose**: Factory for creating caches
- **Problem**: Over-engineered for one cache type

#### 3. `cache_port.py` (240 bytes)
- **Purpose**: Abstract interface
- **Problem**: Nearly empty, just a protocol

## 🔍 Dependency Analysis

### Who Uses What?

```
eegpt_architecture.py
  ↑
  └── eegpt_wrapper.py
        ↑
        ├── eegpt_model.py (redundant wrapper)
        ├── eegpt_linear_probe.py
        ├── eegpt_linear_probe_robust.py
        └── eegpt_two_layer_probe.py

linear_probe.py (standalone, generic)
```

### Import Graph Shows Problems

- **CLI imports**: `eegpt_model.py` AND `eegpt_wrapper.py` (redundant)
- **API routes import**: `eegpt_model.py` (should use wrapper)
- **Training scripts import**: `eegpt_wrapper.py` (correct)
- **Tasks import**: Different probes randomly

## 🔥 Critical Issues

### 1. Model Wrapper Redundancy
- `eegpt_model.py` (26KB) duplicates `eegpt_wrapper.py` (6KB)
- Both wrap the same `eegpt_architecture.py`
- Different APIs for same functionality

### 2. Probe Proliferation
- 4 different probe implementations
- No clear reason for variants
- Each task uses a different one randomly

### 3. Naming Confusion
- `eegpt_model.py` vs `eegpt_wrapper.py` - unclear difference
- `linear_probe.py` vs `eegpt_linear_probe.py` - overlap?
- "robust" variant suggests others are fragile

### 4. Cache Over-Engineering
- Factory pattern for ONE cache type
- Abstract port with no implementations
- Could be one simple file

## 📋 Recommended Actions

### Immediate Cleanup

#### 1. Merge Probe Implementations
```python
# Single configurable probe in eegpt_probe.py
class EEGPTProbe(nn.Module):
    def __init__(self,
                 n_layers=1,  # 1 or 2 layer
                 robust=False,  # NaN handling
                 channel_adapter=True):  # Channel adaptation
        # Combine all variants
```

#### 2. Remove eegpt_model.py
- Move useful functions to `eegpt_wrapper.py`
- Update all imports to use wrapper
- Delete the 26KB monster file

#### 3. Simplify Cache
- Merge all cache files into one `cache.py`
- Remove factory and port abstractions
- Keep it simple

### Migration Plan

```bash
# Step 1: Identify all imports
grep -r "from.*eegpt_model" src/
grep -r "from.*linear_probe" src/

# Step 2: Create unified probe
# Combine features from all variants

# Step 3: Update imports gradually
# Replace eegpt_model with eegpt_wrapper

# Step 4: Delete redundant files
```

## 📊 Impact Analysis

### Files That Need Updates
- **CLI** (`cli.py`): Switch from eegpt_model to wrapper
- **API routes** (`eegpt.py`, `sleep.py`): Use wrapper
- **Tasks**: Standardize on single probe
- **Tests**: Update imports

### Risk Assessment
- **High Risk**: Breaking API routes if not careful
- **Medium Risk**: Tasks might expect specific probe behavior
- **Low Risk**: Cache changes (barely used)

## 🎯 Priority Order

1. **Keep using what works**: Don't touch `eegpt_architecture.py` and `eegpt_wrapper.py`
2. **Fix probe mess**: Merge 4 probes into 1 configurable class
3. **Remove eegpt_model.py**: After updating imports
4. **Simplify cache**: Low priority, not breaking anything

## 📈 Current Usage Stats

Based on grep analysis:
- `eegpt_model.py`: 8 imports (needs migration)
- `eegpt_wrapper.py`: 7 imports (correct usage)
- `eegpt_linear_probe.py`: 1 import
- `eegpt_linear_probe_robust.py`: 0 direct imports
- `eegpt_two_layer_probe.py`: 1 import

## 🚨 DO NOT TOUCH

These files are working and were just fixed:
- `eegpt_architecture.py` - Core implementation
- `eegpt_wrapper.py` - Active wrapper

## 🗑️ Can Probably Delete

After proper migration:
- `eegpt_linear_probe_robust.py` - Never imported directly
- `cache_port.py` - Useless abstraction
- `cache_factory.py` - Over-engineered

## Next Steps

1. **Document current behavior** of each probe variant
2. **Create unified probe** with all features
3. **Test extensively** before removing anything
4. **Update imports** one module at a time
5. **Delete redundant files** only after verification

The infra/ml_models directory needs serious cleanup, but BE CAREFUL - lots of code depends on these files!

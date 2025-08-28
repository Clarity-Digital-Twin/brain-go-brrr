# THE REAL FIX PLAN - UNFUCKING THE EEGPT MESS

## THE CURRENT DISASTER (10 EEGPT FILES!)

### ML Models Layer (4 files)
1. **eegpt_architecture.py** - Creates raw EEGPT encoder
2. **eegpt_wrapper.py** - Adds normalization to encoder 
3. **eegpt_compat.py** - Another wrapper (used by API)
4. **eegpt_probe_unified.py** - Encoder + linear head (used by tasks)

### Adapters Layer (2 files)
5. **eegpt_classifier.py** - Adapter for classification
6. **eegpt_feature_extractor.py** - Adapter for feature extraction

### Preprocessing Layer (2 files)  
7. **eegpt_preprocessing.py** - One preprocessing approach
8. **eegpt_prepare.py** - Another preprocessing approach

### Pipeline Layer (1 file)
9. **eegpt_orchestration.py** - Orchestrates EEGPT pipeline

### API Layer (1 file)
10. **api/routers/eegpt.py** - REST endpoints

## WHO USES WHAT (THE DEPENDENCY MESS)

### eegpt_architecture.py (The Foundation)
- Used by: eegpt_wrapper.py, eegpt_compat.py
- Purpose: Creates raw encoder model
- VERDICT: **KEEP** - This is the core

### eegpt_wrapper.py (Primary Wrapper)
- Used by: CLI, orchestration, probe_unified
- Purpose: Adds normalization to encoder
- VERDICT: **KEEP** - This is the main wrapper

### eegpt_compat.py (API Wrapper)
- Used by: API endpoints, feature extractor adapter
- Purpose: Alternative wrapper with different interface
- VERDICT: **MERGE INTO WRAPPER** - Redundant

### eegpt_probe_unified.py (Task Implementation)
- Used by: abnormality_detection, enhanced_abnormality
- Purpose: Complete model (encoder + classifier head)
- VERDICT: **KEEP** - Needed for tasks

### eegpt_classifier.py & eegpt_feature_extractor.py (Adapters)
- Used by: API routers
- Purpose: Adapt models for API
- VERDICT: **DELETE** - Can use wrapper directly

### eegpt_preprocessing.py & eegpt_prepare.py
- Used by: Various places
- Purpose: Redundant preprocessing
- VERDICT: **DELETE ONE** - Keep best one

## THE ROOT PROBLEMS

1. **THREE WAYS TO USE EEGPT:**
   - eegpt_wrapper.py (CLI/orchestration)
   - eegpt_compat.py (API)
   - eegpt_probe_unified.py (tasks)

2. **NORMALIZATION CONFUSION:**
   - Some normalize in wrapper
   - Some normalize in dataset
   - Some don't normalize (BROKEN!)

3. **CHANNEL CONFUSION:**
   - TUAB needs 19 channels (no Fz)
   - TUEV needs 20 channels (with Fz)
   - Some code assumes 23 channels (WRONG!)

4. **WINDOW SIZE CONFUSION:**
   - EEGPT trained on 4s @ 256Hz = 1024 samples
   - Some code uses 8s windows (WRONG!)
   - Some code uses 1000 samples (WRONG!)

## THE ACTUAL FIX PLAN

### Phase 1: Document Everything (CURRENT)
✅ Map all dependencies
✅ Identify redundancies
✅ Understand usage patterns

### Phase 2: Create Single Source of Truth

#### A. Single EEGPT Model Interface
```python
# src/brain_go_brrr/infra/ml_models/eegpt.py (NEW)
class EEGPT:
    """The ONE way to use EEGPT."""
    
    def __init__(self, checkpoint_path=None, normalize=True):
        self.encoder = create_eegpt_architecture(checkpoint_path)
        self.normalize = normalize
        
    def extract_features(self, x):
        """Get features for any downstream task."""
        if self.normalize:
            x = self._normalize(x)
        return self.encoder(x)
        
    def _normalize(self, x):
        """ALWAYS z-score normalize."""
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)
```

#### B. Single Preprocessing Pipeline
```python
# src/brain_go_brrr/domain/preprocessing/eegpt_preprocess.py (KEEP ONE)
def preprocess_for_eegpt(raw, channels, window_size=4.0):
    """The ONE way to preprocess for EEGPT."""
    # 1. Resample to 256Hz
    # 2. Select correct channels (19 for TUAB, 20 for TUEV)
    # 3. Window to 4s = 1024 samples
    # 4. Return in Volts (MNE default)
```

#### C. Task-Specific Heads
```python
# src/brain_go_brrr/application/use_cases/tasks/probes.py
class LinearProbe(nn.Module):
    """Reusable linear probe head."""
    def __init__(self, in_features=2048, out_features=2):
        self.head = nn.Linear(in_features, out_features)
```

### Phase 3: Migration Steps (WITH TESTS GREEN)

1. **Create new unified EEGPT class**
   - Combines best of wrapper/compat
   - Single normalization strategy
   - Clear interface

2. **Add deprecation warnings to old modules**
   ```python
   # In eegpt_compat.py
   warnings.warn("Use brain_go_brrr.infra.ml_models.eegpt instead", DeprecationWarning)
   ```

3. **Update imports one by one**
   - Start with least used (adapters)
   - Then API
   - Then tasks
   - Finally experiments

4. **Delete deprecated code**
   - Only after all imports updated
   - Only after tests pass

### Phase 4: Enforce Standards

#### Channel Standards
```python
TUAB_CHANNELS = ["FP1", "FP2", ...]  # 19 channels, NO Fz
TUEV_CHANNELS = ["FP1", "FPZ", ...]  # 20 channels, WITH Fz

assert len(channels) in [19, 20], f"Expected 19 or 20 channels, got {len(channels)}"
```

#### Window Standards
```python
WINDOW_SECONDS = 4.0
SAMPLING_RATE = 256
WINDOW_SAMPLES = 1024

assert n_samples == 1024, f"Expected 1024 samples, got {n_samples}"
```

#### Normalization Standards
```python
# ALWAYS z-score normalize
# NEVER identity normalization
# ALWAYS check scale
assert abs(data.mean()) < 0.1, "Data not centered"
assert 0.8 < data.std() < 1.2, "Data not normalized"
```

## THE IMMEDIATE ACTIONS

### 1. Fix Broken Normalization (DONE ✅)
- Changed identity fallback to 50μV scale

### 2. Restore Deleted Files (DONE ✅)
- Restored eegpt_probe_unified.py
- Restored tuab_enhanced_dataset.py
- Restored tuab_cached_dataset.py

### 3. Next: Create Unified Interface
- Write single EEGPT class
- Test it works
- Migrate one component
- Repeat

## SUCCESS CRITERIA

1. **ONE way to use EEGPT** (not 3)
2. **ONE preprocessing pipeline** (not 2)
3. **ZERO redundant adapters**
4. **ALWAYS normalized data**
5. **CORRECT channel counts** (19 or 20)
6. **CORRECT window size** (1024 samples)
7. **ALL TESTS GREEN**

## WHAT NOT TO DO (LESSONS LEARNED)

❌ Don't delete files without checking dependencies
❌ Don't create parallel implementations
❌ Don't assume normalization happens somewhere
❌ Don't trust that "someone else" handles preprocessing
❌ Don't build in isolation

## THE TRUTH

We have a working system buried under layers of redundancy. The fix is:
1. Identify the good parts (architecture, wrapper, probe_unified)
2. Merge redundant parts (compat → wrapper)
3. Delete unnecessary adapters
4. Enforce standards everywhere
5. Test at every step

This will take time but it's the only way to unfuck this properly.
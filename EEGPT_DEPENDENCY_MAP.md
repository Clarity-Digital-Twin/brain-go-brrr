# COMPLETE EEGPT DEPENDENCY MAP

## Import Analysis (Who Uses What)

### 1. eegpt_architecture.py
**Imports it:**
```
src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py
src/brain_go_brrr/infra/ml_models/eegpt_compat.py
tests/unit/test_eegpt_architecture.py
tests/unit/test_models_eegpt_wrapper.py
tests/unit/test_models_eegpt_model.py
```
**Status:** CRITICAL - Foundation layer, must keep

### 2. eegpt_wrapper.py
**Imports it:**
```
src/brain_go_brrr/infra/ml_models/eegpt_probe_unified.py
src/brain_go_brrr/infra/ml_models/eegpt_compat.py
src/brain_go_brrr/domain/quality/controller.py
src/brain_go_brrr/cli.py
src/brain_go_brrr/application/pipeline/eegpt_orchestration.py
tests/unit/test_eegpt_wrapper.py (multiple test files)
```
**Status:** PRIMARY INTERFACE - Most used

### 3. eegpt_compat.py
**Imports it:**
```
src/brain_go_brrr/api/app.py
src/brain_go_brrr/api/routers/eegpt.py
src/brain_go_brrr/api/routers/sleep.py
src/brain_go_brrr/infra/adapters/eegpt_feature_extractor.py
```
**Status:** API ONLY - Could merge into wrapper

### 4. eegpt_probe_unified.py
**Imports it:**
```
src/brain_go_brrr/application/use_cases/tasks/abnormality_detection.py
src/brain_go_brrr/application/use_cases/tasks/enhanced_abnormality_detection.py
src/brain_go_brrr/application/pipeline/eegpt_orchestration.py
tests/unit/test_robust_eegpt_probe.py
tests/unit/test_models_linear_probe.py
```
**Status:** TASK CRITICAL - Needed for probes

### 5. eegpt_classifier.py & eegpt_feature_extractor.py
**Imports them:**
```
src/brain_go_brrr/api/routers/eegpt.py (indirect through deps)
```
**Status:** REDUNDANT - Just thin wrappers

### 6. eegpt_preprocessing.py & eegpt_prepare.py
**Imports them:**
```
Various scattered imports
```
**Status:** REDUNDANT - Pick one

## THE INHERITANCE HIERARCHY

```
eegpt_architecture.py (Raw Encoder)
    ↓
eegpt_wrapper.py (Adds Normalization)
    ↓                    ↓
eegpt_compat.py    eegpt_probe_unified.py
(API Interface)    (Adds Linear Head)
    ↓                    ↓
eegpt_feature_     AbnormalityDetectionProbe
extractor.py       (Task Implementation)
```

## CRITICAL PATHS

### Path 1: API → Analysis
```
API Request
→ eegpt_compat.py
→ eegpt_wrapper.py
→ eegpt_architecture.py
```

### Path 2: Task → Training
```
Training Script
→ AbnormalityDetectionProbe
→ eegpt_probe_unified.py
→ eegpt_wrapper.py
→ eegpt_architecture.py
```

### Path 3: CLI → Processing
```
CLI Command
→ eegpt_wrapper.py
→ eegpt_architecture.py
```

## NORMALIZATION LOCATIONS (THE MESS)

1. **eegpt_wrapper.py:55-65** - Fallback normalization (FIXED)
2. **experiments/datasets/tuab_mne_dataset.py:150** - Dataset normalization
3. **experiments/datasets/tuev_mne_dataset.py:185** - Dataset normalization
4. **src/infra/data/tuab_dataset.py** - Unknown
5. **src/infra/data/tuab_enhanced_dataset.py** - Unknown
6. **src/infra/data/tuab_cached_dataset.py** - Unknown

**PROBLEM:** Multiple normalization points = confusion

## CHANNEL HANDLING LOCATIONS

1. **experiments/utils/channel_validation.py** - Channel enforcement
2. **Various dataset classes** - Channel selection
3. **eegpt_probe_unified.py:71** - Channel adapter

**PROBLEM:** No single source of truth for channels

## WINDOW SIZE HANDLING

1. **Hardcoded 1024 in some places**
2. **Configurable in datasets**
3. **Sometimes 1000 (WRONG!)**
4. **Sometimes 8s = 2048 (WRONG!)**

**PROBLEM:** Inconsistent window sizes

## THE MIGRATION PRIORITY

### HIGH PRIORITY (Breaking Issues)
1. Fix all normalization to single approach
2. Fix all window sizes to 1024
3. Fix channel counts (19 for TUAB, 20 for TUEV)

### MEDIUM PRIORITY (Redundancy)
1. Merge eegpt_compat into eegpt_wrapper
2. Delete redundant adapters
3. Consolidate preprocessing

### LOW PRIORITY (Nice to Have)
1. Better naming conventions
2. More documentation
3. Better test coverage

## FILES TO DELETE (EVENTUALLY)

After migration with deprecation warnings:
1. eegpt_compat.py (merge into wrapper)
2. eegpt_classifier.py (redundant adapter)
3. eegpt_feature_extractor.py (redundant adapter)
4. One of the preprocessing files
5. Redundant dataset implementations

## FILES TO KEEP (CRITICAL)

1. eegpt_architecture.py (foundation)
2. eegpt_wrapper.py (main interface)
3. eegpt_probe_unified.py (task probes)
4. One preprocessing implementation
5. One dataset per dataset type

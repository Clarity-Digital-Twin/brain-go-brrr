# Experiments Folder Analysis - August 28, 2025

## ✅ What's Already Fixed
- Datasets are now thin shims importing from src/
- Both tuab_mne_dataset.py and tuev_mne_dataset.py are <20 lines
- No Lightning imports (avoided the training hang bug)
- Proper deprecation warnings in place

## 🔴 Issues Still Present

### 1. sys.path.insert Hacks (4 files)
**Files with sys.path.insert:**
```
experiments/eegpt_linear_probe/mne_integration/cache_builder.py
experiments/eegpt_linear_probe/test_tuev_implementation.py
experiments/eegpt_linear_probe/train_tuab_mne.py
experiments/eegpt_linear_probe/train_tuev_mne.py
```

**Problem:** Using sys.path manipulation to import from experiments/
**Solution:** Remove sys.path.insert, use proper imports from src/

### 2. Duplicate Preprocessor
**Files:**
- `experiments/eegpt_linear_probe/mne_integration/preprocessor.py`
- `src/brain_go_brrr/infra/preprocessing/mne_preprocessor.py`

**Problem:** 99% identical code, only differs in docstring formatting
**Solution:** Delete experiments version, import from src/

### 3. Duplicate TUEV Preprocessor
**Files:**
- `experiments/eegpt_linear_probe/mne_integration/tuev_preprocessor.py`
- Should use src/ version if one exists

**Problem:** Another duplicate preprocessor
**Solution:** Check if src has TUEV preprocessor, if not move it there

### 4. Imports Still Using experiments/
**Training scripts still import from experiments:**
```python
from experiments.eegpt_linear_probe.datasets.tuab_mne_dataset import TUABMNEDataset
from experiments.eegpt_linear_probe.utils.collate_tuab import collate_tuab_batch
```

**Solution:** Change to:
```python
from brain_go_brrr.infra.data.tuab_dataset import TUABDataset
from brain_go_brrr.utils.collate import collate_tuab_batch  # if exists
```

### 5. Utils Not in src/
**Collate functions only in experiments:**
- `experiments/eegpt_linear_probe/utils/collate_tuab.py`
- `experiments/eegpt_linear_probe/utils/collate_tuev.py`
- `experiments/eegpt_linear_probe/utils/custom_collate_fixed.py`

**Problem:** Reusable utils should be in src/
**Solution:** Move to src/brain_go_brrr/utils/collate/

## 📋 Action Plan

### Phase 1: Move Missing Components to src/
1. ✅ Datasets - DONE (already shims)
2. ❌ Collate functions - Move to src/brain_go_brrr/utils/
3. ❌ TUEV preprocessor - Move to src/ if not duplicate

### Phase 2: Fix Imports
1. Remove all sys.path.insert lines
2. Update all imports to use src/ components
3. Test that training still works

### Phase 3: Delete Duplicates
1. Delete experiments/preprocessor.py (use src/ version)
2. Delete dataset shims once all imports updated
3. Delete empty directories

## 🎯 End Goal
```
experiments/eegpt_linear_probe/
├── configs/           # Experiment configs only
├── scripts/           # Launch scripts only
├── train_tuab_mne.py  # Training loop only (<200 lines)
├── train_tuev_mne.py  # Training loop only (<200 lines)
└── test_*.py          # Experiment-specific tests

# Everything else imported from src/
```

## ⚠️ Critical Note
DO NOT break training that's currently working! Test each change carefully.

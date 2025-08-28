# 🔬 EEGPT Training Failure Investigation

**Date**: 2025-08-28  
**Issue**: TUAB training achieved 0.50 AUROC (random chance) for all 10 epochs  
**Root Cause**: Multiple systemic issues with normalization and data pipeline

## 📊 Key Evidence

### Training Metrics (FAILED)
- **AUROC**: 0.5000 for ALL epochs (perfect random chance)
- **Loss**: ~0.70-0.72 (stuck near -ln(0.5) = 0.693)
- **Training completed**: 10 full epochs but learned NOTHING

### Cached Data Analysis
```python
# Sample windows from cache:
Window 000000: shape=[19, 1024], label=0.0, range=[-2.19e-05, 2.19e-05]
Window 025000: shape=[19, 1024], label=1.0, range=[-4.51e-05, 5.15e-05]
# Data is in VOLTS (1e-5 scale), not normalized!
```

## 🔴 Critical Problems Identified

### 1. **Normalization Chain Completely Broken**

#### Data Scale Issues:
- **MNE outputs**: Volts (1e-5 to 1e-4 range = 10-100 μV)
- **Cache contains**: Raw Volts (NOT normalized)
- **EEGPT expects**: Normalized data ~N(0,1)
- **EEGPT receives**: 1e-5 scale values (appears as zeros to the model!)

#### The Failed Chain:
1. MNE preprocessor → outputs Volts ✓
2. Cache builder → saves raw Volts (NO normalization) ❌
3. Dataset loader → loads raw Volts (NO normalization) ❌
4. Training script → uses basic EEGPTWrapper ❌
5. EEGPTWrapper → looks for normalization file (doesn't exist) ❌
6. Falls back → identity normalization (mean=0, std=1) ❌
7. Model sees → 1e-5 values instead of ~N(0,1) ❌

### 2. **Multiple Redundant/Conflicting Systems**

#### Collate Functions (3 versions!):
- `utils/custom_collate_fixed.py` - Old version with 20→19 channel hack
- `utils/collate_tuab.py` - TUAB-specific (19 channels)
- `utils/collate_tuev.py` - TUEV-specific (20 channels)

#### Preprocessors (2 versions):
- `mne_integration/preprocessor.py` - TUABPreprocessor (19 channels)
- `mne_integration/tuev_preprocessor.py` - TUEVPreprocessor (20 channels, inherits from TUAB)

#### Dataset Implementations (3+ versions):
- `datasets/tuab_mne_dataset.py` - MNE-based TUAB
- `datasets/tuev_mne_dataset.py` - MNE-based TUEV
- `datasets/tuev_dataset_cached.py` - TUEVCachedDatasetPadded (why "Padded"?)
- Main codebase has MORE: `tuab_dataset.py`, `tuab_cached_dataset.py`, `tuab_enhanced_dataset.py`

#### Normalization Systems (2 approaches):
- `EEGPTWrapper` - Basic wrapper with identity normalization fallback
- `create_normalized_eegpt()` - Advanced function that EXISTS but ISN'T USED!

### 3. **Configuration/Documentation Mess**

#### Multiple overlapping docs:
- CHANNEL_FIX_SUMMARY.md
- CHANNEL_SPECIFICATIONS.md  
- FINAL_AUDIT_BEFORE_LAUNCH.md
- FINAL_FIX_SUMMARY.md
- FIX_AUDIT_SUMMARY.md
- READY_FOR_AUDIT.md
- TECH_DEBT_CRITICAL.md
- TUEV_FINAL_CHECKLIST.md
- TUEV_FIXES_SUMMARY.md
- TUEV_POLISH_SUMMARY.md

*Do we even know which is authoritative?*

## 🔍 Further Investigation Needed

### Questions to Answer:

1. **Why does `create_normalized_eegpt()` exist but isn't used?**
   - Location: `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py`
   - Has sophisticated normalization but bypassed in training

2. **What's the difference between all the dataset implementations?**
   - Why "Padded" in TUEVCachedDatasetPadded?
   - Which one should we actually use?

3. **Where else might normalization be happening (or not)?**
   - Check eegpt_architecture.py
   - Check domain/preprocessing/eegpt_preprocessing.py
   - Check if any dataset does normalization internally

4. **Are there competing configuration systems?**
   - YAML configs vs hardcoded values
   - Environment variables vs config files

5. **Cache contamination scope:**
   - How many cached datasets exist?
   - Are they all affected by the normalization issue?
   - When were they built (before or after channel fixes)?

## 🎯 Proposed Solution Path

### Immediate Fix (for testing):
```python
# In train_tuab_mne.py, after line 290:
model = EEGPTWrapper(checkpoint_path=eegpt_checkpoint)
# ADD:
model.set_normalization_params(mean=0.0, std=1e-5)  # Scale up!
```

### Proper Fix (after investigation):
1. Choose ONE normalization strategy
2. Choose ONE dataset implementation  
3. Choose ONE collate function per dataset
4. Delete all redundant code
5. Rebuild cache with proper normalization
6. Create SSOT documentation

## 📝 Next Investigation Steps

- [ ] Check what `create_normalized_eegpt()` actually does
- [ ] Trace through eegpt_preprocessing.py
- [ ] Find all cache directories and check their build dates
- [ ] Map the complete data flow from raw EDF to model
- [ ] Check if other training scripts exist that might work

## 🚨 Key Insight

**The system has multiple parallel implementations trying to solve the same problem, none communicating with each other, and the critical normalization step is missing from ALL paths!**

This is not a simple bug - it's a systemic architectural issue where redundant systems create confusion about which path is canonical.
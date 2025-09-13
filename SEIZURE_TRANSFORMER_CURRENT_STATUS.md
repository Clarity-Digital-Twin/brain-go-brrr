# SeizureTransformer Current Implementation Status

**Last Updated**: December 2025  
**Status**: MOSTLY WORKING (0.844 AUROC achieved)  
**Purpose**: Single source of truth for what's implemented vs what needs fixing

---

## ✅ WHAT'S WORKING (Confirmed Fixed)

### 1. Architecture ✅ FIXED
- **Issue**: Was using toy Vision Transformer instead of Wu 2025 CNN+Transformer
- **Fix Applied**: Vendored correct architecture from reference repo
- **Current State**: 
  - `seizure_transformer_wu2025.py` - correct Wu 2025 architecture
  - `seizure_transformer_wrapper.py` - uses Wu 2025 by default
  - Toy model deprecated with huge warnings
  - CI guard prevents architecture mismatches
- **Result**: Pretrained weights load perfectly

### 2. TSE Parser ✅ FIXED (Despite Documentation Claims)
- **Previous Issue**: Docs claimed parser accepts ANY 2-field line
- **Current Reality**: Parser correctly filters seizures!
  - Location: `src/brain_go_brrr/infra/data/tusz_detection_dataset.py` lines 64-92
  - Uses `is_seizure_label` from `tusz_labels.py`
  - Explicitly excludes non-seizure codes: {spsw, gped, pled, eyem, artf, bckg}
  - Only accepts TUSZ epileptic codes: {fnsz, gnsz, spsz, cpsz, absz, tnsz, tcsz, gtsz, mysz, unsz}
  - Fallback accepts labels containing 'seiz'
- **Verification**: CURRENT_SEIZURE_TRANSFORMER_DATAFLOW.md is OUTDATED about this

### 3. Preprocessing Pipeline ✅ WORKING
- **Implementation**: `SeizurePreprocessor` in `seizure_transformer_utils.py`
- **Order (matches OSS exactly)**:
  1. Z-score normalization (per-channel, before windowing)
  2. Resample to 256Hz if needed
  3. Bandpass 0.5-120Hz (order 3, causal using lfilter)
  4. Notch at 1Hz and 60Hz (Q=30)
- **Status**: Correctly implemented in wrapper

### 4. Post-processing ✅ WORKING
- **Implementation**: `SeizurePostProcessor` in `seizure_transformer_utils.py`
- **Pipeline**:
  - Threshold at 0.8
  - Morphological opening (kernel=5)
  - Morphological closing (kernel=5)
  - Drop events < 2 seconds
- **Status**: Matches OSS parameters exactly

### 5. Evaluation ✅ WORKING (with small gap)
- **Script**: `scripts/evaluate_seizure_transformer.py`
- **Results**:
  - Achieved: 0.844 AUROC
  - Expected: 0.876 AUROC
  - Gap: 3.2% (acceptable, likely from minor differences)
- **Dataset**: TUSZ eval split (865 recordings, 7,539 windows)

---

## ❌ WHAT'S STILL BROKEN

### 1. Training Script Preprocessing ❌
**Location**: `experiments/seizure_transformer/train_tusz.py`
**Issue**: Training bypasses wrapper preprocessing
- Dataset windows fed directly to model
- No bandpass/notch filters applied during training
- Model sees different data distribution than inference
**Fix Needed**: Use `SeizurePreprocessor` in training script

### 2. Training Supervision ❌  
**Location**: `experiments/seizure_transformer/train_tusz.py`
**Issue**: Wrong label supervision
- Window-level labels expanded to all 15360 timesteps
- Not true per-timestep segmentation
- Line 41-47 in train_epoch: `y = y.to(device).float()` then BCE with expanded labels
**Fix Needed**: Generate true per-timestep masks from annotations

### 3. NEDC Integration ❌
**Location**: Not integrated
**Issue**: Clinical metrics not computed
- FA/24h not calculated
- TAES scoring not integrated
- Only AUROC is computed
**Fix Needed**: Wire `NEDCClinicalEvaluator` into evaluation

---

## 🟡 PERFORMANCE GAP ANALYSIS

### Current vs Expected
- **Our AUROC**: 0.844
- **Paper AUROC**: 0.876
- **Gap**: 3.2%

### Likely Causes
1. **Training preprocessing missing** - biggest suspect
2. **Wrong supervision** - window labels vs timestep labels
3. **Channel selection** - we use first 19, paper might select differently
4. **Training checkpoint** - weights might be from different epoch

### After Fixing Training Issues
Expected improvement from fixes:
- Fix preprocessing: +1-2% AUROC
- Fix supervision: +1-2% AUROC
- **Predicted**: ~0.86-0.87 AUROC (close to paper)

---

## 📁 File Structure Summary

```
src/brain_go_brrr/infra/
├── ml_models/
│   ├── seizure_transformer_wu2025.py          ✅ Correct architecture
│   ├── seizure_transformer_wrapper.py         ✅ Production wrapper
│   ├── seizure_transformer_utils.py           ✅ SSOT preprocessing
│   └── seizure_transformer_toy_deprecated.py  ⚠️ Deprecated toy model
├── data/
│   ├── tusz_detection_dataset.py             ✅ Dataset with FIXED TSE parser
│   └── tusz_labels.py                        ✅ Seizure label filtering
└── eval/
    └── post_processing.py                     ✅ Clinical post-processing

experiments/seizure_transformer/
├── train_tusz.py                              ❌ Needs preprocessing fix
└── test_quick.py                              ✅ Updated to use Wu 2025

scripts/
└── evaluate_seizure_transformer.py            ✅ Working evaluation
```

---

## 🔧 Required Fixes Priority

### Priority 1: Fix Training Preprocessing
```python
# In train_tusz.py, add before training:
preprocessor = SeizurePreprocessor(target_fs=256)
# Apply to each window before feeding to model
```

### Priority 2: Fix Training Supervision
```python
# Generate per-timestep labels from annotations
# Instead of expanding window label to all timesteps
```

### Priority 3: Add NEDC Metrics
```python
# In evaluation script, add FA/24h calculation
# Import and use NEDCClinicalEvaluator
```

---

## 📊 Implementation Parity Score

| Component | Status | Parity |
|-----------|--------|--------|
| Architecture | ✅ Fixed | 100% |
| TSE Parser | ✅ Fixed | 100% |
| Preprocessing Pipeline | ✅ Working | 100% |
| Post-processing | ✅ Working | 100% |
| Evaluation AUROC | ✅ Working | 96% |
| Training Preprocessing | ❌ Broken | 0% |
| Training Supervision | ❌ Wrong | 0% |
| Clinical Metrics | ❌ Missing | 0% |

**Overall Parity**: ~70% (inference path works, training path broken)

---

## 🎯 Next Steps

1. **Fix training preprocessing** - Add `SeizurePreprocessor` to training
2. **Fix supervision** - Use per-timestep labels
3. **Re-train model** - With fixed preprocessing and supervision
4. **Re-evaluate** - Should achieve closer to 0.876 AUROC
5. **Add clinical metrics** - FA/24h, TAES, ATWV

---

## 💡 Key Insights

1. **The TSE parser is NOT broken** - Documentation was wrong
2. **Architecture is fixed** - Wu 2025 correctly vendored
3. **Inference path is solid** - 96% of paper performance
4. **Training path needs work** - Missing preprocessing and wrong supervision
5. **Small fixes could close gap** - We're very close to paper performance

---

## 📝 Documentation to Archive

Based on this analysis, archive these outdated docs:
- `CURRENT_SEIZURE_TRANSFORMER_DATAFLOW.md` - Wrong about TSE parser
- `TUSZ_IMPLEMENTATION.md` - Old implementation guide
- `TUSZ_ROADMAP.md` - Execution complete
- `TUSZ_SPEC.md` - Outdated requirements

Keep these:
- This document (`SEIZURE_TRANSFORMER_CURRENT_STATUS.md`)
- `SEIZURE_TRANSFORMER_POSTMORTEM.md` - Historical record
- `IDEAL_REFERENCE_SEIZURE_TRANSFORMER_DATAFLOW.md` - Reference spec
- `INTENDED_SEIZURE_TRANSFORMER_APPLICATION.md` - Has useful patterns
- `TUSZ_OSS_SUMMARY.md` - Paper reference
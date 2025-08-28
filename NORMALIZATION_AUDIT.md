# NORMALIZATION AUDIT - WHERE THE FUCK IS IT HAPPENING?

## CRITICAL FINDING: DOUBLE NORMALIZATION RISK!

### Dataset Classes (src/infra/data/)

1. **tuab_dataset.py**
   - Line 390-393: NORMALIZES if `self.normalize=True`
   - Method: Per-channel z-score
   - Default: normalize=True

2. **tuab_enhanced_dataset.py**
   - Line 97: Has normalize parameter
   - Line 183: Passes to parent class
   - Inherits normalization from parent

3. **tuab_cached_dataset.py**
   - Line 30: Has normalize parameter
   - Line 58: Stores it
   - BUT WHERE DOES IT APPLY IT?

### Model Wrappers (src/infra/ml_models/)

4. **eegpt_wrapper.py**
   - Line 55-65: ALWAYS NORMALIZES (after our fix)
   - Method: Global z-score with 50μV std
   - No way to disable!

5. **eegpt_compat.py**
   - Uses eegpt_wrapper internally
   - So also normalizes

6. **eegpt_probe_unified.py**
   - Uses eegpt_wrapper internally
   - So also normalizes

### Experiments (experiments/eegpt_linear_probe/)

7. **datasets/tuab_mne_dataset.py**
   - Line 150-153: NORMALIZES per window
   - Method: Per-window z-score
   - Always on!

8. **datasets/tuev_mne_dataset.py**
   - Line 185-188: NORMALIZES per window
   - Method: Per-window z-score
   - Always on!

## THE DOUBLE NORMALIZATION BUG!

**HOLY FUCK - WE'RE NORMALIZING TWICE:**

1. Dataset normalizes: data → z-score (mean=0, std=1)
2. Wrapper normalizes AGAIN: z-score → weird scale

**This explains everything!**
- First norm: 50μV → N(0,1)
- Second norm: N(0,1) → divided by 50μV → tiny values!

## THE FIX

### Option 1: Normalize ONLY in Dataset
```python
# In eegpt_wrapper.py
self.normalize = False  # Let dataset handle it
```

### Option 2: Normalize ONLY in Model
```python
# In datasets
self.normalize = False  # Let model handle it
```

### Option 3: Smart Detection
```python
# In eegpt_wrapper.py
if abs(x.mean()) < 0.1 and 0.8 < x.std() < 1.2:
    # Already normalized, skip
    return x
else:
    # Normalize it
    return (x - x.mean()) / (x.std() + 1e-8)
```

## RECOMMENDED FIX: Option 1 (Dataset Only)

**Why:**
1. Dataset knows the data scale (Volts vs μV)
2. Dataset can compute proper statistics
3. Model just processes what it gets
4. Single point of control

**Implementation:**
1. Keep normalization in datasets (they do it right)
2. Disable normalization in wrapper
3. Add assertion to check data is normalized

## THE IMMEDIATE FIX

```python
# In eegpt_wrapper.py:__init__
if normalization_path is None:
    # Data should be normalized by dataset
    self.normalize = False
    logger.info("Expecting pre-normalized data from dataset")
```

## VERIFICATION CHECKLIST

- [ ] Check dataset outputs are N(0,1)
- [ ] Check model inputs are N(0,1)
- [ ] Check no double normalization
- [ ] Check training converges
- [ ] Check AUROC > 0.5

## THE TRUTH

We've been normalizing twice:
1. Dataset: Raw → N(0,1) ✅ CORRECT
2. Model: N(0,1) → Tiny values ❌ WRONG

This is why training might still fail even with "normalization"!
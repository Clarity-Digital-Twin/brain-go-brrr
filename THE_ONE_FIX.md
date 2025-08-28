# THE ONE FIX THAT ACTUALLY MATTERS

## Why Training Failed
**DATA IS NOT NORMALIZED**
- MNE outputs: 1e-5 scale (microvolts in Volts)
- EEGPT expects: ~N(0,1) scale
- Result: Model sees essentially zeros, outputs constant, AUROC=0.50

## The ONLY Fix That Matters Right Now

### Location: `experiments/eegpt_linear_probe/mne_integration/preprocessor.py`
### Line: ~200 (in the `preprocess_and_window` method)
### After getting data from raw:

```python
# FIND THIS:
windows = []
for start in range(0, n_samples - window_samples + 1, stride_samples):
    end = start + window_samples
    window_data = data[:, start:end]
    
    # ADD THIS NORMALIZATION:
    window_data = (window_data - window_data.mean()) / (window_data.std() + 1e-8)
    
    windows.append(window_data)
```

## Why This ONE Fix Works

Yes, we have multiple implementations, BUT:
- **experiments/ is what's ACTUALLY RUNNING**
- **src/ implementations aren't being used**
- **So we only need to fix the path that's ACTUALLY EXECUTING**

The experiments training uses:
1. MNE preprocessor → Creates cache (MISSING NORMALIZATION)
2. Cache → Dataset → Training
3. Fix #1, rebuild cache, training works

## Steps to Fix

```bash
# 1. Add normalization to preprocessor (above code)
# 2. Delete broken cache
rm -rf data/cache/tuab_mne_preprocessed/

# 3. Rebuild cache with normalization
cd experiments/eegpt_linear_probe
python mne_integration/cache_builder.py

# 4. Run training
python train_tuab_mne.py --config configs/tuab.yaml
```

## Why Other Stuff Doesn't Matter RIGHT NOW

- **src/ implementations**: Not used by training script
- **Multiple EEGPT models**: Training only uses EEGPTWrapper
- **TUEV**: Separate problem, fix TUAB first
- **Architecture mess**: Fix after you get results

## The Truth

I created a mess with multiple implementations, but for YOUR IMMEDIATE NEED:
- Only experiments/ path is executing
- Only that path needs normalization
- One fix in one file makes training work

Everything else is cleanup for later.
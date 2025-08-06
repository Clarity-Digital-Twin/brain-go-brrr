# 🚨 CRITICAL: ROOT CAUSE OF NaN TRAINING CRASH FOUND!

## THE SMOKING GUN

The training crashed because of **CATASTROPHIC NORMALIZATION FAILURE**:

```
Input mean: 0.000000000009385832980114282
Input std: 0.00002124020075157719
```

When the model normalizes data: `(x - mean) / std`, it's dividing by 0.000021, causing a **47,000x explosion** in values!

## Why This Happened

1. **Wrong normalization file**: The `normalization.json` was computed on Sleep-EDF data (which has tiny microvolt values)
2. **TUAB data has different scale**: TUAB data is already normalized (mean=0, std≈1)
3. **Dividing by near-zero**: When you divide normalized data by 0.000021, you get values around 47,000
4. **fp16 overflow**: These huge values cause gradients to explode in mixed precision training
5. **NaN propagation**: Once one layer produces NaN, it spreads through the entire network

## The Evidence

From the investigation:
```
Before normalization:
  Data scale: mean=0.000000, std=0.998217

After normalization:
  Normalized scale: mean=0.000206, std=46974.476562  <-- BOOM! 47,000x scale!
```

## Why I Missed This

1. I focused on the wrong things (label smoothing, learning rate, etc.)
2. I didn't check the ACTUAL normalization values being used
3. I assumed the normalization file was correct
4. I didn't trace through the data pipeline step by step

## The Fix

### Option 1: Remove the normalization file
```bash
rm /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/models/eegpt/pretrained/normalization.json
```

### Option 2: Disable normalization in the model
```python
backbone = create_normalized_eegpt(
    checkpoint_path="...",
    normalize=False  # <-- Disable normalization
)
```

### Option 3: Compute correct normalization stats for TUAB
```python
# TUAB data is already normalized, so we should use:
mean = 0.0
std = 1.0
```

## Immediate Action

```bash
# Remove the broken normalization file
rm data/models/eegpt/pretrained/normalization.json

# Rerun training with stable config
./experiments/eegpt_linear_probe/RUN_STABLE_TRAINING.sh
```

## Lessons Learned

1. **ALWAYS CHECK THE DATA PIPELINE** - not just the model
2. **Verify normalization stats** match your dataset
3. **Trace actual values** through the pipeline, not just shapes
4. **Don't trust existing files** - they might be from different experiments

This is a classic case of "the bug was in the data, not the code"!
# NaN Issue Fix Guide

## Problem Summary
During training, we observed frequent "WARNING: NaN detected in EEGPT features" messages. This indicates numerical instability that will hurt model performance.

## Root Causes
1. **Input normalization**: Division by near-zero std when channels have constant values
2. **Mixed precision (fp16)**: Can cause overflow in transformer activations  
3. **Extreme input values**: Some EEG windows have very large amplitudes
4. **EEGPT backbone**: The pretrained model may output NaN for certain edge cases

## Fixes Implemented

### 1. Robust Linear Probe (`eegpt_linear_probe_robust.py`)
- Added input validation and clipping
- Robust normalization with epsilon (1e-5)
- Feature validation after EEGPT
- Gradient-friendly operations
- Statistics tracking (nan_count, clip_count)

### 2. Robust Training Script (`train_nan_robust.py`)
- NaN debug callback to track issues
- Input data validation
- Gradient clipping (norm=1.0)
- Full precision (fp32) for stability
- Anomaly detection enabled
- Conservative hyperparameters

### 3. Key Changes
```python
# Before (causes NaN):
std = x.std(dim=-1, keepdim=True)  # Can be 0!
x = (x - mean) / std

# After (robust):
variance = (x_centered ** 2).mean(dim=-1, keepdim=True)
std = torch.sqrt(variance + 1e-5)  # Never zero
x_normalized = x_centered / std
x_normalized = torch.clamp(x_normalized, min=-10, max=10)
```

## Usage

### For Next Training Run:
```bash
# Test fixes first
python experiments/eegpt_linear_probe/test_nan_fixes.py

# Run robust training
python experiments/eegpt_linear_probe/train_nan_robust.py
```

### To Use Robust Model in Code:
```python
from brain_go_brrr.models.eegpt_linear_probe_robust import RobustEEGPTLinearProbe

# Drop-in replacement
model = RobustEEGPTLinearProbe(
    checkpoint_path=checkpoint,
    n_input_channels=20,
    n_classes=2
)
```

## Monitoring
The robust model tracks:
- `nan_count`: How many batches had NaN
- `clip_count`: How many batches needed clipping

Check these after training to assess data quality.

## Expected Improvements
1. No more NaN warnings during training
2. Stable loss curves (no sudden spikes)
3. Better final AUROC (target: ≥0.87)
4. Consistent validation metrics

## If Issues Persist
1. Check specific EDF files causing issues
2. Consider preprocessing outlier removal
3. Try even more conservative hyperparameters
4. Inspect EEGPT backbone outputs directly
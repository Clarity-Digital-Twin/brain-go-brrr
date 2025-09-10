# TUEV Channel Mismatch Analysis - CRITICAL FINDING

## The Problem
Our TUEV training crashes with channel/patch dimension mismatch when trying to use EEGPT.

## Root Cause Discovery

### What EEGPT Reference Does (CORRECT):
1. **Input**: 23 TUEV channels 
2. **Channel Conversion**: Conv2d(23, 20, kernel_size=1) - Maps 23→20 channels
3. **EEGPT Model**: Initialized with `img_size=[20, 1000]` 
4. **Key insight**: The model EXPECTS 20 channels in its patch embedding!

From `reference_repos/EEGPT/downstream_tueg/run_class_finetuning_EEGPT_change_tuev.py`:
```python
# Line 201-206: Define 20 target channels
use_channels_names = [      
    'FP1','FPZ', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2' 
]  # 20 channels!

# Line 208-209: Define 23 input channels
ch_names = ['EEG FP1-REF', 'EEG FP2-REF', ..., 'EEG T2-REF']  # 23 channels

# Line 212-217: Create model
model = EEGPTClassifier(
    in_channels=len(ch_names),      # 23 input channels
    img_size=[len(use_channels_names), 1000],  # [20, 1000] for EEGPT!
    use_chan_conv=True,              # Enable 23→20 conversion
)
```

### What We Were Doing (WRONG):
1. **Input**: 23 TUEV channels
2. **Channel Conversion**: Maps 23→20 correctly
3. **EEGPT Model**: Still configured for 19 or default channels!
4. **Result**: Patch embedding dimension mismatch

## The Critical Misunderstanding

We thought EEGPT was a fixed model that always expects 19 channels. **WRONG!**

The truth:
- EEGPT's pretrained weights are for a certain config (58 channels originally)
- For fine-tuning, they MODIFY the patch embedding to handle different channel counts
- For TUEV, they specifically configure it for 20 channels after mapping from 23

## The Fix Required

### Option 1: Modify EEGPT Architecture (CORRECT)
Configure EEGPT's patch embedding for 20 channels:
```python
# In our eegpt_architecture.py
model = EEGPTModel(
    img_size=[20, 1000],  # NOT [19, 1024] or whatever default!
    patch_size=patch_size,
    in_channels=20,       # After channel mapping
    ...
)
```

### Option 2: Skip Channel Mapping (SIMPLER BUT WRONG)
Just use 19 of the 23 channels directly - but this loses information!

## Implementation Path Forward

1. **Modify our EEGPT initialization** to accept configurable channel counts
2. **Set img_size=[20, 1000]** for TUEV specifically 
3. **Keep the 23→20 channel mapper** as we have it
4. **Pass 20 channels to EEGPT**, not 19!

## Why This Matters

The EEGPT model's patch embedding creates patches based on the configured image size. If we say img_size=[19, X] but pass 20 channels, the patch dimensions won't match and we get the error:
```
RuntimeError: The size of tensor a (16) must match the size of tensor b (32) at non-singleton dimension 2
```

This is because the model is creating patches for 19 channels but receiving 20!

## Files That Need Fixing

1. `experiments/eegpt_linear_probe/train_tuev_events.py` - Configure EEGPT for 20 channels
2. `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py` - Make channels configurable
3. `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py` - Pass through channel config

## Validation

After fixing, we should see:
- Model loads without errors
- Training starts with correct shapes
- Balanced accuracy approaching 62.32% as per paper
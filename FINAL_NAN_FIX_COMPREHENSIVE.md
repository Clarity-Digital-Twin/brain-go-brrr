# Comprehensive NaN Fix Implementation

## Root Cause Summary
The training crashed due to normalization statistics computed on the wrong dataset (Sleep-EDF), resulting in:
- Mean: 9.4e-09 
- Std: 2.1e-05
- This caused a 47,000x amplification of TUAB data, leading to NaN in mixed precision training

## Implementation Plan

### 1. Remove Broken Normalization (COMPLETED)
```bash
rm data/models/eegpt/pretrained/normalization.json
```

### 2. Add Guard Rails to Model
- Modified `eegpt_wrapper.py` to use identity normalization (mean=0, std=1) when no file exists
- Added warning log when using default normalization

### 3. Add Runtime Safeguards
The following safeguards need to be added:
- Input validation in dataset __getitem__
- Loss NaN detection in training_step
- Gradient clipping already enabled (1.0)
- Use fp32 precision instead of fp16

### 4. Update Configuration
- Already created `tuab_stable.yaml` with:
  - precision: 32 (no mixed precision)
  - learning_rate: 2e-4 (reduced from 5e-4)
  - accumulate_grad_batches: 2 (reduced from 4)
  - scheduler: cosine (instead of onecycle)
  - Removed label smoothing from loss function

### 5. Launch Script
Created `RUN_STABLE_TRAINING.sh` with all stability fixes
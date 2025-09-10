# TUEV Implementation: Master Documentation

**Status**: Implementation complete, training achieves BAC=0.19-0.24 (target: 0.62)  
**Last Updated**: September 10, 2025  
**Current Issue**: Severe class imbalance despite WeightedRandomSampler

## Table of Contents
1. [Critical Issues & Status](#critical-issues--status)
2. [What TUEV Is](#what-tuev-is)
3. [Current Implementation](#current-implementation)
4. [Reference Implementation](#reference-implementation)
5. [Key Divergences](#key-divergences)
6. [Training Commands](#training-commands)
7. [Troubleshooting](#troubleshooting)
8. [Implementation Files](#implementation-files)

## Critical Issues & Status

### 🔴 Issue #1: Class Imbalance (CURRENT BLOCKER)
- **Problem**: Class 0 (spsw) has 19/2695 samples (0.7%), Class 5 (bckg) has 1168/2695 (43%)
- **Impact**: Model collapses to predicting majority classes, BAC stuck at ~0.19
- **Attempted Fix**: WeightedRandomSampler added but insufficient
- **Status**: Need more aggressive balancing or different approach

### ✅ Issue #2: Channel Mismatch (FIXED)
- **Problem**: EEGPT configured for wrong channel count
- **Solution**: Configure with exactly 20 TUEV channels
- **Status**: Fixed in train_tuev_events.py

### ✅ Issue #3: Task Misunderstanding (FIXED)
- **Problem**: Implemented sliding windows instead of event-centered segments
- **Solution**: Extract 5s segments around annotated events only
- **Status**: Fixed via TUEVEventDataset

### ✅ Issue #4: WSL/tmux Stability (FIXED)
- **Problem**: Training hangs with num_workers>0 and pin_memory=True
- **Solution**: Use num_workers=0, pin_memory=False
- **Status**: Fixed with proper flags

## What TUEV Is

**Temple University EEG Events (TUEV)** - Multi-class classification of 6 epileptiform event types:
1. **spsw** (0): Spike and slow wave
2. **gped** (1): Generalized periodic epileptiform discharge  
3. **pled** (2): Periodic lateralized epileptiform discharge
4. **eyem** (3): Eye movement
5. **artf** (4): Artifact
6. **bckg** (5): Background

**Target Performance**: 62.32% ± 1.14% balanced accuracy (from EEGPT paper)

## Current Implementation

### Data Pipeline
```python
# Event-centered extraction (NOT sliding windows)
# Location: src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py
- Extract 5s @ 200Hz segments around annotated events
- Filter: 0.1-75 Hz bandpass + 50 Hz notch
- Output: (23 channels, 1000 samples) per segment
- Cache: 2695 train, 1048 eval segments
```

### Model Architecture
```python
# 23→20 channel mapping + EEGPT
# Location: experiments/eegpt_linear_probe/train_tuev_events.py

Input (23, 1000) → ChannelMapper → (20, 1000) → EEGPT → (4, 512) → Classifier → (6,)

# Key configurations:
- 20 TUEV channels for EEGPT
- Parity mode: time_steps=1000, patch_stride=64
- Fallback mode: pad to 1024
```

### Training Configuration
```python
# Hyperparameters (matching paper)
lr = 5e-4
weight_decay = 0.05
layer_decay = 0.65
warmup_epochs = 5
epochs = 30
batch_size = 32  # Effective 400 via accumulation
label_smoothing = 0.1
```

## Reference Implementation

From `reference_repos/EEGPT/downstream_tueg/`:

### Key Files
- `dataset_maker/make_TUEV.py`: Event extraction
- `utils.py`: TUEVLoader class
- `run_class_finetuning_EEGPT_change_tuev.py`: Training script
- `finetune_TUEV_EEGPT.sh`: Hyperparameters

### Critical Details
1. **NO bipolar montage** - uses referential channels
2. **23→20 mapping** via learned Conv2d
3. **Event-only segments** - no sliding windows
4. **Unweighted loss** with label smoothing=0.1
5. **Class labels**: Subtract 1 from original labels (1-6 → 0-5)

## Key Divergences

### What We Have Right ✅
- Event-centered 5s segments
- 200 Hz sampling rate
- 23→20 channel mapping
- Label smoothing, warmup, layer decay
- Referential (not bipolar) channels

### What's Still Wrong ❌
1. **Class imbalance handling**: WeightedRandomSampler insufficient
2. **Learning rate**: May need adjustment (try 1e-4 or 3e-4)
3. **Unknown**: Possible data normalization differences

## Training Commands

### Stable WSL Command (RECOMMENDED)
```bash
tmux new -d -s tuev "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && \
  CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  uv run python experiments/eegpt_linear_probe/train_tuev_events.py \
  --data_dir data/datasets/tuev \
  --eegpt_checkpoint data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt \
  --save_dir experiments/eegpt_linear_probe/output/tuev_$(date +%Y%m%d_%H%M%S) \
  --use_parity \
  --epochs 30 \
  --lr 5e-4 \
  --batch_size 32 \
  --num_workers 0 \
  --seed 42 \
  2>&1 | tee experiments/eegpt_linear_probe/logs/tuev_$(date +%Y%m%d_%H%M%S).log"

# Watch progress:
tmux attach -t tuev
```

**Environment Variables Explained**:
- `CUDA_LAUNCH_BLOCKING=1`: Synchronous CUDA execution for better error messages
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`: Prevents GPU memory fragmentation

### Monitor Training
```bash
# Watch live
tmux attach -t tuev

# Check progress
tmux capture-pane -t tuev -p | grep BAC

# Watch log file
tail -f experiments/eegpt_linear_probe/logs/tuev_*.log
```

## Troubleshooting

### If Training Hangs
```bash
# Kill everything
pkill -9 -f train_tuev_events
tmux kill-server

# Check GPU
nvidia-smi

# Clear GPU memory
nvidia-smi --gpu-reset
```

### Expected Progress
- Epoch 1: BAC ~0.17 (random baseline)
- Epoch 3: BAC >0.25 (learning signal)
- Epoch 10: BAC >0.40 (convergence starting)
- Epoch 30: BAC ~0.62 (target)

**Current Issue**: Stuck at BAC ~0.19 due to class imbalance

### Debug Class Distribution
```bash
# Check train distribution (if cache exists)
uv run python -c "import json; print(json.load(open('data/datasets/tuev/cache/tuev_event_segments/train/index.json'))['class_counts'])"

# Check eval distribution (if cache exists)
uv run python -c "import json; print(json.load(open('data/datasets/tuev/cache/tuev_event_segments/eval/index.json'))['class_counts'])"

# Actual distribution from our training:
# Train: {0: 19, 1: 715, 2: 282, 3: 185, 4: 326, 5: 1168} = 2695 total
# Class 0 (spsw): 0.7%, Class 5 (bckg): 43.3%
```

### Cache Management
```bash
# Check if cache exists
ls -la data/datasets/tuev/cache/tuev_event_segments/

# Rebuild cache if needed (takes ~30 minutes)
rm -rf data/datasets/tuev/cache/
uv run python -c "from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset; TUEVEventDataset('data/datasets/tuev', 'train')"

# Cache is NOT affected by training script changes
# Only rebuild if you change preprocessing/extraction logic
```

## Implementation Files

### Core Components (src/)
- `src/brain_go_brrr/infra/data/tuev_event_dataset.py` - Dataset class
- `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py` - Segment extraction
- `src/brain_go_brrr/infra/ml_models/channel_mapper.py` - 23→20 mapping
- `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py` - EEGPT interface

### Training Script (experiments/)
- `experiments/eegpt_linear_probe/train_tuev_events.py` - Main trainer

### Cache Location
- `data/datasets/tuev/cache/tuev_event_segments/` - Preprocessed segments

## Next Steps

### Immediate Actions Required
1. **Verify sampler is working**: Add batch label distribution logging
2. **Try more aggressive balancing**: Square the class weights
3. **Reduce learning rate**: Test 1e-4 instead of 5e-4
4. **Check reference exactly**: Any normalization we're missing?

### Diagnostic Code to Add
```python
# In training loop, log batch class distribution
batch_labels = y.cpu().numpy()
unique, counts = np.unique(batch_labels, return_counts=True)
print(f"Batch distribution: {dict(zip(unique, counts))}")
```

### If Still Stuck
1. Compare exact preprocessing with reference
2. Check if reference uses any data augmentation
3. Verify our label mapping (0-5 vs 1-6)
4. Consider focal loss instead of cross-entropy

---

## Archived Documentation

The following documents have been consolidated into this master file:
- TUEV_IMPLEMENTATION_PLAN.md
- TUEV_DIVERGENCE_ANALYSIS.md  
- TUEV_INVESTIGATION.md
- TUEV_CHANNEL_MISMATCH_ANALYSIS.md
- TUEV_TROUBLESHOOTING_GUIDE.md

Keep for historical reference but **THIS DOCUMENT IS THE SINGLE SOURCE OF TRUTH**.
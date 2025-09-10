# TUEV Implementation: Master Documentation

**Status**: All major fixes applied; ready for parity run (target: ~0.62 BAC)  
**Last Updated**: September 10, 2025

## Document Map
- This file (`TUEV.md`) is the Single Source of Truth for our implementation: how it works, how to run it, and how to validate it.
- Reference spec from the authors: `TUEV_REFERENCE.md` (read-only facts about their pipeline and hyperparameters).
- Differences and remediation plan: `TUEV_GAP_ANALYSIS.md` (tracks alignment vs. reference and any open deltas).

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

### ✅ Issue #0: WRONG DATA SPLITS (FIXED)
- **Problem**: Code was using wrong cache with seed=42 split
- **Solution**: Deleted wrong cache, now uses official train/eval directories
- **Impact**: Now using 359 train files, 159 eval files (correct splits)
- **Status**: FIXED - cache rebuilding with correct splits

### ✅ Issue #1: Class Imbalance Handling (FIXED)
- **Problem**: Severe imbalance - Class 0 (spsw) <1%, Class 5 (bckg) >40%
- **Solution**: Removed WeightedRandomSampler - reference achieves 62% BAC WITHOUT balancing
- **Impact**: Model will learn from natural distribution like reference
- **Status**: FIXED - using shuffle=True without sampler

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

### ✅ Split Fix Summary (Consolidated)
- We now use the official TUEV directory splits: `edf/train` and `edf/eval`.
- The cache builder scans those trees directly; no custom re-splitting is performed.
- Eval subject grouping uses the parent directory (`000–079`) to avoid label-derived subject names.
- Validation: run `scripts/validate_tuev_cache.py --data_dir data/datasets/tuev` and expect zero overlap.

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

### Channel Configuration (Single Source of Truth)

**23 Referential Input Channels (Extractor Order)**:
```python
['EEG FP1-REF','EEG FP2-REF','EEG F3-REF','EEG F4-REF','EEG C3-REF','EEG C4-REF',
 'EEG P3-REF','EEG P4-REF','EEG O1-REF','EEG O2-REF','EEG F7-REF','EEG F8-REF',
 'EEG T3-REF','EEG T4-REF','EEG T5-REF','EEG T6-REF','EEG A1-REF','EEG A2-REF',
 'EEG FZ-REF','EEG CZ-REF','EEG PZ-REF','EEG T1-REF','EEG T2-REF']
```

**20 Target Channels for EEGPT**:
```python
['FP1','FPZ','FP2','F7','F3','FZ','F4','F8',
 'T7','C3','CZ','C4','T8','P7','P3','PZ','P4','P8','O1','O2']
```
Note: `chan_ids` is a 1D tensor built from these names via `CHANNEL_DICT`.

### Normalization Behavior
EEGPTWrapper normalizes inputs using mean=0 and std=50μV by default if no stats file is provided. Datasets emit Volts (SI units). For TUEV paper parity, we DISABLE wrapper normalization and instead scale inputs to microvolts in the model (`x = x * 1e6`) so the backbone sees raw μV values like the reference. For production, compute corpus-level stats and pass them to the wrapper.

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
batch_size = 32  # Effective batch ≈ 400 via accumulation (e.g., batch_size=32 → 12 steps → 384)
label_smoothing = 0.1
```
Rule of thumb: choose gradient accumulation steps so `batch_size × steps ≈ 400`.

### Natural Sampling (NO BALANCING)
**CRITICAL**: Do NOT use WeightedRandomSampler! The reference achieves ~62% BAC with natural distribution.

The training script:
1. Reads class distribution from cache index for monitoring
2. Prints natural class_counts to show imbalance (this is expected)
3. Uses `shuffle=True` in DataLoader WITHOUT any sampler
4. Trusts the model to learn from natural distribution like the reference

### Key Fixes Summary (Consolidated)
- Data splits: Official `edf/train` and `edf/eval` only; stale cache removed and rebuilt
- Sampling: No `WeightedRandomSampler`; `shuffle=True` with natural class distribution
- Input scale: Wrapper normalization disabled; inputs scaled to μV before the mapper
- Pooling: Mean pooling to 512-dim head (matches reference)
- Batch: Effective batch ≈ 400 via gradient accumulation (logged)
- Regularization: DropPath implemented at rate 0.2 with per-layer decay (logged by model)

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
5. **Class labels**: Reference subtracts 1 (labels 1-6 → 0-5); we use explicit map {spsw:0, gped:1, pled:2, eyem:3, artf:4, bckg:5}, consistent with ref semantics

## Key Divergences

### What We Have Right ✅
- Event-centered 5s segments
- 200 Hz sampling rate
- 23→20 channel mapping
- Label smoothing, warmup, layer decay
- Referential (not bipolar) channels

### Recent Fixes Applied ✅
1. **Data splits**: Now using official train/eval directories (359/159 files)
2. **Sampling**: Removed WeightedRandomSampler, using natural distribution
3. **Normalization**: Disabled normalization, using raw μV like reference
4. **Mean pooling**: Enabled to match reference (512 features instead of 2048)
5. **Batch size**: Fixed to target 400 effective batch size
6. **DropPath**: Implemented stochastic depth with `drop_path_rate=0.2` (logs on model init)

### Validation Checklist
- Cache rebuilt with official splits: `edf/train` and `edf/eval`
- `cache/tuev_event_segments/train/index.json` exists; `n_segments > 0`
- `cache/tuev_event_segments/eval/index.json` exists; `n_segments > 0`
- Eval subject grouping uses parent directory (000–079); no overlap with train subjects
- Training logs include:
  - `Setting up training WITHOUT balanced sampling...`
  - `Normalization DISABLED - using raw values like reference`
  - `DropPath=0.2 enabled ...` (trainer) and `DropPath enabled: rate=0.2 ...` (model)
  - `Effective batch size: ... (batch=X, accum=Y)`
  - Per-epoch confusion matrix and per-class report

## Training Commands

### Stable WSL Command (RECOMMENDED)

**Use the provided launch script**:
```bash
./experiments/eegpt_linear_probe/scripts/launch_tuev_safe.sh
```

Or manually:
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

**WSL Note**: Our CLI default is `--pin_memory False`; do not pass `--pin_memory` on WSL.

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

# Clear GPU memory (if supported)
nvidia-smi --gpu-reset  # Note: GPU reset availability depends on driver/runtime
```

### Legacy MNE Warning During Cache Build
- Message: `NOTE: pick_channels() is a legacy function. New code should use inst.pick(...)`
- Cause: Some preprocessors still call `raw.pick_channels(...)`.
- Impact: Benign; does not affect extracted segments or training.
- Planned fix: Migrate to `inst.pick(...)` or route through our `mne_compat.pick_channels` helper post‑parity.

### Expected Progress (Acceptance Gates)
By epoch 2-3: BAC ≥ 0.25; by epoch 5: BAC ≥ 0.40; by epoch 10-12: if BAC < 0.30, collect confusion matrix and per-split class distributions, then revise sampling/LR; final target: 0.62 ± 0.02 by epoch ~30.

**Current Observation (latest run)**: Eval BAC ≈ 0.24 at ~epoch 20; strong bckg/gped, weak spsw/pled/eyem/artf. This suggests under-learning of rare classes despite parity settings.

### Guardrails
- **No PyTorch Lightning** (CI guard in place)
- **torch.load safety**: Cache files loaded with `weights_only=True`; EEGPT checkpoint loaded via safe loader with explicit `# nosec:weights_only` justification
- **chan_ids shape**: Do not batch `chan_ids`. EEGPT expects a 1D tensor of length 20; input to EEGPT must be shaped `(B, 20, 1000)` after the mapper. Ensure the mapper squeezes the spatial dimension so it outputs `(B, 20, 1000)` before EEGPT.

### If BAC Stalls (<0.30 by epoch 10–12)
- Confirm logs include: natural sampling, normalization disabled, DropPath enabled, effective batch, parity mode.
- Exact effective batch 400: try `--batch_size 40` (accum=10) for remaining epochs or in a short follow-up run.
- Diagnostic scale A/B (debug-only): keep μV scaling, then toggle wrapper normalization on for 5 epochs to test sensitivity to scale; pick the better setting for the next full run.
- Input sanity log: print min/median/max of `x*1e6` just before mapper on a batch to verify μV magnitudes.
- Optional diagnostic: Consider `patch_size=50, patch_stride=50` at 200 Hz (restores ~0.25 s per patch). This deviates from the checkpoint kernel (1×64), so treat strictly as a diagnostic ablation, not the final parity run.

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

### Evaluation Output
- Each evaluation prints a confusion matrix and per-class `classification_report`. Use per-class recall to detect systematic misses on rare classes (e.g., spsw/gped/pled).

### Cache Management
```bash
# Check if cache exists
ls -la data/datasets/tuev/cache/tuev_event_segments/

# Rebuild cache if needed (takes ~5-30 minutes depending on disk speed and I/O)
rm -rf data/datasets/tuev/cache/
uv run python -c "from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset; TUEVEventDataset('data/datasets/tuev', 'train')"

# Training will also build eval cache on first use. To pre-build both:
# TUEVEventDataset('data/datasets/tuev', 'train') and TUEVEventDataset('data/datasets/tuev', 'eval')

# Cache is NOT affected by training script changes
# Only rebuild if you change preprocessing/extraction logic
```

## Implementation Files

### Core Components (src/)
- `src/brain_go_brrr/infra/data/tuev_event_dataset.py` - Dataset class
- `src/brain_go_brrr/infra/preprocessing/tuev_event_extractor.py` - Segment extraction
- `src/brain_go_brrr/infra/ml_models/channel_mapper.py` - 23→20 mapping
- `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py` - EEGPT interface

### Training Scripts (experiments/)
- `experiments/eegpt_linear_probe/train_tuev_events.py` - Main trainer
- `experiments/eegpt_linear_probe/scripts/launch_tuev_safe.sh` - Safe launch script

### Utility Scripts
- `scripts/data/verify_tuev_dataset.py` - Verify dataset integrity
- `scripts/testing/debug_tuev_training.py` - Debug training issues

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

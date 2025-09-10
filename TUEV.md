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

### 🔴 Issue #-1: MISSING WEIGHT NORMALIZATION (CRITICAL)
- **Problem**: Using nn.Linear(30720, 6) instead of LinearWithConstraint
- **Solution**: Must use LinearWithConstraint with max_norm=1 
- **Impact**: Without weight renorm, training collapses to majority classes
- **Status**: NOT FIXED - THIS IS THE SMOKING GUN

### 🔴 Issue #-2: CHANNEL MAPPER ARCHITECTURE WRONG
- **Problem**: Simple Conv2d(23, 20, 1) vs complex pipeline with constraints
- **Solution**: Conv2dWithConstraint + BatchNorm + GELU + DepthwiseConv + BatchNorm + Dropout(0.8)
- **Impact**: Missing regularization and normalization in channel mapping
- **Status**: NOT FIXED - Another critical divergence

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

**IMPLEMENTED (CORRECT) ✅**:
```python
# Full parity with reference implementation
Input (23, 1000) → [Conv2dWithConstraint(23→20) → BatchNorm → GELU → 
                    DepthwiseConv(1,55) → BatchNorm → Dropout(0.8)] →
                   EEGPT → Flatten(30720) → Dropout(0.8) → LinearWithConstraint(30720→6, max_norm=1)
```

**Key Components (ALL IMPLEMENTED)**:
1. **LinearWithConstraint**: Weight renormalization (max_norm=1) prevents gradient explosion ✅
2. **Conv2dWithConstraint**: Channel mapper with weight constraints ✅
3. **BatchNorm layers**: Stabilize channel mapping ✅
4. **DepthwiseConv(1,55)**: Additional temporal processing in mapper ✅
5. **Dropout(0.8) in mapper**: Extra regularization before EEGPT ✅
6. **Per-iteration LR scheduling**: Cosine annealing every step, not epoch ✅
7. **timm.loss.LabelSmoothingCrossEntropy**: Exact reference loss function ✅

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
- Temporal tokens: Flatten ALL temporal summary tokens (30720) with Dropout(0.8) → Linear(6) (matches reference)
- Batch: Effective batch ≈ 400 via gradient accumulation (logged)
- Regularization: DropPath implemented at rate 0.2 with per-layer decay (logged by model)

### Event Extraction Specifics (Parity Alignment)
- Bandpass 0.1–75 Hz; notch at 50 Hz; resample to 200 Hz.
- Referential channels only; reorder to the 23-channel standard order.
- Segmenting: use annotation start/end; slice [start−2s : end+2s] on a threefold-extended signal buffer to avoid boundary drops; segments are exactly 5 s (1000 samples).

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

### Recent Fixes Applied ✅ (Sep 10, 2025)

#### Critical Fixes (Resolved Training Collapse)
1. **LinearWithConstraint in classifier head** 🔴 CRITICAL
   - Added weight renormalization with max_norm=1.0
   - Prevents gradient explosion with 30,720 input features
   - This was THE smoking gun causing collapse to majority classes

2. **timm.loss.LabelSmoothingCrossEntropy** 
   - Switched from custom implementation to timm's version
   - Exact match with reference implementation

3. **Per-iteration LR scheduling**
   - Implemented cosine scheduler updating every iteration
   - Replaced per-epoch scheduling for smoother optimization

4. **Layer-wise LR decay verification**
   - Confirmed proper application with logging
   - Deeper layers receive lower learning rates as expected

#### Previously Fixed Issues
5. **Data splits**: Now using official train/eval directories (4213/1471 segments)
6. **Sampling**: Removed WeightedRandomSampler, using natural distribution
7. **Normalization**: Disabled wrapper normalization, using raw μV like reference
8. **Temporal tokens**: Using ALL temporal tokens (30720) instead of 4‑token pooling
9. **Batch size**: Fixed to target 400 effective batch size (40×10 accumulation)
10. **DropPath**: Implemented stochastic depth with `drop_path_rate=0.2`
11. **Channel mapper**: Full parity with Conv2dWithConstraint + depthwise conv pipeline

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
  - `Using TEMPORAL TOKEN FLATTENING: 15×4×512 = 30720`
  - Per-epoch confusion matrix and per-class report

## 🔴 CRITICAL: Data Structure Requirements

### CORRECT Directory Structure (MUST MATCH EXACTLY):
```
data/datasets/tuev/
├── edf/
│   ├── train/     # 359 .edf files
│   └── eval/      # 159 .edf files  
├── cache/
│   └── tuev_event_segments/
│       ├── train/
│       │   ├── index.json    # Must show 4213 segments
│       │   └── *.pkl         # 4213 pickle files
│       └── eval/
│           ├── index.json    # Must show 1471 segments
│           └── *.pkl         # 1471 pickle files
```

### Common Path Issues and Fixes:
1. **WRONG**: `--data_dir data/datasets/tuev/raw` ❌ (no /raw subdirectory!)
2. **CORRECT**: `--data_dir data/datasets/tuev` ✅
3. **Cache dir**: `--cache_dir data/datasets/tuev/cache` ✅

### Verify Cache Before Training:
```bash
# Check train cache
ls data/datasets/tuev/cache/tuev_event_segments/train/*.pkl | wc -l  # Should be 4213
# Check eval cache  
ls data/datasets/tuev/cache/tuev_event_segments/eval/*.pkl | wc -l   # Should be 1471

# If eval cache is 0, rebuild it:
uv run python -c "
from pathlib import Path
from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
eval_dataset = TUEVEventDataset(
    root_dir=Path('data/datasets/tuev'),
    split='eval',
    cache_dir=Path('data/datasets/tuev/cache'),
    force_rebuild=True
)
print(f'Rebuilt eval cache: {len(eval_dataset)} segments')
"
```

## Training Commands

### CORRECT Training Command:
```bash
tmux new -d -s tuev "cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr && \
  uv run python experiments/eegpt_linear_probe/train_tuev_events.py \
  --data_dir data/datasets/tuev \
  --cache_dir data/datasets/tuev/cache \
  --eegpt_checkpoint data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt \
  --save_dir experiments/eegpt_linear_probe/output/tuev_$(date +%Y%m%d_%H%M%S) \
  --use_parity \
  --epochs 30 \
  --batch_size 40 \
  --num_workers 0"
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

## 🚨 Troubleshooting

### CRITICAL PATH ISSUES (MOST COMMON FAILURES)

#### Issue: "Loaded eval cache: 0 segments" or ZeroDivisionError
**Root Cause**: Eval pickle files missing or wrong data path
```bash
# Check if pickle files exist
ls data/datasets/tuev/cache/tuev_event_segments/eval/*.pkl | wc -l

# If 0, rebuild eval cache with CORRECT path:
uv run python -c "
from pathlib import Path
from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
eval_dataset = TUEVEventDataset(
    root_dir=Path('data/datasets/tuev'),  # NO /raw!
    split='eval',
    cache_dir=Path('data/datasets/tuev/cache'),
    force_rebuild=True
)
print(f'Rebuilt: {len(eval_dataset)} segments')
"
```

#### Issue: Wrong data_dir causes empty cache
**WRONG**: `--data_dir data/datasets/tuev/raw` ❌
**CORRECT**: `--data_dir data/datasets/tuev` ✅

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

## COMPLETE FINDINGS SUMMARY - AWAITING SENIOR AUDIT

### 🔴 CRITICAL ISSUES (Training Collapse Root Causes)
1. **LinearWithConstraint Missing**: 30,720→6 head needs weight renorm (max_norm=1)
2. **Channel Mapper Wrong**: Missing Conv2dWithConstraint, BatchNorm, GELU, DepthwiseConv, Dropout(0.8)

### 🟡 MAJOR ISSUES (Performance Impact)
3. **No Cosine LR Schedule**: Reference uses cosine annealing for LR (5e-4→1e-6)
4. **No Cosine WD Schedule**: Reference also anneals weight decay!
5. **Wrong Loss Function**: Using custom instead of timm.loss.LabelSmoothingCrossEntropy

### ✅ ALREADY FIXED
- Temporal tokens (30,720 features) ✅
- Boundary handling (triple concat) ✅
- Natural sampling ✅
- μV scale ✅
- Exact batch 400 ✅
- DropPath 0.2 ✅

### 🚨 IMPLEMENTATION REQUIRED (DO NOT START YET - AWAIT AUDIT)
```python
# 1. LinearWithConstraint for head
class LinearWithConstraint(nn.Linear):
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=1)
        return super().forward(x)

# 2. Conv2dWithConstraint for mapper  
class Conv2dWithConstraint(nn.Conv2d):
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=1)
        return super().forward(x)

# 3. Complete channel mapper
self.chan_conv = nn.Sequential(
    Conv2dWithConstraint(23, 20, 1, max_norm=1),
    nn.BatchNorm2d(20),
    nn.GELU(),
    nn.Conv2d(20, 20, kernel_size=(1,55), groups=20, padding='same'),
    nn.BatchNorm2d(20),
    nn.Dropout(0.8)
)

# 4. Cosine schedulers
lr_schedule = cosine_scheduler(5e-4, 1e-6, epochs=30, warmup=5)
wd_schedule = cosine_scheduler(0.05, 0.05, epochs=30)
```

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

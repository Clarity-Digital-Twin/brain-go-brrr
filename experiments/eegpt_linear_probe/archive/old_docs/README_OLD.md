# EEGPT Linear Probe Training

## 🚨 CRITICAL: Only ONE Working Script!

### ✅ THE ONLY WORKING SCRIPT

**`train_pytorch_nan_safe.py`** - **THIS IS THE ONLY RELIABLE TRAINING SCRIPT!**
- Pure PyTorch with FULL NaN protection
- Gradient clipping, anomaly detection, input validation
- Currently running at 62.5% accuracy
- Launch with: `./launch_nan_safe_training.sh`
- **DO NOT USE ANYTHING ELSE!**

### ❌ ALL OTHER SCRIPTS ARE BROKEN

1. **`train_pytorch_stable.py`** (ARCHIVED - CRASHED OVERNIGHT!)
   - NO NaN protection
   - Crashed after 171 batches (0.6% of epoch 0)
   - Wasted entire night of compute
   - Now in `archive/crashed_overnight/`

2. **ALL PyTorch Lightning scripts** (ARCHIVED)
   - Lightning 2.5.2 HANGS FOREVER on large datasets
   - Cannot be fixed with ANY settings
   - Wasted DAYS debugging this shit
   - All in `archive/lightning_broken/`

### 📁 What's in This Folder

**Working Files:**
- `train_pytorch_nan_safe.py` - The ONLY training script to use
- `launch_nan_safe_training.sh` - Launch script
- `custom_collate_fixed.py` - Handles variable channels
- `inference_example.py` - Example inference code
- `configs/` - Configuration files
- Documentation (`.md` files)

**Archive (DO NOT USE):**
- `archive/crashed_overnight/` - Scripts that crashed during training
- `archive/lightning_broken/` - All Lightning-based scripts
- `archive/failed_attempts/` - Various broken experiments
- See `archive/ARCHIVE_README.md` for why each is broken

## 🎯 Current Training Status

```bash
# Monitor current training
tmux attach -t eegpt_nan_safe

# Quick status check
tail -f logs/nan_safe_training_*/training.log
```

## 🛡️ NaN Protection Features

The `train_pytorch_nan_safe.py` includes:

1. **Input validation** - Checks every batch for NaN/Inf
2. **Gradient clipping** - Prevents gradient explosions (clip_norm=1.0)
3. **Learning rate warmup** - Starts at 2e-6, warms up over 5 epochs
4. **Anomaly detection** - PyTorch's autograd anomaly mode on first batch
5. **Safe operations** - All log/div operations protected with clamps
6. **Checkpoint validation** - Ensures saved models don't contain NaN

## Overview

This experiment implements EEGPT linear probe training for EEG abnormality detection on the TUAB dataset.

### Architecture
- **Backbone**: Frozen EEGPT (25.3M parameters)
- **Probe**: Two-layer MLP with dropout (34.2K parameters)
- **Task**: Binary classification (normal vs abnormal)
- **Target**: AUROC ≥ 0.93

### Dataset
- **TUAB**: TUH Abnormal EEG Corpus v3.0.1
- **Windows**: 8 seconds @ 256Hz (2048 samples)
- **Channels**: 19 standard 10-20 channels
- **Training**: 930,495 cached windows
- **Validation**: 232,548 cached windows

### Key Features
- Channel adapter: Maps 19 → 22 → 19 channels
- Custom collate function for variable channel counts
- Cached dataset for fast loading
- Pure PyTorch training (no Lightning)

## 📁 Directory Structure

```
experiments/eegpt_linear_probe/
├── train_pytorch_nan_safe.py    # ✅ USE THIS - NaN-safe training
├── train_pytorch_stable.py      # ✅ Works - basic PyTorch version
├── launch_nan_safe_training.sh  # ✅ Launch script for nan-safe
├── launch_training.sh           # Launch script for stable version
├── custom_collate_fixed.py      # ✅ Handles variable channels
├── inference_example.py         # ✅ Example inference code
├── configs/                     # Training configurations
│   ├── tuab_nan_safe.yaml      # ✅ Safe configuration
│   ├── tuab_stable.yaml        # Basic config
│   └── tuab_cached.yaml        # Dataset caching config
├── archive/                     # ❌ BROKEN SCRIPTS
│   └── lightning_broken/
│       └── train_enhanced.py    # ❌ Lightning version - HANGS
├── logs/                        # Training logs
├── output/                      # Model checkpoints
├── LIGHTNING_BUG_REPORT.md      # Critical bug documentation
└── CHANNEL_MAPPING_EXPLAINED.md # Channel naming details
```

## Performance

Training on RTX 4090:
- Speed: ~2.4 iterations/second
- Memory: ~12GB VRAM
- Time: ~10 hours for 50 epochs

## Results

Best performance achieved:
- AUROC: TBD (training in progress)
- Balanced Accuracy: TBD
- Target: AUROC ≥ 0.93 (from paper)

## 🐛 Known Issues & Fixes

### 1. PyTorch Lightning Hanging
- **Issue**: Lightning 2.5.2 hangs with datasets >100k samples
- **Fix**: Use pure PyTorch implementation

### 2. NaN Loss
- **Issue**: Training explodes to NaN after a few batches
- **Fix**: Reduced learning rate, gradient clipping, warmup schedule

### 3. persistent_workers Error
- **Issue**: `persistent_workers=True` with `num_workers=0` causes hang
- **Fix**: Set `num_workers=2` and `persistent_workers=True`

### 4. Import Errors
- **Issue**: Wrong import paths after reorganization
- **Fix**: Use absolute imports from project root

## 🔍 Debugging Commands

```bash
# Check for NaN in logs
grep -i nan logs/nan_safe_training_*/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training speed (should be ~10 it/s)
tail -f logs/nan_safe_training_*/training.log | grep -o "[0-9.]*it/s"

# View full training progress
tmux attach -t eegpt_nan_safe
```

## ⚠️ Important Notes

1. **DO NOT USE PYTORCH LIGHTNING** - It has unfixable bugs with large datasets
2. **Always monitor first few batches** for NaN/gradient issues
3. **Keep num_workers > 0** to avoid dataloader hangs
4. **Use the cached dataset** for fast loading (930k training samples)
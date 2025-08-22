# EEGPT Linear Probe Training

This directory contains the training pipelines for EEGPT-based linear probes on TUAB (abnormality detection) and TUEV (event detection) datasets.

## Overview

We train lightweight linear probes on top of frozen EEGPT features for two clinical EEG tasks:
- **TUAB**: Binary classification (normal vs abnormal EEG)
- **TUEV**: 6-class event detection (seizure types and patterns)

## Current Status

✅ **Ready for Training** - All critical bugs fixed, using 100% of EEGPT features (32,768 dimensions for TUAB)

### Expected Performance
| Dataset | Metric | Target | Status |
|---------|--------|--------|--------|
| TUAB | AUROC | 0.87 | Training |
| TUEV | BAcc | 0.62 | Pending |

## Quick Start

### 1. Train TUAB (Abnormality Detection)
```bash
cd experiments/eegpt_linear_probe
./scripts/launch_tuab.sh

# Monitor progress
tmux attach -t tuab_training
```

### 2. Train TUEV (Event Detection)
```bash
# After TUAB completes
./scripts/launch_tuev.sh

# Monitor progress
tmux attach -t tuev_training
```

## Directory Structure

```
eegpt_linear_probe/
├── README.md                # This file
├── train_tuab.py           # TUAB training script
├── train_tuev.py           # TUEV training script
├── tuab_dataset.py         # TUAB dataset loader (memory-mapped)
├── tuev_dataset.py         # TUEV dataset loader
├── tuev_dataset_cached.py  # TUEV cached variant
├── custom_collate_fixed.py # Collate function for dataloaders
├── configs/
│   ├── tuab.yaml          # TUAB configuration (4s windows, batch=100)
│   └── tuev.yaml          # TUEV configuration (10s windows, batch=500)
├── scripts/
│   ├── launch_tuab.sh     # TUAB launch script
│   ├── launch_tuev.sh     # TUEV launch script
│   ├── build_tuev_cache.py      # Build TUEV cache (2048 samples)
│   └── build_tuev_cache_1024.py # Build TUEV cache (1024 samples)
├── logs/                   # Training logs
├── output/                 # Model checkpoints and results
└── archive/               # Old experiments and documentation

```

## Technical Details

### EEGPT Feature Extraction
- **Model**: EEGPT Large (10M parameters)
- **Checkpoint**: `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Input**: 20 EEG channels @ 256Hz
- **Patch Size**: 64 samples (250ms)
- **Features**: All temporal patches × summary tokens × embedding dimension

### TUAB Specifications
- **Window Size**: 4 seconds (1024 samples)
- **Features**: 16 patches × 4 tokens × 512 dims = 32,768
- **Classes**: 2 (normal/abnormal)
- **Dataset**: ~1.86M training windows
- **Batch Size**: 100 (paper-aligned)

### TUEV Specifications
- **Window Size**: 10 seconds (2560 samples) 
- **Features**: 40 patches × 4 tokens × 512 dims = 81,920
- **Classes**: 6 (background, seizure types, patterns)
- **Dataset**: ~264k training windows
- **Batch Size**: 500 (paper-aligned)

## Training Configuration

Both models use:
- **Optimizer**: AdamW with weight decay
- **Scheduler**: OneCycleLR
- **Probe**: 2-layer MLP with LazyLinear
- **Early Stopping**: Patience of 10 epochs
- **Validation**: Every 2 epochs

## Monitoring Training

```bash
# View active training sessions
tmux ls

# Attach to session
tmux attach -t tuab_training

# Detach from session
Ctrl+B, then D

# Check logs
tail -f logs/tuab_training_*.log
```

## Results

Training results are saved to:
- `output/tuab_*/best_model.pt` - Best checkpoint
- `output/tuab_*/history.json` - Training metrics
- `logs/tuab_training_*.log` - Full training log

## Citation

Based on EEGPT paper:
```
Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). 
EEG conformer: Convolutional transformer for EEG decoding and visualization. 
IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31, 710-719.
```
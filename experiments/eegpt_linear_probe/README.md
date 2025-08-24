# EEGPT Linear Probe Training

This directory contains the training pipelines for EEGPT-based linear probes on TUAB (abnormality detection) and TUEV (event detection) datasets.

## Overview

We train lightweight linear probes on top of frozen EEGPT features for two clinical EEG tasks:
- **TUAB**: Binary classification (normal vs abnormal EEG)
- **TUEV**: 6-class event detection (seizure types and patterns)

## Current Status

🔄 **Active Training** - TUAB abnormality detection
- Progress: ~33% complete (9,600/29,143 batches)
- Speed: ~1.4 it/s
- Monitor: `tmux attach -t tuab_training`

### Expected Performance
| Dataset | Metric | Target | Status |
|---------|--------|--------|--------|
| TUAB | AUROC | 0.87 | Training (33%) |
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
├── train_tuab.py              # TUAB training script (ACTIVE)
├── train_tuev.py              # TUEV training script
├── tuab_dataset.py            # TUAB memory-mapped dataset
├── tuev_dataset.py            # TUEV dataset loader
├── tuev_dataset_cached.py    # TUEV cached variant
├── configs/
│   ├── tuab.yaml             # TUAB config (4s windows, batch=64)
│   └── tuev.yaml             # TUEV config (10s windows, batch=500)
├── scripts/
│   ├── launch_tuab.sh        # TUAB training launcher
│   ├── launch_tuev.sh        # TUEV training launcher
│   ├── monitor_training.sh   # Training progress monitor
│   ├── build_tuev_cache.py  # TUEV cache builder
│   └── deployment/           # Deployment utilities
├── utils/
│   └── custom_collate_fixed.py  # Batch collation
├── logs/                      # Training logs
├── output/                    # Model checkpoints
└── __pycache__/              # Python cache

```

## Technical Details

### EEGPT Feature Extraction
- **Model**: EEGPT Large (10M parameters)
- **Checkpoint**: `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Input**: 20 EEG channels @ 256Hz
- **Patch Size**: 64 samples (250ms)
- **Features**: 4 summary tokens × 512 dims → flattened to 2,048 for probes

### TUAB Specifications
- **Window Size**: 4 seconds (1024 samples @ 256Hz)
- **Probe Features**: 2,048 dims (4 summary tokens × 512, flattened)
- **Classes**: 2 (normal/abnormal)
- **Dataset**: ~1.86M training windows
- **Batch Size**: 64 (reduced for memory stability on WSL)

### TUEV Specifications
- **Window Size**: 10 seconds (2560 samples)
- **Probe Features**: 2,048 dims (4 summary tokens × 512, flattened)
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

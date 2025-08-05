# EEGPT Linear Probe Training

## Overview

This directory contains the implementation for training linear probes on top of frozen EEGPT features for EEG abnormality detection using the TUAB dataset.

**Current Status**: Training active with 8s windows, achieving ~0.68 AUROC (target: >0.85)

## Quick Start

```bash
# 1. Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# 2. Run training
bash RUN_TRAINING_8S.sh

# 3. Monitor progress
tmux attach -t eegpt_training
```

## Key Files

### Active Training Scripts
- `train_paper_aligned.py` - Main training script (pure PyTorch)
- `RUN_TRAINING_8S.sh` - Launch script for 8-second windows
- `smoke_test_paper_aligned.py` - Quick validation before training

### Configuration
- `configs/tuab_4s_paper_aligned.yaml` - Target paper-aligned config
- `configs/tuab_8s_temp.yaml` - Current 8s window config
- `configs/tuab_stable.yaml` - Stable training config

### Utilities
- `custom_collate_fixed.py` - Handles variable channel counts
- `build_tuab_4s_cache.py` - Creates 4-second window cache (pending)

### Documentation
- `SETUP_COOKBOOK.md` - Complete setup guide and templates
- `ISSUES_AND_FIXES.md` - All problems encountered and solutions
- `TRAINING_STATUS.md` - Current training progress

## Architecture

```
EEGPT (Frozen) → Linear Probe → Binary Classification
     ↓                ↓
  512-dim         2 classes
 features      (normal/abnormal)
```

## Current Performance

| Metric | Current | Target | Paper |
|--------|---------|--------|-------|
| AUROC  | 0.683   | >0.85  | 0.87  |
| Window | 8s      | 4s     | 4s    |
| Epochs | 5/200   | -      | 200   |

## Next Steps

1. **Complete current 8s training** (2-3 hours)
2. **Build 4s window cache** using `build_tuab_4s_cache.py`
3. **Run paper-aligned training** with exact specifications
4. **Fine-tune if needed** (unfreeze last transformer block)

## Directory Structure

```
├── configs/          # Training configurations
├── output/           # Training outputs and checkpoints
├── logs/             # Training logs (archived)
├── archive/          # Old scripts and failed attempts
├── *.py              # Active training scripts
└── *.md              # Documentation
```

## Known Issues

1. **PyTorch Lightning 2.5.2**: Hangs with large datasets - use pure PyTorch
2. **Path Resolution**: Manual handling of ${BGB_DATA_ROOT} required
3. **Channel Variability**: Some files have 19 channels, others 20

See `ISSUES_AND_FIXES.md` for complete list and solutions.

## References

- EEGPT Paper: [Large Brain Model for Learning Generic Representations](https://arxiv.org/abs/2312.14406)
- TUAB Dataset: Temple University Abnormal EEG Corpus v3.0.1
- Target Performance: AUROC ≥ 0.87 for abnormality detection
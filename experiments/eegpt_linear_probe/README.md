# EEGPT Linear Probe Training

## 🚨 CRITICAL: PyTorch Lightning Bug Warning

**DO NOT USE PyTorch Lightning for training!** Lightning 2.5.2 has a critical bug that causes training to hang indefinitely at "Loading train_dataloader to estimate number of stepping batches" with large cached datasets (>100k samples).

See [LIGHTNING_BUG_REPORT.md](LIGHTNING_BUG_REPORT.md) for full details.

## ✅ Working Training Script

Use **`train_pytorch_stable.py`** - Pure PyTorch implementation that works perfectly.

```bash
# Launch training
./launch_training.sh

# Monitor progress
tmux attach -t eegpt_training
```

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

## Files

### Core Training
- `train_pytorch_stable.py` - ✅ WORKING training script
- `launch_training.sh` - Professional launch script
- `custom_collate_fixed.py` - Handles channel padding

### Configuration
- `configs/tuab_stable.yaml` - Training configuration
- `configs/tuab_cached.yaml` - Dataset caching config

### Documentation
- `LIGHTNING_BUG_REPORT.md` - Critical bug documentation
- `CHANNEL_MAPPING_EXPLAINED.md` - Channel naming details

### Legacy (DO NOT USE)
- `train_enhanced.py` - ❌ BROKEN - Uses Lightning, will hang

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
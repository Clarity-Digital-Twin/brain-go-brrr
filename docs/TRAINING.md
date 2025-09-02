# Training Guide

## Overview

This guide covers training linear probes on frozen EEGPT features for EEG classification tasks.

## Prerequisites

### 1. Download EEGPT Pretrained Weights

**Official EEGPT Model** (Required):
- **Download from**: [Figshare](https://figshare.com/s/e37df4f8a907a866df4b)
- **File path**: `Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Place at**: `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Size**: ~40MB
- **Architecture**: 10M parameters, Vision Transformer
- **Training**: 58 channels, 256Hz, 4s windows

```bash
# Create directory and download
mkdir -p data/models/pretrained
# Download manually from Figshare link above
# Or use wget if you have direct link
```

### 2. Obtain EEG Datasets

#### TUAB Dataset (Abnormality Detection)
- **Source**: [Temple University Hospital EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
- **Dataset**: TUH Abnormal EEG Corpus v2.0.0
- **Request access**: Follow instructions on website
- **Place at**: `data/datasets/tuab/`
- **Size**: ~120GB compressed

#### TUEV Dataset (Event Detection)
- **Source**: [Temple University Hospital EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
- **Dataset**: TUH EEG Events v2.0.0
- **Request access**: Academic agreement required
- **Place at**: `data/datasets/tuev/`
- **Size**: ~60GB compressed

#### Sleep-EDF Dataset (Sleep Staging - Optional)
- **Source**: [PhysioNet Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Location**: Configured via DataConfig (default: `data/datasets/sleep-edf/`)
- **Size**: 197 PSG recordings

### 3. Environment Setup

```bash
# Install dependencies with uv
uv sync

# Verify PyTorch and CUDA
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## TUAB Training (Abnormality Detection)

### IMPORTANT: Feature Extraction Details

EEGPT outputs 4 summary tokens of 512 dimensions each. For linear probing:
- **Correct approach**: Flatten all 4 tokens → 2,048 features total
- **NOT**: Use temporal patches (would be 32,768 - way too many)
- **NOT**: Average the 4 tokens to 512 (loses information)

### Quick Start

```bash
cd experiments/eegpt_linear_probe
./scripts/launch_tuab.sh
```

### Monitor Progress

```bash
# Attach to training session
tmux attach -t tuab_training

# Check logs
tail -f logs/tuab_training_*.log
```

### Configuration

**File**: `configs/tuab.yaml`

```yaml
data:
  batch_size: 256        # Optimized for WSL/GPU
  window_duration: 4.0   # 4-second windows
  sampling_rate: 256     # Hz

training:
  max_epochs: 10
  learning_rate: 1e-3
  early_stopping:
    patience: 3

model:
  probe:
    hidden_dim: 256
    dropout: 0.25
    n_classes: 2  # Normal vs Abnormal
```

### Expected Performance

- **Target AUROC**: 0.87 (paper performance)
- **Training Time**: ~3-4 hours on GPU
- **Memory Usage**: ~8GB GPU RAM

### Dataset Details

- **Windows**: ~1.86M training samples
- **Classes**: Binary (0=normal, 1=abnormal)
- **Features**: 2,048 dimensions (4 summary tokens × 512, flattened)

## TUEV Training (Event Detection)

### Quick Start

```bash
cd experiments/eegpt_linear_probe
./scripts/launch_tuev.sh
```

### Configuration

**File**: `configs/tuev.yaml`

```yaml
data:
  batch_size: 500      # Paper-aligned
  window_duration: 10  # 10-second windows
  sampling_rate: 250   # Hz

training:
  max_epochs: 100
  learning_rate: 5e-4  # Constant LR

model:
  channel_adapter:
    in_channels: 23    # TUEV has 23 channels
    out_channels: 20   # EEGPT expects 20
    kernel_size: 55    # Table 13 architecture
    dropout: 0.5       # Higher dropout for TUEV
```

### Classes

1. SPSW - Spike-and-slow-wave
2. GPED - Generalized periodic epileptiform discharge
3. PLED - Periodic lateralized epileptiform discharge
4. EYEM - Eye movement
5. ARTF - Artifact
6. BCKG - Background

### Expected Performance

- **Target Balanced Accuracy**: 0.62
- **Weighted F1**: 0.82
- **Cohen's Kappa**: 0.64

## Training Scripts

### Main Training Script

```python
# experiments/eegpt_linear_probe/train_tuab_mne.py

# Key components:
1. Uses TUABDataset from src/brain_go_brrr/infra/data/
2. EEGPTWrapper from src/brain_go_brrr/infra/ml_models/
3. Frozen EEGPT backbone (no gradient updates)
4. OneCycleLR scheduler for optimal convergence
5. AUROC tracking for binary classification
```

### Dataset Usage

```python
# Training scripts use datasets from src/brain_go_brrr/infra/data/
from brain_go_brrr.infra.data.tuab_dataset import TUABDataset
from brain_go_brrr.infra.data.tuev_dataset import TUEVDataset

# Features:
# - MNE-based processing
# - Automatic channel validation
# - Built-in caching
# - Correct channel ordering enforcement
```

## Monitoring Training

### Metrics to Watch

1. **Loss**: Should decrease steadily
2. **AUROC**: Main metric for TUAB (target: 0.87)
3. **Balanced Accuracy**: Main metric for TUEV (target: 0.62)
4. **Learning Rate**: OneCycle schedule

### Tensorboard (Not Configured)

Currently using text logs only. Tensorboard can be added if needed.

## Troubleshooting

### WSL Issues

```python
# Force single-threaded DataLoader for WSL
num_workers=0
pin_memory=False
```

### GPU Memory

If OOM errors:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training

### Slow Training

1. Check disk I/O (use SSD for cache)
2. Verify GPU utilization (`nvidia-smi`)
3. Profile DataLoader bottlenecks

## Output Files

```
output/tuab_TIMESTAMP/
├── best_model.pt       # Best checkpoint
├── config.yaml         # Training configuration
├── history.json        # Training metrics
└── final_model.pt      # Last checkpoint
```

## Using Trained Models

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint = torch.load("output/tuab_*/best_model.pt")

# Load probe weights
probe.load_state_dict(checkpoint["probe_state_dict"])

# Get performance
print(f"Best AUROC: {checkpoint['val_auroc']:.4f}")
```

## Advanced Options

### Resume Training

```bash
python train_tuab_mne.py --resume output/tuab_*/checkpoint_*.pt
```

### Custom Configuration

```bash
python train_tuab_mne.py --config configs/tuab.yaml
```

### Multi-GPU Training

Not implemented - single GPU is sufficient for linear probes.

## Performance Tips

1. **Use memory-mapped arrays** for large datasets
2. **Keep EEGPT frozen** - only train probe
3. **Monitor GPU utilization** - should be >90%
4. **Use OneCycleLR** for faster convergence
5. **Validate every 2 epochs** to save time

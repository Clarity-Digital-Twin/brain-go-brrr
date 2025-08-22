# Training Guide

## Overview

This guide covers training linear probes on frozen EEGPT features for EEG classification tasks.

## Prerequisites

1. **EEGPT Checkpoint**: Download the pretrained model
   ```bash
   # Place at: data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
   ```

2. **Datasets**: Prepare TUAB/TUEV datasets
   ```bash
   # TUAB: data/datasets/tuab/
   # TUEV: data/datasets/tuev/
   ```

3. **Environment**: Python 3.11 with PyTorch
   ```bash
   uv sync  # Install dependencies
   ```

## TUAB Training (Abnormality Detection)

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
- **Features**: 32,768 dimensions (16 patches × 4 tokens × 512)

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
# experiments/eegpt_linear_probe/train_tuab.py

# Key components:
1. Memory-mapped dataset for efficiency
2. Frozen EEGPT backbone
3. OneCycleLR scheduler
4. Early stopping with patience
5. AUROC tracking
```

### Custom Dataset

```python
# experiments/eegpt_linear_probe/tuab_dataset.py

class TUABMemoryMappedDataset(Dataset):
    """Ultra-fast dataset using memory-mapped arrays.

    - No RAM usage - streams from disk
    - >1 GB/s read speed
    - Automatic windowing
    """
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
python train_tuab.py --resume output/tuab_*/last.pt
```

### Custom Configuration

```bash
python train_tuab.py --config configs/custom.yaml
```

### Multi-GPU Training

Not implemented - single GPU is sufficient for linear probes.

## Performance Tips

1. **Use memory-mapped arrays** for large datasets
2. **Keep EEGPT frozen** - only train probe
3. **Monitor GPU utilization** - should be >90%
4. **Use OneCycleLR** for faster convergence
5. **Validate every 2 epochs** to save time

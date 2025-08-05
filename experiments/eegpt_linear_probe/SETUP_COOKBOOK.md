# EEGPT Linear Probe Training Setup Cookbook

## 🎯 Overview

This cookbook documents all the issues we encountered while setting up EEGPT linear probe training and their solutions. Use this as a reference when creating new training scripts or debugging issues.

## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Common Issues & Solutions](#common-issues--solutions)
3. [Training Script Template](#training-script-template)
4. [Configuration Best Practices](#configuration-best-practices)
5. [Debugging Checklist](#debugging-checklist)

---

## 🛠️ Environment Setup

### Required Environment Variables

```bash
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
```

### Directory Structure Requirements

```
data/
├── models/
│   └── eegpt/
│       └── pretrained/
│           └── eegpt_mcae_58chs_4s_large4E.ckpt  # 1GB model file
├── datasets/
│   └── external/
│       └── tuh_eeg_abnormal/
│           └── v3.0.1/
│               └── edf/
│                   ├── train/
│                   └── eval/
└── cache/
    ├── tuab_index.json  # CRITICAL: Pre-built file index
    └── tuab_enhanced/   # Cached windows
```

---

## 🐛 Common Issues & Solutions

### 1. PyTorch Lightning Hanging Issue

**Problem**: Lightning 2.5.2 hangs indefinitely at "Loading `train_dataloader`" with large cached datasets.

**Solution**: Use pure PyTorch implementation instead.

```python
# ❌ DON'T USE
import pytorch_lightning as pl

# ✅ USE PURE PYTORCH
import torch
from torch.utils.data import DataLoader
```

### 2. Path Resolution Issues

**Problem**: `${BGB_DATA_ROOT}` not resolved in config files.

**Solution**: Manually resolve environment variables in the dataloader creation:

```python
def create_dataloaders(config):
    # Resolve environment variables in paths
    data_root = os.environ.get('BGB_DATA_ROOT', '/default/path')
    
    # Replace ${BGB_DATA_ROOT} in paths
    root_dir = config['data']['root_dir']
    if '${BGB_DATA_ROOT}' in root_dir:
        root_dir = root_dir.replace('${BGB_DATA_ROOT}', data_root)
    
    cache_dir = config['data']['cache_dir']
    if '${BGB_DATA_ROOT}' in cache_dir:
        cache_dir = cache_dir.replace('${BGB_DATA_ROOT}', data_root)
```

### 3. Dataset Interface Mismatch

**Problem**: Different datasets have different initialization parameters.

**Solution**: Use the correct dataset class with proper parameters:

```python
# For cached dataset with pre-built index
from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

train_dataset = TUABCachedDataset(
    root_dir=Path(root_dir),
    split='train',
    window_duration=8.0,
    window_stride=4.0,
    sampling_rate=256,
    preload=False,
    normalize=True,
    cache_dir=Path(cache_dir),
    cache_index_path=Path(data_root) / "cache" / "tuab_index.json"
)
```

### 4. Variable Channel Count Issue

**Problem**: Some EEG files have 19 channels, others have 20.

**Solution**: Use custom collate function to pad channels:

```python
from custom_collate_fixed import collate_eeg_batch_fixed

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    collate_fn=collate_eeg_batch_fixed  # Handles variable channels
)
```

### 5. Model Loading Issues

**Problem**: EEGPTWrapper vs EEGPTBackbone confusion.

**Solution**: Use EEGPTWrapper for inference:

```python
from brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

backbone = EEGPTWrapper(
    checkpoint_path=model_checkpoint
)
# Note: No freeze_backbone parameter in wrapper
backbone.eval()  # Manually set to eval mode
```

### 6. Config Key Errors

**Problem**: Nested config structure causing KeyError.

**Solution**: Pass the correct config section:

```python
# ❌ WRONG
probe = LinearProbe(config)

# ✅ CORRECT
probe = LinearProbe(config['model'])
```

### 7. OneCycleLR Type Error

**Problem**: max_lr read as string from YAML.

**Solution**: Cast to float:

```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=float(config['training']['scheduler']['max_lr']),
    # ... other params
)
```

### 8. EEGPT Output Dimension Mismatch

**Problem**: Expected 768-dim features but got 512-dim.

**Solution**: Check actual model output and adjust config:

```yaml
probe:
  input_dim: 512  # Not 768 for this specific model
```

---

## 📝 Training Script Template

```python
#!/usr/bin/env python
"""Template for EEGPT linear probe training scripts."""

import os
import sys
from pathlib import Path
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import custom modules
from src.brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
from src.brain_go_brrr.models.eegpt_wrapper import EEGPTWrapper

# Import custom collate function
sys.path.insert(0, str(Path(__file__).parent))
from custom_collate_fixed import collate_eeg_batch_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_env_vars(config):
    """Recursively resolve environment variables in config."""
    if isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        env_var = config[2:-1]
        return os.environ.get(env_var, config)
    elif isinstance(config, dict):
        return {k: resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    return config


def create_dataloaders(config):
    """Create dataloaders with proper path resolution."""
    # Resolve paths
    data_root = os.environ.get('BGB_DATA_ROOT', '/default/path')
    cache_index_path = Path(data_root) / "cache" / "tuab_index.json"
    
    # Resolve config paths
    root_dir = config['data']['root_dir']
    if '${BGB_DATA_ROOT}' in root_dir:
        root_dir = root_dir.replace('${BGB_DATA_ROOT}', data_root)
    
    cache_dir = config['data']['cache_dir']
    if '${BGB_DATA_ROOT}' in cache_dir:
        cache_dir = cache_dir.replace('${BGB_DATA_ROOT}', data_root)
    
    # Create datasets
    train_dataset = TUABCachedDataset(
        root_dir=Path(root_dir),
        split='train',
        window_duration=config['data']['window_duration'],
        window_stride=config['data']['window_stride'],
        sampling_rate=config['data']['sampling_rate'],
        preload=False,
        normalize=True,
        cache_dir=Path(cache_dir),
        cache_index_path=cache_index_path
    )
    
    val_dataset = TUABCachedDataset(
        root_dir=Path(root_dir),
        split='eval',
        window_duration=config['data']['window_duration'],
        window_stride=config['data']['window_duration'],  # No overlap for val
        sampling_rate=config['data']['sampling_rate'],
        preload=False,
        normalize=True,
        cache_dir=Path(cache_dir),
        cache_index_path=cache_index_path
    )
    
    # Create loaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers'],
        prefetch_factor=config['data']['prefetch_factor'],
        collate_fn=collate_eeg_batch_fixed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers'],
        prefetch_factor=config['data']['prefetch_factor'],
        collate_fn=collate_eeg_batch_fixed
    )
    
    return train_loader, val_loader


def main():
    # Load config
    config_path = "configs/your_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables
    config = resolve_env_vars(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model_checkpoint = config['model']['backbone']['checkpoint_path']
    if '${BGB_DATA_ROOT}' in model_checkpoint:
        data_root = os.environ.get('BGB_DATA_ROOT', '/default/path')
        model_checkpoint = model_checkpoint.replace('${BGB_DATA_ROOT}', data_root)
    
    backbone = EEGPTWrapper(checkpoint_path=model_checkpoint)
    backbone.to(device)
    backbone.eval()  # Freeze backbone
    
    # Create probe (your custom probe class)
    probe = YourProbeClass(config['model'])
    probe.to(device)
    
    # Training loop...
    

if __name__ == "__main__":
    main()
```

---

## ⚙️ Configuration Best Practices

### YAML Config Template

```yaml
experiment:
  name: "experiment_name"
  seed: 42

model:
  backbone:
    name: "eegpt"
    checkpoint_path: "${BGB_DATA_ROOT}/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    n_channels: 19
    
  probe:
    type: "linear"
    input_dim: 512  # Check actual model output!
    hidden_dim: 32
    n_classes: 2
    dropout: 0.5

data:
  dataset: "tuab"
  root_dir: "${BGB_DATA_ROOT}/datasets/external/tuh_eeg_abnormal/v3.0.1/edf"
  cache_dir: "${BGB_DATA_ROOT}/cache/tuab_enhanced"
  
  window_duration: 8.0
  window_stride: 4.0
  sampling_rate: 256
  
  batch_size: 256
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

training:
  max_epochs: 200
  
  optimizer:
    name: "AdamW"
    lr: 2.5e-4
    weight_decay: 0.01
    
  scheduler:
    name: "OneCycleLR"
    max_lr: 5e-4  # Will be cast to float
    epochs: 200
    pct_start: 0.3
    div_factor: 20
    final_div_factor: 1000
    
  gradient_clip_val: 5.0
  
  early_stopping:
    monitor: "val_auroc"
    patience: 20
    mode: "max"
```

---

## 🔍 Debugging Checklist

1. **Before Running**:
   - [ ] Set all environment variables
   - [ ] Check `tuab_index.json` exists
   - [ ] Verify model checkpoint exists
   - [ ] Confirm dataset path is correct

2. **Common Errors**:
   - [ ] FileNotFoundError → Check path resolution
   - [ ] KeyError → Check config structure
   - [ ] TypeError in scheduler → Cast numeric types
   - [ ] Shape mismatch → Check model output dimensions

3. **Performance Issues**:
   - [ ] Slow data loading → Check cache exists
   - [ ] High memory usage → Reduce batch size
   - [ ] Training hanging → Don't use PyTorch Lightning

4. **Monitoring**:
   ```bash
   # Launch in tmux
   tmux new-session -d -s training_session "python train_script.py"
   
   # Watch progress
   tmux attach -t training_session
   
   # Monitor logs
   tail -f output/*/training.log | grep -E "Epoch|AUROC|loss"
   ```

---

## 🚀 Quick Start Commands

```bash
# 1. Set environment
export BGB_DATA_ROOT=/path/to/data
export CUDA_VISIBLE_DEVICES=0

# 2. Run smoke test
python smoke_test_paper_aligned.py

# 3. Launch training
bash RUN_TRAINING_8S.sh

# 4. Monitor
tmux attach -t eegpt_training
```

---

## 📚 References

- EEGPT Paper: 4-second windows, 256Hz sampling
- PyTorch Lightning Bug: [Issue #2025](https://github.com/Lightning-AI/pytorch-lightning/issues/...)
- TUAB Dataset: Temple University Abnormal EEG Corpus v3.0.1
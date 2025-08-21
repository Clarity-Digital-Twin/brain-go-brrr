# EEGPT Linear Probe Training Guide

**Current Status**: 🟡 Training scripts ready, model training in progress

## Overview

This guide explains how to train the linear probe for EEGPT abnormality detection on the TUAB dataset.

## Prerequisites

1. **EEGPT Checkpoint**: Download to `data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
2. **TUAB Dataset**: Download and extract to `data/datasets/external/tuh_eeg_abnormal/`
3. **GPU**: Recommended for training (works on CPU but very slow)
4. **Memory**: 16GB+ recommended

## Training Script Location

**Main Script**: `experiments/eegpt_linear_probe/train_paper_aligned.py`

⚠️ **DO NOT USE PyTorch Lightning** - It has a critical bug with large cached datasets that causes training to hang indefinitely.

## Quick Start

### 1. Prepare Data Cache (Recommended)

First, build the cached dataset for faster training:

```bash
cd experiments/eegpt_linear_probe
python build_4s_cache_FINAL.py
```

This creates cached 4-second windows at 256Hz in `data/cache/tuab_4s_final/`.

### 2. Start Training

```bash
# Basic training
python train_paper_aligned.py --config configs/tuab_4s_paper_target.yaml

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train_paper_aligned.py --config configs/tuab_4s_paper_target.yaml --device cuda

# Monitor in tmux
tmux new -s training
python train_paper_aligned.py --config configs/tuab_4s_paper_target.yaml
# Detach with Ctrl+B, D
# Reattach with: tmux attach -t training
```

## Configuration

### Key Parameters (configs/tuab_4s_paper_target.yaml)

```yaml
data:
  window_size: 4.0  # CRITICAL: Must be 4 seconds for EEGPT
  sampling_rate: 256  # CRITICAL: Must be 256Hz
  batch_size: 256  # Adjust based on GPU memory
  n_channels: 20  # TUAB channel count

model:
  hidden_dim: 256
  dropout: 0.5
  learning_rate: 1e-3
  weight_decay: 1e-4

training:
  epochs: 100
  early_stopping_patience: 10
  val_check_interval: 0.25  # Validate 4x per epoch
```

### Channel Mapping

TUAB uses old channel naming. The dataset automatically converts:
- T3 → T7
- T4 → T8
- T5 → P7
- T6 → P8

## Monitoring Training

### Real-time Monitoring

```bash
# Watch training progress
tail -f experiments/eegpt_linear_probe/logs/training_*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### TensorBoard (if enabled)

```bash
tensorboard --logdir experiments/eegpt_linear_probe/lightning_logs
```

## Expected Performance

### Target Metrics (from EEGPT paper)
- **AUROC**: ≥0.869 on TUAB test set
- **Balanced Accuracy**: ~80%
- **Training Time**: ~2-4 hours on V100 GPU

### Current Status
- Training typically converges within 20-30 epochs
- Best models saved to `experiments/eegpt_linear_probe/checkpoints/`

## Common Issues

### 1. CUDA Out of Memory
**Solution**: Reduce batch_size in config (try 128 or 64)

### 2. Training Hangs at "Loading train_dataloader"
**Cause**: PyTorch Lightning bug with large datasets
**Solution**: Use the provided `train_paper_aligned.py` (pure PyTorch)

### 3. NaN Loss
**Causes**:
- Learning rate too high (try 1e-4)
- Bad data samples (enable data validation)
- Gradient explosion (enable gradient clipping)

### 4. Poor Performance
**Checks**:
- Ensure 4-second windows at 256Hz
- Verify channel mapping is correct
- Check class balance in dataset
- Try different dropout rates (0.3-0.7)

## Advanced Options

### Resume Training

```bash
python train_paper_aligned.py --resume checkpoint_epoch_20.pt
```

### Hyperparameter Search

```bash
# Grid search over learning rates
for lr in 1e-3 5e-4 1e-4; do
    python train_paper_aligned.py --learning_rate $lr --name lr_$lr
done
```

### Multi-GPU Training

```bash
# Not yet implemented - single GPU recommended
```

## Evaluation

After training, evaluate the model:

```bash
python evaluate_model.py --checkpoint best_model.pt --test_data data/cache/tuab_4s_final/test/
```

## Integration

To use the trained model in the main pipeline:

1. Copy best checkpoint to `data/models/linear_probes/`
2. Update config to point to trained model
3. Test with API endpoint

```python
# In application code
from brain_go_brrr.infra.ml_models import EEGPTWithProbe

model = EEGPTWithProbe(
    eegpt_checkpoint="data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt",
    probe_checkpoint="data/models/linear_probes/best_auroc_0.87.pt"
)
```

## References

- EEGPT Paper: [NeurIPS 2024](https://github.com/BINE022/EEGPT)
- TUAB Dataset: [Temple University Hospital EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/)
- Training Script: `experiments/eegpt_linear_probe/train_paper_aligned.py`

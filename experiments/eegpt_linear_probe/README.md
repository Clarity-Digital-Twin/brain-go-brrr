# EEGPT Linear Probe Experiments

This directory contains the implementation of EEGPT linear probing for EEG analysis tasks, following the paper "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals".

## Overview

Linear probing is a technique where we freeze a pretrained model (EEGPT) and only train a lightweight classification head on top. This approach:
- Requires minimal training time (minutes vs hours)
- Prevents overfitting on small datasets
- Validates the quality of pretrained representations

## Current Implementation

### 1. Abnormality Detection (TUAB Dataset)

- **Task**: Binary classification (normal vs abnormal EEG)
- **Target Performance**: AUROC ≥ 0.93 (from paper)
- **Dataset**: TUH Abnormal EEG Corpus v3.0.0
- **Channels**: 23 standard 10-20 channels
- **Window**: 30 seconds at 256 Hz

## Project Structure

```
eegpt_linear_probe/
├── configs/
│   └── tuab_config.yaml         # Training configuration
├── checkpoints/                 # Saved model checkpoints
├── logs/                        # TensorBoard logs
├── train_tuab_probe.py          # Main training script
├── inference_example.py         # Example inference script
└── README.md                    # This file
```

## Usage

### Prerequisites

1. Download EEGPT pretrained weights:
```bash
# Download from EEGPT repository
wget https://github.com/BINE022/EEGPT/releases/download/v1.0/eegpt_mcae_58chs_4s_large4E.ckpt
# Place in: data/models/eegpt/pretrained/
```

2. Prepare TUAB dataset:
```bash
# Dataset should be organized as:
data/datasets/external/tuh_eeg_abnormal/v3.0.0/
├── train/
│   ├── normal/*.edf
│   └── abnormal/*.edf
├── val/
│   ├── normal/*.edf
│   └── abnormal/*.edf
└── test/
    ├── normal/*.edf
    └── abnormal/*.edf
```

### Training

```bash
# From project root
cd experiments/eegpt_linear_probe

# Run training
python train_tuab_probe.py
```

The script will:
1. Load pretrained EEGPT and freeze its weights
2. Train only the channel adapter and classification head
3. Save checkpoints and logs
4. Report validation AUROC after each epoch

### Monitoring Training

```bash
# View training progress in TensorBoard
tensorboard --logdir logs/
```

### Inference

```bash
# Use trained probe for predictions
python inference_example.py
```

## Model Architecture

The linear probe consists of:

1. **Channel Adapter**: 1×1 convolution (23 → 58 channels)
2. **Frozen EEGPT**: Extracts 2048-dimensional features
3. **Classification Head**:
   - Linear(2048, 2048) + GELU + Dropout
   - Linear(2048, 2) for binary classification

Total trainable parameters: ~4.5M (vs 10M in full EEGPT)

## Expected Results

Based on the EEGPT paper:

| Metric | Target | Notes |
|--------|--------|-------|
| AUROC | ≥ 0.93 | Primary metric |
| Training Time | < 1 hour | On single GPU |
| Inference Speed | > 30× real-time | Frozen features |

## Customization

### Modify Training Configuration

Edit `configs/tuab_config.yaml`:

```yaml
training:
  batch_size: 64      # Adjust for GPU memory
  epochs: 10          # More epochs may help
  learning_rate: 5e-4 # Paper uses OneCycleLR
```

### Add New Tasks

1. Create task-specific probe class (e.g., `sleep_staging.py`)
2. Create corresponding config file
3. Modify training script or create task-specific version

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Enable gradient checkpointing
- Use mixed precision (`precision: 16`)

### Poor Performance
- Ensure data is properly preprocessed (0.5-50 Hz bandpass)
- Check channel ordering matches TUAB standard
- Verify EEGPT checkpoint loaded correctly
- Try longer training (more epochs)

### Missing Channels
- The dataset loader pads with zeros for missing channels
- Performance may degrade with many missing channels

## Next Steps

1. **Sleep Staging**: Implement 5-class sleep stage classification
2. **Multi-GPU Training**: Add DistributedDataParallel support
3. **Hyperparameter Search**: Optimize learning rate, weight decay
4. **Cross-Dataset Evaluation**: Test on CHB-MIT, Sleep-EDF

## References

- [EEGPT Paper](https://arxiv.org/abs/2410.20150)
- [EEGPT Code](https://github.com/BINE022/EEGPT)
- [TUH EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/)

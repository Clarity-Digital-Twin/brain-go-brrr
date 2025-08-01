# EEGPT Linear Probe Training - PRODUCTION SETUP

## 🚀 Quick Start

```bash
# 1. Run smoke test FIRST
uv run python experiments/eegpt_linear_probe/RUN_THIS_FIRST_smoke_test.py

# 2. Launch training
bash launch_bulletproof_training.sh

# 3. Monitor
tmux attach -t eegpt_bulletproof_[timestamp]
```

## 📁 Directory Structure

```
experiments/eegpt_linear_probe/
├── train_paper_aligned.py      # MAIN TRAINING SCRIPT (use this!)
├── train_tuab_probe.py         # Base trainer class
├── launch_bulletproof_training.sh  # Launch script
├── RUN_THIS_FIRST_smoke_test.py   # Pre-flight checks
├── configs/
│   └── tuab_config.yaml       # Training configuration
├── logs/                       # Training logs
├── checkpoints/                # Model checkpoints
└── archive/                    # Old scripts (ignore)
```

## ⚠️ CRITICAL PATHS - GET THESE RIGHT!

```python
# CORRECT dataset path (includes /edf/)
root_dir = data_root / "datasets/external/tuh_eeg_abnormal/v3.0.1/edf"

# CORRECT checkpoint path
checkpoint = data_root / "models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
```

## 📊 Dataset Info

- **Total Windows**: 512,137 (465,934 train + 46,203 val)
- **Train**: 2,717 EDF files (1,371 normal, 1,346 abnormal)
- **Eval**: 276 EDF files (150 normal, 126 abnormal)
- **Window**: 8 seconds @ 256Hz = 2,048 samples
- **Channels**: 20 (modern 10-20 naming)

## 🔧 Configuration

Edit `configs/tuab_config.yaml`:
```yaml
training:
  batch_size: 100      # Paper value for TUAB
  learning_rate: 5e-4  # Paper value
  epochs: 30           # Paper value
  precision: 32        # FP32 for stability
```

## 🐛 Common Issues

1. **Missing channels warnings**: EXPECTED - dataset uses old naming (T3→T7)
2. **NaN in predictions**: Fixed with guards in `train_paper_aligned.py`
3. **Wrong path**: Must use `/v3.0.1/edf/` not just `/v3.0.1/`

## 📈 Expected Performance

- **Target AUROC**: ≥0.93 (from paper)
- **Training time**: 4-6 hours on RTX 4090
- **GPU memory**: ~4-6GB
- **Validation**: Every 0.5 epochs

## 🔍 Monitoring

```bash
# Watch training progress
tail -f logs/bulletproof_*.log

# GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir logs/
```

## ✅ Success Criteria

1. Training completes 30 epochs
2. AUROC > 0.90 on validation
3. No NaN/inf in losses
4. Checkpoints saved every epoch

## 🚨 DO NOT

- Change dataset paths
- Use old scripts in archive/
- Skip the smoke test
- Use mixed precision (causes NaN)
- Modify channel mappings
# EEGPT Linear Probe Training

## 🎯 Mission: Achieve Paper-Level Performance

Train a linear probe on frozen EEGPT features for EEG abnormality detection using the TUAB dataset.

**Target**: AUROC ≥ 0.869 (paper performance with 4-second windows)

## 🟢 Current Status

**TRAINING ACTIVE**: 4-second window configuration running
- Session: `tmux attach -t eegpt_4s_final`
- Expected completion: ~3-4 hours
- Monitor: `tail -f output/tuab_4s_paper_aligned_20250805_181351/training.log`

## ⚡ Quick Start

```bash
# Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data

# Run smoke test to verify setup
python smoke_test_paper_aligned.py

# Launch training (4-second windows - CORRECT)
bash launch_paper_aligned_training.sh

# Monitor progress
tmux attach -t eegpt_4s_final
```

## 📁 Clean Directory Structure

```
experiments/eegpt_linear_probe/
├── configs/                      # Training configurations
│   ├── tuab_4s_paper_aligned.yaml  # ✅ ACTIVE - Paper-aligned 4s config
│   ├── tuab_8s_temp.yaml           # 8s config (suboptimal)
│   └── archive/                    # Old configs
├── output/                       # Training outputs
│   └── tuab_4s_paper_aligned_*/    # Current training
├── archive/                      # Obsolete/failed attempts
│   └── old_scripts/              # Deprecated scripts
├── train_paper_aligned.py       # ✅ MAIN training script
├── smoke_test_paper_aligned.py  # Pre-flight checks
├── custom_collate_fixed.py      # Handles variable channels
├── launch_paper_aligned_training.sh  # Launch script
└── *.md                          # Documentation
```

## 🔑 Critical Insights

### Why 4-Second Windows Are Essential

| Window Size | AUROC | Status | Notes |
|------------|-------|--------|-------|
| **4 seconds** | **0.869** | **✅ Paper** | EEGPT pretrained on 4s |
| 8 seconds | ~0.81 | ❌ Too low | Mismatched with pretraining |

**The pretrained EEGPT model expects 4-second windows!**

### Architecture

```
Input (4s @ 256Hz) → EEGPT (frozen) → Linear Probe → Binary Classification
     1024 samples       512-dim            2 classes
                       features         (normal/abnormal)
```

## ⚠️ Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| PyTorch Lightning hangs | Use pure PyTorch (`train_paper_aligned.py`) |
| Missing cache index | Copy from 8s cache or build new |
| Channel count mismatch | Use `custom_collate_fixed.py` |
| Wrong window size | MUST use 4 seconds |
| Old channel names | TUAB uses T3/T4/T5/T6 (handled automatically) |

## 📊 Performance Benchmarks

| Metric | Current | Target | Paper |
|--------|---------|--------|-------|
| AUROC | TBD (training) | ≥0.85 | 0.869 |
| Balanced Acc | TBD | >80% | 85.4% |
| Window Size | 4s | 4s | 4s |
| Epochs | 0/200 | - | 200 |

## 🛠️ Key Configuration

```yaml
# configs/tuab_4s_paper_aligned.yaml
data:
  window_duration: 4.0  # CRITICAL: Must be 4 seconds
  window_stride: 2.0    # 50% overlap for training
  sampling_rate: 256
  
model:
  backbone:
    checkpoint_path: eegpt_mcae_58chs_4s_large4E.ckpt
  probe:
    input_dim: 512  # EEGPT embedding dimension
```

## 📋 Monitoring Commands

```bash
# Live training view
tmux attach -t eegpt_4s_final

# Check process
ps aux | grep train_paper_aligned

# Watch logs
tail -f output/tuab_4s_paper_aligned_*/training.log | grep -E "Epoch|AUROC"

# GPU usage
watch -n 1 nvidia-smi
```

## 📚 Documentation

- `TRAINING_STATUS.md` - Live training updates
- `ISSUES_AND_FIXES.md` - Problems encountered & solutions
- `SETUP_COOKBOOK.md` - Detailed setup guide

## 🎯 Success Criteria

- [ ] AUROC ≥ 0.869 on validation set
- [ ] Stable training (no NaN/divergence)
- [ ] Reproducible results (seed=42)
- [ ] Saved best checkpoint

## 🚀 Next Steps

1. **Let current training complete** (3-4 hours)
2. **Evaluate on test set** once training finishes
3. **Save best model** for production inference
4. **Document final results** in TRAINING_STATUS.md

---

**Remember**: The key to success is using 4-second windows to match the EEGPT pretraining!
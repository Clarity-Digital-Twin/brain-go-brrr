# EEGPT Linear Probe Training Status

## 🔴 ACTIVE TRAINING - TUAB

**Session**: `tmux attach -t tuab_training`
**Started**: Aug 22, 2025 23:55
**Progress**: ~3.5% (Batch 1000+ of 29143 per epoch)
**Status**: Running smoothly @ 1.3-1.5 it/s

### Current Configuration
- **Dataset**: TUAB (abnormality detection)
- **Batch Size**: 64 (reduced from 256 for stability)
- **Epochs**: 10
- **Target AUROC**: 0.869 (paper performance)
- **Checkpointing**: Every 500 batches
- **Memory cleanup**: Every 100 batches

### Files
- **Training Script**: `train_tuab.py` (fixed with error handling)
- **Launch Script**: `launch_tuab.sh` -> `scripts/launch_tuab_fixed.sh`
- **Monitor**: `./monitor_training.sh`
- **Config**: `configs/tuab.yaml`
- **Log**: `logs/tuab_training_20250822_235519.log`
- **Output**: `output/tuab_20250822_235519/`

### Improvements Applied
1. ✅ Added exception handling in training loop
2. ✅ Periodic memory cleanup (every 100 batches)
3. ✅ Checkpoint saving (every 500 batches)
4. ✅ Reduced batch size (64 from 256)
5. ✅ Better logging and monitoring

## 🔵 PENDING - TUEV

**Status**: Not started
**Files Ready**:
- `train_tuev.py` - Training script
- `scripts/launch_tuev.sh` - Launch script
- `tuev_dataset.py` - Dataset loader
- `tuev_dataset_cached.py` - Cached version

## 📁 Folder Structure (Cleaned)

```
eegpt_linear_probe/
├── configs/           # Training configs
├── scripts/           # Launch scripts
│   ├── launch_tuab_fixed.sh  # Current TUAB launcher
│   ├── launch_tuev.sh         # TUEV launcher
│   └── build_tuev_cache*.py   # Cache builders
├── logs/              # Active training logs (old archived)
├── output/            # Current training outputs
├── archive/           # Old scripts/logs/outputs
├── utils/             # Helper utilities
│
├── train_tuab.py      # TUAB training (ACTIVE)
├── train_tuev.py      # TUEV training (pending)
├── tuab_dataset.py    # TUAB data loader
├── tuev_dataset*.py   # TUEV data loaders
├── monitor_training.sh # Training monitor
└── launch_tuab.sh     # Symlink to scripts/launch_tuab_fixed.sh
```

## 🎯 Next Steps

1. **Monitor current TUAB training** until completion (~6-7 hours per epoch)
2. **Check for checkpoint saves** at batch 500, 1000, 1500...
3. **After TUAB completes**, start TUEV training if needed

## 📊 Expected Timeline

- **Epoch 1**: ~6-7 hours
- **Full training (10 epochs)**: ~60-70 hours
- **Early stopping**: May trigger after 3 epochs if no improvement

## 🚀 Quick Commands

```bash
# Monitor live training
tmux attach -t tuab_training

# Check status
./monitor_training.sh

# View log
tail -f logs/tuab_training_20250822_235519.log

# If crashed, restart
./launch_tuab.sh
```

# TUEV Paper Parity Workspace Status
**Last Updated**: September 9, 2025  
**Status**: ✅ CLEAN AND READY FOR PAPER PARITY TRAINING

## 🎯 Current Mission
Achieve EEGPT paper's 62.32% BAC on TUEV using exact paper approach:
- 23 input channels with learnable Conv2d(23→20) mapper
- Exact hyperparameters: lr=5e-4, wd=0.05, smoothing=0.1
- No class weights, no preprocessing tricks

## 📁 Clean Directory Structure

```
experiments/eegpt_linear_probe/
├── configs/
│   ├── tuev_paper_parity.yaml  # ✅ ACTIVE - Paper parity config
│   ├── tuab.yaml               # Keep for TUAB experiments
│   ├── tuab_smoke_test.yaml    # Keep for TUAB testing
│   └── archive/                # Old TUEV configs (wrong approach)
├── scripts/
│   ├── launch_tuev_paper_parity.sh  # ✅ ACTIVE - Main training script
│   ├── build_tuev_23ch_cache.sh     # ✅ ACTIVE - Cache builder
│   ├── launch_tuab_mne.sh           # Keep for TUAB
│   ├── monitor_mne_training.sh      # Utility script
│   ├── run_smoke_test.sh            # General testing
│   └── validate_cache.py            # Cache validation
├── output/                          # 🧹 EMPTY - Ready for new runs
├── logs/                            # 🧹 EMPTY - Ready for new logs
├── archive/
│   └── 2025-09-09_pre_paper_parity/ # All old experiments archived
└── train_tuev_mne.py               # ✅ UPDATED - Has mapper integration
```

## ✅ What's Ready

1. **Code**: All components implemented and tested
   - `TUEVChannelMapper` in `infra/ml_models/channel_mapper.py`
   - `Conv2dWithConstraint` in `domain/constraints.py`
   - Dataset supports `use_paper_parity=True`
   - Training script integrated with mapper

2. **Config**: `tuev_paper_parity.yaml` with exact paper settings

3. **Scripts**: Clean, focused scripts for paper parity
   - `launch_tuev_paper_parity.sh` - Main launcher
   - `build_tuev_23ch_cache.sh` - Cache builder

4. **Workspace**: Clean directories, all old experiments archived

## 🚀 Next Steps

### 1. Build 23-Channel Cache (4-6 hours)
```bash
tmux new -s tuev_23ch_cache
cd experiments/eegpt_linear_probe
./scripts/build_tuev_23ch_cache.sh
# Detach: Ctrl+B, D
```

### 2. Launch Training (after cache)
```bash
tmux new -s tuev_parity_training
cd experiments/eegpt_linear_probe
./scripts/launch_tuev_paper_parity.sh
# Detach: Ctrl+B, D
```

### 3. Monitor Progress
```bash
# Check cache build
tmux attach -t tuev_23ch_cache

# Check training  
tmux attach -t tuev_parity_training

# Watch metrics
tail -f logs/tuev_paper_parity_*.log | grep -E "BAC|balanced"
```

## 📊 Expected Timeline

- **Cache Build**: 4-6 hours
- **Training**: 8-12 hours (100 epochs)
- **Target Metrics**:
  - Balanced Accuracy: 62.32% ± 1.14%
  - Weighted F1: 81.87% ± 0.63%
  - Cohen's Kappa: 0.635 ± 0.013

## 🗑️ What Was Archived

Moved to `archive/2025-09-09_pre_paper_parity/`:
- 25 old training output directories
- 15+ log files from failed approaches
- Outdated scripts and configs
- Old documentation

All used wrong 20-channel preprocessing approach (22% BAC).

## 📝 Notes

- Output and logs directories are empty and ready
- No background processes running
- All paths use environment variables ($BGB_DATA_ROOT)
- Ready to start fresh with paper parity approach
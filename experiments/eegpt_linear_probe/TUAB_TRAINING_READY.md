# TUAB Training - READY TO LAUNCH! 🚀

## Status: ✅ 100% Ready

All critical bugs have been fixed and verified!

## What Was Fixed
1. ✅ **Feature extraction**: Now using ALL temporal patches (16 × 4 × 512 = 32,768 features)
2. ✅ **Dynamic dimensions**: LazyLinear automatically infers input size
3. ✅ **Batch size**: Aligned with paper (100 samples)
4. ✅ **Cache**: Pre-built and ready (2998 files)
5. ✅ **Model**: EEGPT checkpoint verified and loaded

## Expected Performance
- **Target AUROC**: 0.87 (paper's reported performance)
- **Previous (broken)**: 0.79 (only used 0.8% of features)
- **Expected improvement**: ~10% AUROC increase

## How to Train

### Quick Launch
```bash
cd experiments/eegpt_linear_probe
./LAUNCH_TUAB_TRAINING.sh
```

### Manual Launch
```bash
cd experiments/eegpt_linear_probe
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
python train_tuab.py --config configs/tuab.yaml
```

### Monitor Training
```bash
# Attach to tmux session
tmux attach -t tuab_training

# Or watch logs
tail -f logs/tuab_training_*.log
```

## Training Configuration
- **Dataset**: TUAB (Temple University Abnormal EEG)
- **Task**: Binary classification (normal vs abnormal)
- **Windows**: 4 seconds @ 256Hz
- **Overlap**: 50% stride for training
- **Backbone**: EEGPT (frozen)
- **Probe**: 2-layer MLP with ReLU and dropout
- **Optimizer**: AdamW with OneCycleLR
- **Early stopping**: Patience=10 on validation AUROC

## File Structure
```
experiments/eegpt_linear_probe/
├── train_tuab.py              # Main training script ✅
├── tuab_dataset.py            # Dataset loader ✅
├── configs/tuab.yaml          # Configuration ✅
├── LAUNCH_TUAB_TRAINING.sh    # Launch script ✅
├── VERIFY_100_PERCENT_GUCCI.py # Verification ✅
└── archive/                   # Old attempts (ignore)
```

## After Training Completes
The best model will be saved to:
- `output/tuab_YYYYMMDD_HHMMSS/best_model.pt`
- Check final AUROC in logs - should be ~0.87

## Next Steps
Once TUAB reaches target performance (0.87 AUROC):
1. Move to TUEV event detection training
2. Use `train_tuev.py` with similar approach
3. Target: 0.62 balanced accuracy

---

*Ready to achieve paper-level performance! 🎯*
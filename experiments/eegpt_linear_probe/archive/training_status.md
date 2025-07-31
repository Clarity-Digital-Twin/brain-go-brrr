# EEGPT Linear Probe Training Status

## Current Status: 🟢 TRAINING IN PROGRESS

### Configuration
- **Model**: EEGPT Linear Probe (4.2M trainable params)
- **Dataset**: TUAB v3.0.1 (465k train, 46k eval windows)
- **Window Size**: 8s (2048 samples @ 256Hz)
- **Batch Size**: 32
- **Learning Rate**: 5e-4 with OneCycleLR
- **Hardware**: Apple M-series (MPS backend)

### Optimizations Applied
1. ✅ **Channel Mapping**: Fixed TUAB channel naming (EEG FP1-REF → FP1)
2. ✅ **Window Size**: Adjusted to 2048 (divisible by patch_size 64)
3. ✅ **Channel Adapter**: 23 → 20 channels per paper
4. ✅ **MNE Logging**: Reduced to ERROR level
5. ✅ **Data Caching**: Enabled for preprocessed windows
6. ✅ **Quick Test Mode**: 2% train / 10% val for rapid iteration

### Known Issues
- ⚠️ Missing channels (FPZ, OZ) in most files - handled gracefully
- ⚠️ Slow first epoch on M-series due to I/O bottleneck
- ⚠️ MNE legacy warnings - harmless, will be fixed in MNE 2.0

### Performance Expectations
- **First Batch**: 5-10 minutes (I/O bound)
- **Subsequent Batches**: ~1-2 sec/batch with caching
- **Target AUROC**: ≥ 0.93 (paper baseline: 0.87)

### Next Steps
1. Wait for first epoch completion (~20-30 min)
2. Check AUROC in logs/TensorBoard
3. If AUROC > 0.85, proceed with full training
4. If AUROC < 0.70, debug channel mapping

### Monitoring Commands
```bash
# Check process
ps aux | grep train_tuab_probe

# Watch logs
tail -f training_quick_*.log | grep -E "val_auroc|Epoch"

# TensorBoard
tensorboard --logdir lightning_logs

# Interactive monitor
python monitor_training.py
```

---
Last Updated: 2024-07-30 19:21 PST

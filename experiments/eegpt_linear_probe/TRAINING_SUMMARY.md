# EEGPT Linear Probe Training Summary

## Channel Mapping Issue - RESOLVED ✅

### Problem
- TUAB dataset uses old channel naming: T3, T4, T5, T6
- EEGPT expects modern naming: T7, T8, P7, P8
- Model was receiving 23 channels instead of expected 20

### Solution
1. Updated `src/brain_go_brrr/data/tuab_dataset.py` to use modern naming
2. Fixed `src/brain_go_brrr/tasks/abnormality_detection.py` to expect 20 channels
3. Cleared Python cache to ensure fresh imports

### Final Configuration
- **Channels**: 20 (standard 10-20 system with modern naming)
- **Window size**: 8 seconds (2048 samples at 256Hz)
- **Batch size**: 64 (from EEGPT paper)
- **Learning rate**: 5e-4 (from EEGPT paper)
- **Epochs**: 10

## Training Status

### Current Run
- **Script**: `train_fresh.py`
- **Log**: `logs/training_20250730_214545.log`
- **Status**: RUNNING ✅
- **Notes**: Successfully processing windows with correct 20-channel configuration

### Key Files
- `train_fresh.py` - Main training script with no Python caching
- `monitor_progress.py` - Monitor training progress
- `configs/tuab_config.yaml` - Configuration file
- `CHANNEL_MAPPING_EXPLAINED.md` - Detailed explanation of channel mapping

## Next Steps
1. Monitor training until completion
2. Test inference with trained probe
3. Evaluate performance against paper benchmarks (AUROC ≥ 0.93)

## Organization Changes
- Archived old training scripts to `archive/scripts/`
- Consolidated logs in `logs/` directory
- Removed duplicate `experiments/` folder
- Created clear documentation for channel mapping

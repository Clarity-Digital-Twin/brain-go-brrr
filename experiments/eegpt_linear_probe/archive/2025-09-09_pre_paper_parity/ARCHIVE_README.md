# Archive: Pre-Paper Parity Experiments
**Archived Date**: September 9, 2025  
**Reason**: Cleanup before TUEV paper parity implementation (62% BAC target)

## What's Archived Here

### /output/
All training outputs before paper parity implementation:
- **smoke_test runs**: Initial testing runs
- **tuab_mne runs**: TUAB experiments (achieved 82.7% AUROC)
- **tuev_mne runs**: Old 20-channel approach (achieved only 22% BAC)
- **tuev_fixed/v4 runs**: Failed attempts at fixing TUEV

### /logs/
Training logs from all archived runs

### /scripts/
Outdated scripts replaced by paper parity versions:
- `launch_tuev_cache.sh` → replaced by `build_tuev_23ch_cache.sh`
- `launch_tuev_mne.sh` → replaced by `launch_tuev_paper_parity.sh`
- `build_mne_cache.sh` → TUAB-specific, not needed for TUEV
- `clean_tuev_restart.sh` → one-time cleanup script

### /configs/
Old configs using wrong approach:
- `tuev.yaml` → 20-channel preprocessing (wrong)
- `tuev_smoke_test.yaml` → based on wrong approach

### /docs/
- `TRAINING_AUDIT.md` → outdated audit from Sept 4
- `MAPPER_INTEGRATION.md` → integration notes (now complete)
- `test_channel_enforcement.py` → old test file

## Why Archived

These experiments used the **wrong approach** for TUEV:
- Preprocessed to 20 channels (dropped A1/A2/T1/T2)
- Synthesized Fpz channel
- No learnable channel mapper
- Result: 22% BAC (far below 62% target)

## Active Approach

Use paper parity implementation:
- Keep all 23 channels
- Learnable Conv2d(23→20) mapper
- Exact EEGPT hyperparameters
- Target: 62.32% BAC
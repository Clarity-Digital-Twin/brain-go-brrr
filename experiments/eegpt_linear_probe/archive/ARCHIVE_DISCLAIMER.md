# ⚠️ ARCHIVE DISCLAIMER - OUTDATED TRAINING DOCUMENTATION

**CRITICAL**: These archived files contain OUTDATED and INCORRECT information about EEGPT training.

## DO NOT USE THESE FILES

All files in this archive contain deprecated approaches, including:
- Wrong window sizes (8 seconds instead of correct 4 seconds)
- References to non-existent scripts (`train_paper_aligned.py`)
- PyTorch Lightning implementations that crash with large datasets
- Outdated configurations and hyperparameters

## Current Training Approach

**Use ONLY**: `experiments/eegpt_linear_probe/train_tuab.py`
- Pure PyTorch implementation (no Lightning)
- 4-second windows (1024 samples at 256Hz)
- Correct channel mappings (T3→T7, T4→T8, T5→P7, T6→P8)

## Source of Truth

- Configuration: `/experiments/eegpt_linear_probe/configs/tuab.yaml`
- Documentation: `/CLAUDE.md` and `/AGENTS.md`
- Training guide: `/docs/TRAINING.md`

---
*Disclaimer added: 2025-08-22*

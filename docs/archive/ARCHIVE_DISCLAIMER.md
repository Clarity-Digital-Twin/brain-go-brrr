# ⚠️ ARCHIVE DISCLAIMER

**IMPORTANT**: Files in this archive directory contain OUTDATED information and should NOT be used as reference.

## Known Outdated Information

### Window Sizes
- **INCORRECT**: References to 8-second windows for TUAB training
- **CORRECT**: TUAB uses 4-second windows (1024 samples at 256Hz)
- **Source of Truth**: `/experiments/eegpt_linear_probe/configs/tuab.yaml`

### Training Scripts
- **INCORRECT**: `train_paper_aligned.py` or `train_pytorch_stable.py`
- **CORRECT**: `experiments/eegpt_linear_probe/train_tuab.py`
- **DO NOT USE**: PyTorch Lightning modules (critical bug causes crashes)

### YASA Channel Requirements
- **INCORRECT**: YASA requires 2 channels or is limited to Sleep-EDF
- **CORRECT**: YASA works with ANY channel count (1-100+), achieves 85%+ accuracy with 1 channel

### File Paths
- **INCORRECT**: `/src/brain_go_brrr/services/yasa_adapter.py`
- **CORRECT**: `/src/brain_go_brrr/infra/external/yasa_adapter.py`

## For Current Information

Please refer to:
- `/CLAUDE.md` - AI assistant context (current)
- `/AGENTS.md` - Project status and guidelines (current)
- `/docs/ARCHITECTURE.md` - System design (current)
- `/docs/TRAINING.md` - Training guide (current)

---
*This disclaimer added: 2025-08-22*

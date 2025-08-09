# EEGPT Training Resume Guide

## Current Status (Aug 8, 2025)

**Training crashed after 2 days at epoch 15 with AUROC 0.7916** (91% of target 0.869)

## Files That Matter

### Core Training Scripts
- `train_paper_aligned_resume.py` - Resume training from checkpoint (FIXED)
- `train_paper_aligned.py` - Original training script
- `tuab_mmap_dataset_safe.py` - WSL-safe dataset loader for 143GB memory-mapped arrays

### Critical Checkpoint
- `output/tuab_4s_paper_target_20250806_132743/best_model.pt` - **DO NOT DELETE!**
  - Contains epoch 15 checkpoint
  - AUROC: 0.7916
  - Has probe weights, optimizer state, scheduler state

## How to Resume Training

```bash
# 1. Set environment
export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 2. Launch in tmux (so it survives SSH disconnects)
tmux new-session -d -s eegpt_resume \
    "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/.venv/bin/python \
     train_paper_aligned_resume.py \
     --config configs/tuab_4s_wsl_safe.yaml \
     --resume output/tuab_4s_paper_target_20250806_132743/best_model.pt \
     2>&1 | tee logs/resume_$(date +%Y%m%d_%H%M%S).log"

# 3. Monitor progress
tmux attach -t eegpt_resume
```

## Critical Settings for WSL

**NEVER change these or training will crash!**

```yaml
# configs/tuab_4s_wsl_safe.yaml
data:
  num_workers: 0      # MUST BE 0 for WSL with large mmap files!
  pin_memory: false   # WSL stability
  batch_size: 32      # Don't go higher, memory issues
```

## Why Training Crashed

1. **DataLoader workers + 143GB memory-mapped file on WSL = crash**
   - Solution: `num_workers=0` (single process loading)

2. **Tensor device mismatch in extract_features**
   - Fixed: Now handles both numpy arrays and CUDA tensors

3. **Probe structure mismatch**
   - Fixed: Using nn.Sequential structure matching saved model

## Target Metrics

- **Current**: AUROC 0.7916 (epoch 15)
- **Target**: AUROC 0.869 (paper performance)
- **Needed**: +0.077 improvement (~10% more)

## Monitoring

Check training progress:
```bash
# View live training
tmux attach -t eegpt_resume

# Check latest metrics
tail -f logs/resume_*.log | grep AUROC

# GPU usage
watch -n 1 nvidia-smi
```

## If Training Crashes Again

1. **Check the error**: `tail -100 logs/resume_*.log`
2. **Checkpoint is automatically saved** every epoch if validation improves
3. **Resume from latest**: Update `--resume` path to latest checkpoint
4. **Do NOT delete** `output/*/best_model.pt` files!

## Dataset Info

- **Training**: 1,865,106 windows (4-second @ 256Hz)
- **Validation**: 184,938 windows
- **Memory-mapped**: 143GB arrays in `/data/cache/tuab_4s_final/`
- **WSL Warning**: MUST use `num_workers=0` or will crash!

## Expected Training Time

- ~2-3 hours per epoch on single GPU
- Need ~5-10 more epochs to reach target
- Total: 10-30 hours remaining

## Success Criteria

Training complete when:
1. AUROC >= 0.869 (target from paper)
2. Or early stopping triggered (10 epochs without improvement)
3. Or max 50 epochs reached

## Archive Structure

```
archive/
├── old_attempts/     # Failed training scripts
├── build_scripts/    # Cache building scripts
└── debug/           # Debugging utilities
```
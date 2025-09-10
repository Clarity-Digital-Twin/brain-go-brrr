# TUEV Training Troubleshooting Guide

## For OSS Contributors and Future Developers

This guide documents critical issues and solutions discovered during TUEV implementation (Sep 10, 2025).

## 🚨 Critical Issue #1: Channel Mismatch Crash

### Symptom
```
RuntimeError: The size of tensor a (16) must match the size of tensor b (32) at non-singleton dimension 2
```

### Root Cause
EEGPT model initialized with wrong channel count (19 or 58 instead of 20).

### Solution
Pass exactly 20 channel names to EEGPT:
```python
use_channels_names = ['FP1','FPZ','FP2','F7','F3','FZ','F4','F8',
                      'T7','C3','CZ','C4','T8','P7','P3','PZ','P4','P8','O1','O2']
model_kwargs = {"n_channels": use_channels_names}
```

## 🚨 Critical Issue #2: Training Hangs/Crashes in WSL2

### Symptoms
- tmux sessions die without error
- Training hangs after "Loading datasets..."
- GPU memory allocated but no processes shown

### Root Causes
1. **DataLoader workers deadlock** in WSL2 with `pin_memory=True`
2. **GPU memory fragmentation** from killed processes
3. **tmux instability** under memory pressure

### Solutions

#### Immediate Fix - Use Safe Launch Command:
```bash
tmux new -d -s tuev_parity "CUDA_LAUNCH_BLOCKING=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  uv run python experiments/eegpt_linear_probe/train_tuev_events.py \
  --data_dir data/datasets/tuev \
  --eegpt_checkpoint data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt \
  --use_parity \
  --epochs 30 \
  --batch_size 32 \
  --num_workers 0 \
  --save_dir experiments/eegpt_linear_probe/output/tuev_parity_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee experiments/eegpt_linear_probe/logs/tuev_parity_$(date +%Y%m%d_%H%M%S).log"
```

#### Long-term Code Fix:
Add these parameters to DataLoader:
```python
DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,  # Changed from 4
    pin_memory=False,  # Changed from True
    persistent_workers=False  # Explicit
)
```

## 🚨 Critical Issue #3: Cache Building Confusion

### Symptom
Unsure if cache needs rebuilding after code changes.

### Understanding
- Cache contains: 5s @ 200Hz segments (23 channels, 1000 samples)
- Train: 2695 segments, Eval: 1048 segments
- Location: `data/datasets/tuev/cache/tuev_event_segments/`

### When to Rebuild Cache
- ❌ After fixing training script bugs
- ❌ After changing model architecture
- ✅ After changing preprocessing (filters, resampling)
- ✅ After changing segment extraction logic

### Rebuild Command
```bash
rm -rf data/datasets/tuev/cache/
uv run python scripts/build_tuev_cache.py
```

## 📁 Directory Structure Clarification

### `/scripts/` - Generic utilities and tools
```
scripts/
├── data/              # Dataset download/verification
├── testing/           # Testing utilities
├── tools/             # Development tools
└── *.py              # One-off scripts (can be deleted)
```

### `/experiments/` - Training scripts ONLY
```
experiments/
└── eegpt_linear_probe/
    ├── train_*.py     # Training scripts (thin, import from src/)
    ├── scripts/       # Experiment-specific launchers
    ├── output/        # Model checkpoints
    └── logs/          # Training logs
```

### Rule: Everything reusable goes in `/src/`

## 🔍 Debugging Commands

### Check Training Progress
```bash
tmux attach -t tuev_parity
# or
tmux capture-pane -t tuev_parity -p | grep BAC
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

### Check for Hung Processes
```bash
ps aux | grep train_tuev
nvidia-smi | grep python
```

### Kill Everything and Start Fresh
```bash
pkill -9 -f train_tuev_events
pkill -9 -f python
tmux kill-server
```

## 📊 Expected Training Trajectory

Healthy TUEV training should show:
- Epoch 1: BAC ~16.7% (random for 6 classes)
- Epoch 2: BAC >18%
- Epoch 5: BAC >25%
- Epoch 10: BAC >40%
- Epoch 30: BAC ~62% (target)

If BAC stalls below these milestones, check:
1. Learning rate (should be 5e-4)
2. Label smoothing (should be 0.1)
3. Warmup schedule (5 epochs)
4. Data normalization (50μV std)

## 🛡️ Preventing Future Issues

### Before Pushing
```bash
make check-all  # Runs all CI checks
```

### Before Training
1. Check GPU memory: `nvidia-smi`
2. Kill old processes: `pkill -9 -f train_tuev`
3. Use parity mode for memory efficiency
4. Always log to file for crash recovery

### For OSS Contributors
1. Never create parallel implementations in experiments/
2. Always check src/ for existing components
3. Use the safe launch command above
4. Document any new issues in this file

## 📝 Configuration Checklist

- [ ] 23→20 channel mapping via Conv2d
- [ ] EEGPT configured with 20 channel names
- [ ] Parity mode: 1000 samples, patch_stride=64
- [ ] num_workers=0 for WSL2
- [ ] Logging to file
- [ ] Checkpoint saving enabled

## 🆘 Getting Help

If training crashes with new errors:
1. Check this guide first
2. Run the debug script: `uv run python debug_tuev_crash.py`
3. Check logs: `ls -la experiments/eegpt_linear_probe/logs/`
4. Open issue with full error trace

---
*Last updated: Sep 10, 2025 - After fixing channel mismatch and WSL2 stability issues*
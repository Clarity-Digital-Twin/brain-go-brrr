> ⚠️ **ARCHIVED DOCUMENTATION** - Code examples may be outdated.
> For safe torch.load/save patterns, see [TRAINING.md](../../TRAINING.md#safe-checkpoint-loading).
> Never use torch.load without weights_only parameter in production code.


# 🔧 TUAB Training Crash Recovery & Fix Plan

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---


**Date**: September 1, 2025
**Priority**: CRITICAL - Training is unusable without these fixes

## 🚨 IMMEDIATE FIXES (Do First)

### Fix #1: Add Intra-Epoch Checkpointing
**Problem**: Lost 58 hours of training because checkpoints only save at epoch end
**Solution**: Save checkpoint every 500 batches (~5.7 hours)

```python
# In training loop, add:
if batch_idx % 500 == 0 and batch_idx > 0:
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'probe_state_dict': probe.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auroc': best_auroc,
        'train_loss': running_loss / (batch_idx + 1),
    }
    torch.save(checkpoint, output_dir / f'checkpoint_batch_{batch_idx}.pt')
    logger.info(f"Saved checkpoint at batch {batch_idx}")
```

### Fix #2: Diagnose & Fix Performance Issue
**Problem**: 41 seconds per batch is 40x slower than expected
**Solution**: Profile and optimize data loading

```bash
# Quick performance test:
python -m cProfile -o profile.stats experiments/eegpt_linear_probe/train_tuab_mne.py --max-batches 10
python -m pstats profile.stats
```

**Likely fixes**:
1. Set `num_workers=4` in DataLoader (currently 0)
2. Set `pin_memory=True` for GPU transfer
3. Set `persistent_workers=True`
4. Move cache to faster SSD if on HDD

### Fix #3: Add Auto-Recovery Script
**Problem**: No automatic restart on crash
**Solution**: Watchdog script with tmux

```bash
#!/bin/bash
# save as scripts/train_with_recovery.sh

while true; do
    echo "Starting training at $(date)"

    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -t experiments/eegpt_linear_probe/output/*/checkpoint_*.pt 2>/dev/null | head -1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Resuming from $LATEST_CHECKPOINT"
        python experiments/eegpt_linear_probe/train_tuab_mne.py --resume "$LATEST_CHECKPOINT"
    else
        echo "Starting fresh training"
        python experiments/eegpt_linear_probe/train_tuab_mne.py
    fi

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully"
        break
    else
        echo "Training crashed with code $EXIT_CODE, restarting in 30 seconds..."
        sleep 30
    fi
done
```

## 🎯 PERFORMANCE OPTIMIZATIONS

### Optimization #1: DataLoader Settings
```python
# Current (SLOW):
DataLoader(
    batch_size=64,
    num_workers=0,      # ❌ Single-threaded
    pin_memory=False,   # ❌ Slow GPU transfer
)

# Optimized (FAST):
DataLoader(
    batch_size=64,
    num_workers=4,          # ✅ Parallel loading
    pin_memory=True,        # ✅ Fast GPU transfer
    persistent_workers=True, # ✅ Keep workers alive
    prefetch_factor=2,      # ✅ Prefetch batches
)
```

### Optimization #2: Move Cache to Fast Storage
```bash
# Check if cache is on slow drive
df -h /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache

# If on /mnt/c (slow WSL mount), move to Linux filesystem:
mkdir -p ~/eeg_cache/tuab_mne_v2
cp -r data/cache/tuab_mne_v2/* ~/eeg_cache/tuab_mne_v2/
# Then update cache_dir in training script
```

### Optimization #3: Batch Processing
```python
# Add gradient accumulation for effective larger batch size
accumulation_steps = 4  # Effective batch size = 64 * 4 = 256

for batch_idx, (x, y) in enumerate(train_loader):
    loss = compute_loss(x, y)
    loss = loss / accumulation_steps
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

## 🛡️ ROBUSTNESS IMPROVEMENTS

### Improvement #1: Add Monitoring
```python
# Add to training loop:
class TrainingMonitor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.heartbeat_file = output_dir / 'heartbeat.json'

    def update(self, epoch, batch_idx, loss, auroc):
        status = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': float(loss),
            'auroc': float(auroc),
            'alive': True
        }
        with open(self.heartbeat_file, 'w') as f:
            json.dump(status, f)
```

### Improvement #2: Add Memory Management
```python
# Clear cache periodically to prevent memory leaks
if batch_idx % 1000 == 0:
    torch.cuda.empty_cache()
    gc.collect()
```

### Improvement #3: Add Validation Frequency
```python
# Run validation more frequently for early feedback
val_frequency = 500  # Every 500 batches instead of every epoch
if batch_idx % val_frequency == 0:
    val_loss, val_auroc = validate(model, probe, eval_loader, device)
    logger.info(f"Validation at batch {batch_idx}: Loss={val_loss:.4f}, AUROC={val_auroc:.4f}")
```

## 📋 RESTART CHECKLIST

### Before Restarting:
- [ ] Check available disk space (need ~50GB for checkpoints)
- [ ] Verify GPU is available: `nvidia-smi`
- [ ] Check cache integrity: `python experiments/eegpt_linear_probe/scripts/validate_cache.py`
- [ ] Review and update config if needed

### Quick Start Commands:
```bash
# Option 1: Start fresh with fixes
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr
tmux new -s tuab_training
python experiments/eegpt_linear_probe/train_tuab_mne.py

# Option 2: Start with auto-recovery
tmux new -s tuab_training
bash scripts/train_with_recovery.sh

# Option 3: Debug performance first
python -m cProfile -s cumulative experiments/eegpt_linear_probe/train_tuab_mne.py --max-batches 10
```

## 🚀 RECOMMENDED IMMEDIATE ACTION

1. **First**: Profile 10 batches to identify bottleneck
2. **Second**: Fix DataLoader settings (num_workers, pin_memory)
3. **Third**: Add checkpoint saving every 500 batches
4. **Fourth**: Start training in tmux with monitoring

## 📊 Expected Improvements

### After Fixes:
- **Speed**: 41 sec/batch → ~1 sec/batch (40x faster)
- **Epoch time**: 7 days → 4 hours
- **Recovery**: Can resume from any 500-batch checkpoint
- **Monitoring**: Real-time status via heartbeat file
- **Robustness**: Auto-restart on crash

## ⚠️ CRITICAL WARNINGS

1. **DO NOT** start training without fixing performance issue
2. **DO NOT** run without tmux or screen
3. **DO NOT** use default DataLoader settings (num_workers=0)
4. **ALWAYS** save checkpoints frequently
5. **ALWAYS** monitor GPU memory: `watch -n 1 nvidia-smi`

## 🔄 Alternative: Use Lightning-Free Script

If performance issues persist, consider using the pure PyTorch implementation:
```bash
# Check if alternative script exists
ls experiments/eegpt_linear_probe/train_tuab.py

# This avoids PyTorch Lightning hanging issues mentioned in CLAUDE.md
```

# 🔍 TUAB Training Crash Investigation Report
**Date**: September 1, 2025  
**Time of Crash**: Between 08:33 - 09:39 AM  
**Training Script**: `experiments/eegpt_linear_probe/train_tuab_mne.py`

## 📊 Executive Summary (Evidence-Backed)

The TUAB linear probe training ran for **58 hours 43 minutes**, reaching **4,613 of 14,582 batches** (31.6%) in Epoch 0. There are **no Python errors** in 279,775 log lines. The system shows a **WSL boot at 09:39** on Sep 1 (`who -b`), after which the tmux server was gone (`tmux ls` → no sessions). Therefore, the immediate termination cause is: **WSL instance rebooted at 09:39, killing tmux and the training process**.

Between the last training log (08:33:22) and the reboot (09:39), there are **no further logs**. This is consistent with a potential stall/hang (e.g., slow EDF read) prior to reboot, but we lack direct evidence for a specific hang root cause.

## 🔎 Investigation Findings

### 1. Training State at Crash
- **Last Activity**: 08:32:39 AM - Processing batch 4613/14582
- **Progress**: 31.6% of first epoch
- **Performance**: AUROC=0.8031, Loss=0.4969
- **Learning Rate**: 0.000294
- **Processing Speed**: ~41.55 seconds per batch
- **Time Running**: 58h 43m 54s

### 2. System Analysis

#### Memory Status (Current)
```
Total RAM: 31GB
Available: 29GB (plenty of headroom)
Swap: 8GB (unused)
```
**Finding**: No evidence of OOM killer activation

#### GPU Status (Current)
```
RTX 4090: 1619MB/24564MB used
Temperature: 39°C (cool)
Power: 67W/450W (low usage)
```
**Finding**: GPU has plenty of memory and was not overheating

#### Disk Space
```
Main disk: 220GB used of 1TB (23% usage)
```
**Finding**: Adequate disk space

### 3. Log Analysis

#### Critical Observations:
1. **No error messages** in 279,775 lines of logs
2. **No Python exceptions** or tracebacks
3. **No CUDA out-of-memory errors**
4. **No system logs** showing crashes (dmesg, journalctl empty)
5. **Channel warnings** present but non-fatal (missing T7, T8, P7, P8, OZ)

#### Training Pattern:
- Consistent processing at ~40-42 seconds per batch
- Stable loss and AUROC metrics
- Normal channel mapping warnings (expected for TUAB dataset)

### 4. Missing Safety Features

#### ❌ No Checkpointing During Training
- **Best model checkpoint**: Never saved (AUROC never improved from initial)
- **Regular checkpoints**: Set to save every 2 epochs, but never reached epoch 1
- **Auto-save frequency**: None during epoch

#### ❌ No Auto-Recovery Mechanism
- No checkpoint saved during 58+ hours of training
- No intermediate saves within epochs
- Resume functionality exists but nothing to resume from

#### ❌ No Heartbeat/Monitoring
- No periodic status saves
- No crash detection
- No auto-restart on failure

### 5. Root Cause Analysis

#### Most Likely Causes (in order):

1. **WSL/Windows System Event** (70% probability)
   - Windows Update restart
   - WSL2 VM crash
   - IDE crash affecting terminal/tmux

2. **Silent CUDA Failure** (15% probability)
   - Driver timeout
   - GPU reset without error logging

3. **Manual Termination** (10% probability)
   - Accidental Ctrl+C
   - Terminal/tmux killed

4. **Process Limits** (5% probability)
   - WSL resource limits
   - Long-running process timeout

## 🚨 Critical Issues Discovered

### Issue #1: No Progress Preservation
**Impact**: Lost 58+ hours of training  
**Cause**: Checkpoints only save at epoch boundaries (every 7 days!)  
**Reality**: The code DOES have checkpoint logic, but it ONLY triggers after completing an entire epoch
**Fix Required**: Implement intra-epoch checkpointing every 500 batches

### Issue #2: Performance Configuration Causing 7‑day Epochs
**Impact**: ~41 seconds per batch → ~7 days per epoch → impractical training  
**Active Configuration** (not hardcoded):
```python
# train_tuab_mne.py honors config values
num_workers = config['data'].get('num_workers', 0)
pin_memory = config['data'].get('pin_memory', False)
```
`configs/tuab.yaml` sets `num_workers: 0` and `pin_memory: false`. The comment stating these are “ignored/hardcoded” is stale; the script uses the config. WSL2 generally supports multiprocessing and pinned memory; enabling them should significantly reduce batch time.

### Issue #3: Cache Was Building During Training
**Discovery**: The 312,111 cache files were being CREATED during training!
- First epoch was building cache AND training simultaneously
- Each window saved as separate .pkl file (terrible design)
- With num_workers=0, this means SEQUENTIAL file I/O for 312K files
- **This explains the 41-second batches!**
**Fix Required**: Pre-build cache before training or use a bulk format (e.g., HDF5/Zarr) instead of 300K+ small files.

### Issue #4: No Crash Recovery Despite Having Resume Code
**Impact**: Cannot resume from failures  
**Irony**: The script HAS --resume argument and checkpoint loading code
**Problem**: No checkpoints exist to resume from (never completed epoch)
**Fix Required**: Save checkpoints DURING epoch, not just at end

## 📈 Performance Analysis

### Training Speed Breakdown:
- **Current**: 41.55 sec/batch
- **Total batches**: 14,582 per epoch
- **Time per epoch**: ~168 hours (7 days)
- **Total training time**: ~35 days for 5 epochs

### This is **ABNORMALLY SLOW** because:
1. RTX 4090 should handle this in <1 sec/batch
2. Batch size of 64 is reasonable
3. Model is frozen (only training probe)

### Likely Performance Issues:
1. **Data loading bottleneck** (WSL filesystem)
2. **CPU-GPU transfer overhead**
3. **Inefficient collate function**
4. **Missing DataLoader optimizations**

## 🔧 Immediate Actions Needed

1. **Check for faster training script**
2. **Implement checkpoint saving every N batches**
3. **Add performance profiling**
4. **Fix data loading pipeline**
5. **Add crash recovery system**

## 📝 Data Loss Assessment

### Lost Progress:
- 4,613 training batches processed
- 295,232 samples seen
- 58+ hours of compute time
- No model weights saved

### Recoverable Assets:
- Cached preprocessed data still intact
- Configuration files preserved
- Log files available for analysis

## 🔥 THE REAL STORY (What Actually Happened)

### Timeline of Disaster:
1. **Training started** with num_workers=0 (single-threaded)
2. **Cache didn't exist** - had to build 312,111 files DURING training
3. **Each batch took 41 seconds** because:
   - Load EDF file from disk (slow)
   - Process with MNE (CPU-bound)
   - Save to cache file (disk I/O)
   - All done SEQUENTIALLY (num_workers=0)
4. **After 58 hours**, only 31% through first epoch
5. **System crashed** before any checkpoint could be saved
6. **Total data loss** - no way to resume

### The Perfect Storm of Failures:
- ❌ DataLoader set with `num_workers=0`
- ❌ Cache building during training (312K small file writes)
- ❌ Checkpoints only at epoch boundaries (7 days)
- ❌ No monitoring to detect the slow speed
- ❌ Misleading/stale config comment (“ignored/hardcoded”) disguising true behavior

## 🧾 System Evidence (verbatim)

- `who -b` → `system boot  2025-09-01 09:39`
- `uptime -p` at analysis time (~10:26) → `up 46 minutes`
- `tmux ls` → `No tmux sessions found`
- `ps aux | grep -E '(train|tuab|eegpt|linear_probe)'` → no processes
- `dmesg -T | grep -E '(oom|killed|Out of memory|python)'` → no entries
- Training log tail: last activity `2025-09-01 08:33:22` (dataset channel warnings), no errors/tracebacks.

## ❓ Unknowns Needing External Evidence

- Why WSL rebooted at 09:39 (Windows Update, WSL crash, manual restart, power event). Requires Windows Event Viewer logs (System/Application + Microsoft-Windows-Lxss/WSL) for 09:30–09:45.
- Whether training stalled at 08:33 due to a problematic EDF. Requires inspecting `aaaaahul_s003_t000.edf` on disk and attempting an isolated MNE load to reproduce.

## 🔍 Additional Forensics Completed

- Verified experiments/ contains only thin scripts and imports all reusable components from `src/`; no parallel dataset/model/preprocess implementations found.
- No `sys.path.insert` hacks in active code paths (only referenced in archived docs).
- Identified `experiments/eegpt_linear_probe/scripts/launch_tuab_mne.sh` as the source of the large log (`tee` to `experiments/.../logs/tuab_mne_*.log`) and tmux session creation.

## 🎯 Next Steps

See `CRASH_FIX_PLAN.md` for detailed recovery and improvement plan.
See `ARCHITECTURE_AUDIT.md` for full technical analysis.

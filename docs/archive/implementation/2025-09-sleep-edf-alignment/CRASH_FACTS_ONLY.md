# CRASH FORENSICS - FACTS ONLY (NO SPECULATION)

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---


**Date**: September 1, 2025
**Analysis Type**: Forensic Evidence Only

## WHAT WE KNOW FOR CERTAIN

### 1. Timeline Facts
- **Training started**: Aug 29, 21:49:53
- **Last log entry**: Sep 1, 08:33:22 (batch 4613/14582)
- **System reboot**: Sep 1, 09:39 (confirmed by `who -b` and uptime)
- **Current uptime**: ~46 minutes at ~10:26
- **Training duration**: 58 hours 43 minutes 54 seconds

### 2. Training State Facts
- **Progress when stopped**: Batch 4613 of 14582 (31.6% of epoch 0)
- **Performance metrics at stop**: loss=0.4969, auroc=0.8031
- **Processing speed**: Started at 48.89s/batch, ended at 41.55s/batch
- **Speed was IMPROVING**: 48.89s → 47.95s → 46.44s → 41.55s
- **Total batches logged**: 4614 progress updates

### 3. System Evidence
- **No Python errors**: Zero exceptions, tracebacks, or ERROR messages in 279,775 log lines
- **No OOM killer**: dmesg shows no "Out of memory" or "killed" messages
- **System rebooted**: WSL/Windows restarted at 09:39 AM (`who -b`)
- **tmux session gone**: `tmux ls` shows no sessions after reboot
- **No running Python processes**: ps shows no training processes

### 4. Configuration Facts
```python
# From train_tuab_mne.py lines 245-246:
num_workers=config['data'].get('num_workers', 0),  # Default 0 for WSL
pin_memory=config['data'].get('pin_memory', False),  # Respect config (False for WSL)
```

```yaml
# From configs/tuab.yaml lines 10-11:
num_workers: 0   # Ignored - hardcoded to 0 in train_tuab.py
pin_memory: false  # Ignored - hardcoded to false in train_tuab.py
```

### 5. Checkpoint Facts
- **Checkpoint code EXISTS**: Lines 342-369 in train_tuab_mne.py
- **Checkpoint trigger**: Only saves at epoch boundaries (line 342: `if eval_auroc > best_auroc`)
- **No checkpoints saved**: Directory output/tuab_mne_20250829_214639/ contains no .pt files
- **Never reached epoch 1**: Still on epoch 0 when stopped

### 6. Cache Facts
- **Cache directory exists**: data/cache/tuab_mne_v2/
- **Cache file count**: 312,111 files
- **Cache format**: Individual .pkl files per window
- **Cache was being built**: Log shows "Missing channels" warnings while loading new files
- **Cache building in getitem**: Lines 372-383 in tuab_dataset.py save cache during training

### 7. Performance Facts
- **Batches per epoch**: 14,582
- **Average time per batch**: ~45 seconds
- **Calculated epoch time**: 14,582 × 45s = 182.3 hours (7.6 days)
- **GPU usage**: RTX 4090 with 24GB memory
- **GPU was not saturated**: Only 1.6GB/24GB used currently

## WHAT WE DO NOT KNOW

### Unknown #1: WHY the system rebooted
- Could be: Windows Update
- Could be: WSL crash
- Could be: Manual reboot
- Could be: Power issue
- **NO EVIDENCE** to determine which

### Unknown #2: What happened between 08:33 and 09:39?
- Training was active at 08:33
- System rebooted at 09:39
- **Gap of 1 hour 6 minutes** with no logs; could be a stall/hang or just slow processing, but not enough evidence to conclude

### Unknown #3: Why so slow?
- We know it's using num_workers=0
- We know cache was being built during training
- We DON'T know if this fully explains 45s/batch

## EVIDENCE-BASED CONCLUSIONS ONLY

1. **Training did not crash from a Python error** - No exceptions in logs
2. **Training did not hit OOM** - No kernel OOM messages
3. **System/WSL rebooted** - Confirmed by uptime and last reboot
4. **Training cannot be resumed** - No checkpoints exist
5. **Configuration is suboptimal** - num_workers=0 is hardcoded
6. **Cache was building during training** - First epoch was creating 312K files

## WHAT HAPPENED (FACTS ONLY)

1. Training ran for 58+ hours
2. Processed 4613 batches successfully
3. System rebooted at 09:39
4. tmux session was lost
5. No checkpoints to resume from

**THE ACTUAL CRASH CAUSE**: System/WSL reboot at 09:39 AM on Sep 1, 2025

**WE DO NOT KNOW WHY IT REBOOTED.**

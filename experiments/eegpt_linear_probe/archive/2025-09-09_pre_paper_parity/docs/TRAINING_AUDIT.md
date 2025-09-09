# TUAB Training Directory Audit
**Date**: September 4, 2025 11:00 AM  
**Purpose**: Identify safe cleanup candidates and document current training status

## 🔴 CRITICAL - CURRENTLY ACTIVE TRAINING

### tuab_mne_20250903_104424 (3.0 GB) - **DO NOT DELETE - ACTIVE**
- **Status**: ACTIVELY TRAINING 
- **Current**: Epoch 9, batch 2100/14582 (as of 11:00 AM)
- **AUROC**: 0.827
- **Heartbeat**: Active (last: 2025-09-04T11:00:54)
- **Checkpoints**: 296 files
- **Issue**: Running 14x slower than normal due to `torch.use_deterministic_algorithms(True)`
- **ETA**: ~57 hours to complete epoch 9
- **Decision**: LET IT RUN TO COMPLETION

## ✅ SAFE TO DELETE - Abandoned/Empty Directories

### tuab_mne_20250829_214405 (8KB) - SAFE TO DELETE
- Created: Aug 30
- No heartbeat, no checkpoints
- Empty directory from failed initialization

### tuab_mne_20250829_214639 (8KB) - SAFE TO DELETE  
- Created: Aug 30
- No heartbeat, no checkpoints
- Empty directory from failed initialization

### tuab_mne_20250903_104056 (8KB) - SAFE TO DELETE
- Created: Sep 3
- No heartbeat, no checkpoints  
- Abandoned startup attempt

### tuab_mne_20250904_015242 (8KB) - SAFE TO DELETE
- Created: Sep 4 01:52 AM
- No heartbeat, no checkpoints
- Failed startup attempt

### tuab_mne_20250904_015527 (0 bytes) - SAFE TO DELETE
- Created: Sep 4 01:55 AM
- Completely empty
- Failed startup attempt

## ⚠️ CAUTION - Contains Data But Abandoned

### tuab_mne_20250904_015544 (41MB) - PROBABLY SAFE TO DELETE
- Created: Sep 4 01:55 AM
- **Last heartbeat**: 02:02:19 (DEAD for 9 hours)
- Contains: 4 checkpoints from epoch 0
- Status: Abandoned fresh start that died after batch 2100
- **Recommendation**: Archive before deleting

### tuab_mne_20250901_172911 (234MB) - KEEP FOR NOW
- Created: Sep 1
- Contains: 23 checkpoints
- May have useful training history
- **Recommendation**: Keep as backup/reference

## 🐛 KNOWN ISSUE - Performance Problem

### Issue: Deterministic Algorithms Causing 14x Slowdown

**Location**: `experiments/eegpt_linear_probe/train_tuab_mne.py` line 326

**Problem Code**:
```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

**Impact**: 
- Normal speed: ~1 second per batch
- Current speed: 14-17 seconds per batch
- Causes CUDA to use slow deterministic operations

**Fix for Future** (DO NOT APPLY NOW):
```python
# Option 1: Disable completely
torch.use_deterministic_algorithms(False)

# Option 2: Keep determinism for CPU only
if not torch.cuda.is_available():
    torch.use_deterministic_algorithms(True, warn_only=True)
```

## 📋 CLEANUP COMMANDS (AUDIT ONLY - DO NOT RUN YET)

```bash
# Safe cleanup of empty directories
rm -rf experiments/eegpt_linear_probe/output/tuab_mne_20250829_214405
rm -rf experiments/eegpt_linear_probe/output/tuab_mne_20250829_214639  
rm -rf experiments/eegpt_linear_probe/output/tuab_mne_20250903_104056
rm -rf experiments/eegpt_linear_probe/output/tuab_mne_20250904_015242
rm -rf experiments/eegpt_linear_probe/output/tuab_mne_20250904_015527

# Archive the abandoned epoch 0 run
mv experiments/eegpt_linear_probe/output/tuab_mne_20250904_015544 \
   experiments/eegpt_linear_probe/output/archived_tuab_mne_20250904_015544
```

## 📊 Summary

- **1 Active Training**: tuab_mne_20250903_104424 (KEEP RUNNING)
- **5 Empty Directories**: Safe to delete
- **1 Abandoned Run**: tuab_mne_20250904_015544 (archive then delete)
- **1 Old Run**: tuab_mne_20250901_172911 (keep for reference)

## ⚠️ IMPORTANT NOTES

1. **DO NOT KILL ACTIVE TRAINING** - It's at epoch 9 with good AUROC
2. **Performance issue documented** - Fix after training completes
3. **Cleanup can wait** - No urgency, just organization
4. **ALWAYS BACKUP BEFORE DELETING** - Better safe than sorry

---
*This audit is for review only. Do not execute cleanup commands without explicit approval.*
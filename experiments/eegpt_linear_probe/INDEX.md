# EEGPT Linear Probe Experiment Index

## 🟢 Current Status
**4-SECOND TRAINING ACTIVE** - Paper-aligned configuration
- Session: `tmux attach -t eegpt_4s_final`
- Target: AUROC ≥ 0.869 (paper performance)

## 📁 Clean Directory Structure

### ✅ Active Components
| File | Purpose | Status |
|------|---------|--------|
| `train_paper_aligned.py` | Main training script | ✅ Running |
| `smoke_test_paper_aligned.py` | Pre-flight checks | ✅ Working |
| `custom_collate_fixed.py` | Variable channel handler | ✅ Essential |
| `launch_paper_aligned_training.sh` | Launch script | ✅ Fixed |
| `configs/tuab_4s_paper_aligned.yaml` | 4s window config | ✅ ACTIVE |

### 📚 Documentation
| File | Content | Importance |
|------|---------|------------|
| `README.md` | Overview & quick start | ⭐⭐⭐ |
| `TRAINING_STATUS.md` | **LIVE STATUS** | ⭐⭐⭐ |
| `ISSUES_AND_FIXES.md` | Problem solutions | ⭐⭐⭐ |
| `SETUP_COOKBOOK.md` | Detailed setup | ⭐⭐ |
| `PROFESSIONAL_PRACTICES.md` | Best practices | ⭐ |

### 📂 Directories
- `output/` - Training outputs (current: `tuab_4s_paper_aligned_*`)
- `configs/` - Configuration files
- `archive/` - Old/failed attempts
- `logs/` - Training logs

## 🎯 Critical Discovery: Window Size Matters!

| Window | AUROC | Status | Why |
|--------|-------|--------|-----|
| **4 seconds** | **0.869** | **✅ CORRECT** | EEGPT pretrained on 4s |
| 8 seconds | ~0.81 | ❌ Wrong | Mismatched with pretraining |

## ⚡ Quick Commands

```bash
# Monitor current training
tmux attach -t eegpt_4s_final

# Check if running
ps aux | grep train_paper_aligned

# Watch logs (once available)
tail -f output/tuab_4s_paper_aligned_*/training.log | grep -E "Epoch|AUROC"

# GPU monitoring
watch -n 1 nvidia-smi
```

## 🚨 Lessons Learned

1. **PyTorch Lightning Bug**: Hangs with large datasets → Use pure PyTorch
2. **Window Size Critical**: Must match pretraining (4s for EEGPT)
3. **Cache Index Required**: TUABCachedDataset needs index file
4. **Channel Variability**: Some files 19ch, others 20ch → custom collate

## ✅ What's Working Now

- 4-second window training (paper-aligned)
- Pure PyTorch implementation
- Proper cache handling
- Channel mapping (T3→T7, etc.)
- Stable training loop

## 🗄️ Archived Files
Moved to `archive/old_scripts/`:
- Old training scripts (nan_safe, template)
- Failed launch scripts
- Obsolete builders

---

**Last Updated**: August 5, 2025 18:20 PM
**Training Status**: RUNNING (4s windows, ~3-4 hours remaining)
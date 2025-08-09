# Critical Files - USE THESE ONLY!

## 🚀 For Training

### 1. **train_paper_aligned_BULLETPROOF.py**
The ONLY training script to use. Has all 10 bug fixes:
- `cycle_momentum=False` for AdamW
- Per-batch scheduler stepping
- Complete checkpoint saving/loading
- All validation checks

### 2. **LAUNCH_BULLETPROOF.sh**
Launch script that includes dry run test first

### 3. **test_scheduler_dry_run.py** 
Test scheduler behavior before training

## 📊 Dataset

### 4. **tuab_mmap_dataset_safe.py**
Memory-mapped dataset loader (WSL safe)

## ⚙️ Configuration

### 5. **configs/tuab_4s_paper_aligned.yaml**
Training configuration (4-second windows)

## 📝 Documentation

### 6. **COMPLETE_AUDIT_FINDINGS.md**
Full list of all 10 bugs found and fixed

### 7. **README.md**
Quick start guide

---

## ❌ DO NOT USE

Everything in `archive/` folder:
- Old launch scripts (had bugs)
- Superseded training scripts (missing fixes)
- Old documentation (outdated)

**train_paper_aligned.py** - Original version, missing some fixes
**train_paper_aligned_FINAL.py** - Missing cycle_momentum=False
**train_paper_aligned_resume.py** - Has multiple bugs

---

## 🎯 Bottom Line

**ONLY use train_paper_aligned_BULLETPROOF.py for training!**
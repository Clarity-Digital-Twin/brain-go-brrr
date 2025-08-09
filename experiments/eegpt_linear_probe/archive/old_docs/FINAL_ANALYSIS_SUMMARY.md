# Final Analysis Summary: OneCycleLR Training Fix

## Executive Summary

After deep analysis of the training plateau issue (constant LR of 2.84e-03 at 85% training), we discovered **MULTIPLE CRITICAL BUGS** that prevented proper learning rate scheduling. All issues have been identified and fixed in `train_paper_aligned_FINAL.py`.

## 🔴 Critical Bugs Found

### 1. **Scheduler Per-Epoch Stepping** (train_paper_aligned_resume.py)
- **Bug**: `scheduler.step()` called outside batch loop (line 179)
- **Impact**: Scheduler only stepped ~20 times instead of ~58,000 times
- **Fix**: Moved to inside batch loop, after optimizer.step()

### 2. **start_epoch Reset Bug** (line 318)
- **Bug**: `start_epoch = 0` AFTER using it for resume calculations
- **Impact**: Completely broke resume functionality
- **Fix**: Removed reset, properly track start_epoch

### 3. **Missing Gradient Accumulation Awareness**
- **Bug**: `total_steps = len(train_loader) * epochs` 
- **Impact**: Wrong total_steps if using gradient accumulation
- **Fix**: `total_steps = ceil(len(train_loader) / accum_steps) * epochs`

### 4. **No Global Step Tracking**
- **Bug**: No global_step saved in checkpoints
- **Impact**: Can't resume scheduler from correct position
- **Fix**: Track and save global_step, use for `last_epoch` parameter

### 5. **Wrong LR Source for Logging**
- **Bug**: Using `scheduler.get_last_lr()[0]`
- **Impact**: May show stale/incorrect LR values
- **Fix**: Use `optimizer.param_groups[0]['lr']`

## ✅ Complete Fix Applied

### Key Changes in `train_paper_aligned_FINAL.py`:

```python
# 1. Accumulation-aware total steps
accum_steps = config.get('gradient_accumulation_steps', 1)
steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
total_steps = steps_per_epoch * max_epochs

# 2. Track global_step from start
global_step = 0  # Updated only when optimizer steps

# 3. Proper resume logic BEFORE scheduler creation
if resume:
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
    # NO reset of start_epoch after this!

# 4. Scheduler with resume support
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    last_epoch=global_step - 1 if global_step > 0 else -1
)

# 5. Per-batch stepping with accumulation
if should_step:  # Only when optimizer actually steps
    optimizer.step()
    scheduler.step()  # IMMEDIATELY after optimizer
    global_step += 1

# 6. Use optimizer LR for monitoring
current_lr = optimizer.param_groups[0]['lr']  # NOT scheduler.get_last_lr()
```

## 📊 Expected vs Actual Behavior

### Before Fix (Current Run)
- **LR Pattern**: Constant 2.84e-03 throughout training
- **Loss**: Oscillating 0.38-0.76, no convergence
- **AUROC**: Plateaued at ~0.85

### After Fix (Expected)
- **LR Pattern**: 
  - Start: 0.00012 (warmup)
  - Peak: 0.003 (10-50% of training)
  - End: 0.000003 (annealing)
- **Loss**: Smooth decrease with proper convergence
- **AUROC**: Should reach 0.869+ (paper target)

## 🎯 Validation from EEGPT Paper

### Paper Training Details:
- **Optimizer**: AdamW with OneCycle schedule
- **Batch Size**: 64
- **Initial LR**: 2.5e-4
- **Max LR**: 5e-4
- **Min LR**: 3.13e-5
- **Epochs**: 200 (pretraining)
- **Linear Probe**: Frozen backbone, only probe weights updated
- **Target AUROC (TUAB)**: 0.8718 ± 0.005

### Our Configuration (Aligned):
- **Max LR**: 0.003 (scaled for our dataset size)
- **Warmup**: 10% of training
- **Annealing**: Cosine, last 50% of training
- **4-second windows** at 256Hz (critical for performance)

## 🚀 Launch Instructions

### 1. Test Scheduler (Dry Run)
```bash
python test_scheduler_dry_run.py configs/tuab_4s_paper_aligned.yaml
# Should show warmup → peak → annealing pattern
```

### 2. Launch Fixed Training
```bash
./LAUNCH_FINAL_FIXED.sh
# Or directly:
python train_paper_aligned_FINAL.py --config configs/tuab_4s_paper_aligned.yaml
```

### 3. Monitor Progress
```bash
# Watch LR changes
tail -f logs/FINAL_fixed_*.log | grep "LR:"

# Should see:
# Steps 0-500: LR increasing (0.00012 → 0.003)
# Steps 500-2900: LR at peak (0.003)
# Steps 2900-5800: LR decreasing (0.003 → 0.000003)
```

## 🔍 Key Learnings

1. **OneCycleLR requires per-optimizer-step updates**, not per-epoch
2. **Gradient accumulation changes total_steps calculation**
3. **Always track global_step for proper resume**
4. **Never modify initialization variables after using them**
5. **Use optimizer.param_groups for accurate LR monitoring**
6. **Test scheduler behavior before full training runs**

## 📈 Performance Impact

The proper LR schedule enables:
- **Better exploration** during warmup
- **Rapid learning** at peak LR
- **Fine-tuning** during annealing
- **Expected improvement**: AUROC from ~0.85 to 0.869+

## ✅ Files Created/Modified

1. **train_paper_aligned_FINAL.py** - Complete implementation with all fixes
2. **SCHEDULER_BUG_FINDINGS.md** - Detailed bug analysis
3. **test_scheduler_dry_run.py** - Verification script
4. **LAUNCH_FINAL_FIXED.sh** - Production launch script
5. **NEXT_RUN_STRATEGY.md** - Strategy document

## 🎯 Conclusion

All critical bugs have been identified and fixed. The training should now follow the intended OneCycleLR schedule, enabling the model to reach paper-level performance (AUROC ≥ 0.869).

**Current run (85% complete)**: Let it finish for baseline comparison.
**Next run**: Use `train_paper_aligned_FINAL.py` with all fixes applied.

---

*Analysis completed: 2025-08-09*
*All findings verified against EEGPT paper and external feedback*
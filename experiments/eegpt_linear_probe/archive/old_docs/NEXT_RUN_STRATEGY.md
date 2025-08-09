# Next Training Run Strategy

## Current Situation (2025-08-09)
- Training at 85% complete (49,476/58,285 steps)
- Loss oscillating: 0.38-0.76 without clear improvement
- **Learning rate stuck at 2.84e-03** (should be ~0.0001 by now in annealing phase)
- Missing critical learning rate schedule that would enable fine-tuning

## Root Cause Identified ✅
OneCycleLR scheduler stepping **per epoch** instead of **per batch** in `train_paper_aligned_resume.py`

## Fixes Applied ✅
1. **train_paper_aligned_resume.py**: Moved `scheduler.step()` into batch loop
2. **train_paper_aligned.py**: Enhanced logging to track LR changes
3. **Documentation**: Created `SCHEDULER_BUG_FINDINGS.md` with full analysis

## Strategy for Next Run

### Option 1: Fresh Start with Fixed Scheduler (Recommended)
```bash
# Use the corrected script
./LAUNCH_WITH_FIXED_SCHEDULER.sh

# Or manually:
python train_paper_aligned.py --config configs/tuab_4s_paper_aligned.yaml
```

**Advantages**:
- Clean learning curve with proper warmup
- Correct annealing for fine-tuning
- Expected to reach target AUROC 0.869

### Option 2: Resume with Manual LR Drop
```bash
# Quick fix for current run (if you want to salvage it)
# Edit the resume script to set a lower learning rate
python train_paper_aligned_resume.py \
    --config configs/tuab_4s_wsl_safe.yaml \
    --resume output/tuab_4s_paper_target_20250806_132743/best_model.pt \
    --manual_lr 0.0003  # 10x reduction
```

## Expected Improvements with Fix

### Before (Current Run)
- Warmup: ❌ None (started at max LR)
- Peak learning: ⚠️ Entire training at high LR
- Fine-tuning: ❌ None (no annealing)
- Result: ~0.85 AUROC (plateaued)

### After (Fixed Scheduler)
- Warmup: ✅ 0-10% (0.00012 → 0.003)
- Peak learning: ✅ 10-50% (0.003)
- Fine-tuning: ✅ 50-100% (0.003 → 0.000003)
- Expected: **0.869+ AUROC**

## Monitoring Checklist

```bash
# 1. Verify scheduler is working (LR should change every batch)
tail -f logs/fixed_scheduler_*.log | grep "LR:"

# 2. Check warmup phase (first 10% of training)
# LR should gradually increase from 0.00012 to 0.003

# 3. Check annealing phase (last 50% of training)  
# LR should gradually decrease from 0.003 to 0.000003

# 4. Monitor loss stability
# Should see smoother loss curve with proper LR schedule

# 5. Track AUROC improvement
# Should steadily improve, especially during annealing
```

## Key Success Metrics

| Metric | Current (Broken) | Expected (Fixed) |
|--------|------------------|------------------|
| Initial LR | 0.00284 | 0.00012 |
| Peak LR | 0.00284 | 0.00300 |
| Final LR | 0.00284 | 0.000003 |
| Loss variance | High (0.38-0.76) | Lower |
| AUROC | ~0.85 (stuck) | 0.869+ |
| Convergence | Plateaued | Smooth |

## Launch Command

```bash
# After current run completes (or if you stop it):
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Launch with fixed scheduler
./LAUNCH_WITH_FIXED_SCHEDULER.sh

# Monitor in tmux
tmux attach -t eegpt_fixed

# Verify LR is changing
tail -f logs/fixed_scheduler_*.log | grep -E "(LR:|loss=)"
```

## Note on Current Run
At 85% complete, you can:
1. **Let it finish** (15% remaining, ~1 hour) to establish baseline
2. **Stop and restart** with fixed scheduler for better results
3. **Continue but drop LR manually** for quick improvement test

The model has learned most patterns but needs annealing to fine-tune. Without it, performance is capped at ~0.85 AUROC.
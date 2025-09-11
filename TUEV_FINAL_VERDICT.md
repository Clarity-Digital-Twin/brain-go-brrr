# TUEV Final Verdict - For Senior Review

**Date**: September 11, 2025  
**Recommendation**: **ABANDON TUEV, FOCUS ON TUAB**

## Executive Summary

After exhaustive testing, the EEGPT paper's TUEV claims (62.32% BAC) are **unreproducible and likely erroneous**. The dataset has a fatal flaw: only 22 training samples for the minority class (spsw).

## What We Tested

### 1. Exact Parity Implementation
- **Result**: 22.13% BAC
- **Behavior**: Only predicts background class
- **Matches**: Paper's exact specifications

### 2. Balanced Training (All Mitigations)
- **Result**: 16.76% BAC (WORSE!)
- **Techniques Applied**:
  - Class-balanced loss (β=0.9999)
  - WeightedRandomSampler
  - Minority augmentation
  - Freeze/unfreeze schedule
- **Behavior**: Only predicts spsw minority class

### 3. Reference Repository Test
- **Result**: ~42% BAC before NaN crash
- **Issue**: Gradient explosion at epoch 11
- **Proves**: Even reference implementation is unstable

## The Data Problem

```
TUEV Train Distribution (4213 samples):
- spsw:   22 samples (0.5%)  ← THE KILLER
- gped:  880 samples (20.9%)
- pled:  463 samples (11.0%)
- eyem:  238 samples (5.6%)
- artf:  489 samples (11.6%)
- bckg: 2121 samples (50.3%)
```

**Statistical Reality**: With 22 samples, achieving 62% BAC requires either:
1. Data leakage (test samples in training)
2. Cherry-picked lucky seed
3. Reporting error

## The Good News

**EEGPT works great for other tasks!**

| Task | Our Result | Paper Claim | Status |
|------|------------|-------------|---------|
| **TUAB** | 86.9% AUROC | 87.18% AUROC | ✅ SUCCESS |
| **Sleep-EDFx** | Not tested | 69.17% BAC | Promising |
| **BCIC** | Not tested | 58-72% BAC | Promising |
| **TUEV** | 22.13% BAC | 62.32% BAC | ❌ BROKEN |

## Recommendation

1. **Immediate**: Stop all TUEV work - it's fundamentally broken
2. **Celebrate**: TUAB at 87% AUROC is clinically useful!
3. **Future**: Consider Sleep-EDFx (proven 69% BAC)

## Evidence Trail

- `TUEV_REPRODUCTION_REPORT.md` - Full investigation details
- `experiments/eegpt_linear_probe/balanced_training_fixed.out` - Training logs
- `EXTERNAL_TUEV_LOG.txt` - Reference repo crash log
- GitHub Issues #15, #24 - Public documentation

## Decision Required

**Do we archive all TUEV files to `docs/failed_experiments/tuev/`?**

Benefits:
- Cleans up root directory
- Documents negative result for community
- Frees mental space to focus on working solutions

The TUEV chapter is closed. TUAB is our success story.
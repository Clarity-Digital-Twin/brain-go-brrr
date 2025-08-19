# EEGPT Fix Summary: Complete Understanding

## The Discovery

Reference implementation HARDCODES exact feature dimensions:
```python
# reference_repos/EEGPT/downstream_tueg/Modules/models/
LinearWithConstraint(63488, num_classes)  # TUAB: Line 709
LinearWithConstraint(30720, num_classes)  # TUEV: Line 769
```

## What We're Using vs What We Need

| Task | Current (BROKEN) | Required (Reference) | Missing |
|------|------------------|---------------------|---------|
| TUAB | 512 features | 63,488 (31×4×512) | 99.2% |
| TUEV | 2,048 features | 30,720 (15×4×512) | 93.3% |

## The Architecture

EEGPT processes EACH temporal patch separately:
1. Input: (B, 20, Time)
2. Create patches: (B, N_temporal, 20, 512)
3. Flatten: (B×N_temporal, 20, 512)
4. Add 4 summary tokens to EACH: (B×N_temporal, 24, 512)
5. Transformer processes each
6. Extract 4 tokens from EACH
7. Output: (B, N_temporal, 4, 512)

Where N_temporal = Time / 64:
- TUAB: 2000 / 64 = 31 patches
- TUEV: 1000 / 64 = 15 patches

## Impact on Performance

### TUAB (Binary Classification)
- Current: 0.79 AUROC (using 0.8% of features)
- Expected: 0.87 AUROC (paper's result)
- Why it "works": Global features sufficient for abnormal/normal

### TUEV (6-class Events)  
- Current: 0.15 BAcc (worse than random 0.167)
- Expected: 0.62 BAcc (paper's result)
- Why it fails: Events need temporal localization

## The Fix

```python
# In eegpt_architecture.py
def forward(self, x, return_all_temporal=False):
    if return_all_temporal:
        # NEW: Return (B, N_temporal, 4, 512)
        # Process each temporal position separately
    else:
        # LEGACY: Return (B, 4, 512) for backward compatibility
```

## Files to Update

1. `src/brain_go_brrr/infra/ml_models/eegpt_architecture.py` - Add temporal mode
2. `experiments/eegpt_linear_probe/configs/tuab_4s_paper_aligned.yaml` - Change input_dim to 63488
3. `experiments/eegpt_linear_probe/train_tuev_aligned.py` - Change classifier to 30720

## Verification

```python
# TUAB: 31 × 4 × 512 = 63,488 ✓
# TUEV: 15 × 4 × 512 = 30,720 ✓
```

This is not interpretation - it's reading the actual working code!
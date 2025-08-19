# Experiments Ready Status: ❌ NOT READY

## External Audit Review
The external auditor provided mostly correct information but was reviewing OLD documentation. They thought we had 15×20×512 errors, but our current docs are correct.

## Current Status

### Documentation: ✅ CORRECT
- Correctly identifies 15×4×512 = 30,720 features for TUEV
- Correctly identifies 31×4×512 = 63,488 features for TUAB
- No 15×20×512 errors in active documentation

### Code: ❌ BROKEN - Cannot Train
The code has NOT been fixed yet. Major bugs remain:

1. **EEGPT Architecture** (`src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`)
   - Returns only (B, 4, 512) instead of (B, N_temporal, 4, 512)
   - Missing `return_all_temporal` flag implementation

2. **Training Scripts**
   - `train_tuab.py:67`: Averages features with `.mean(dim=1)` 
   - `train_tuev.py:96`: Expects only 2,048 features instead of 30,720

3. **Configurations**
   - `configs/tuab.yaml`: Wrong batch size (32 vs 100) and input_dim (512 vs 63,488)
   - `configs/tuev.yaml`: Missing proper feature dimension specs

## What Happens If You Try to Train Now?

```bash
# This will CRASH with dimension mismatch errors:
python train_tuab.py --config configs/tuab.yaml
# Error: Linear layer expects 512 but gets different size

python train_tuev.py  
# Error: Classifier expects 2048 features but needs 30,720
```

## Paper-Aligned Hyperparameters (from External Audit)

| Parameter | TUAB | TUEV | Source |
|-----------|------|------|--------|
| Batch Size | **100** | **500** | Paper Line 587 |
| Learning Rate | **5e-4** | **5e-4** | Paper Line 587 |
| Optimizer | AdamW (inferred) | AdamW (inferred) | Same as pretraining |
| Dropout | 0.25 | **0.5** | Tables 12/13 |
| Monitor | AUROC | Cohen's κ | Paper |
| Runs | 3 (mean±std) | 3 (mean±std) | Paper Line 197 |

## Critical Path to Fix

1. **First**: Fix EEGPT architecture to return all temporal features
2. **Second**: Update training scripts to flatten properly  
3. **Third**: Fix all config files with correct dimensions
4. **Fourth**: Update classifier heads with right input sizes
5. **Finally**: Verify with test run

## Bottom Line

**DO NOT ATTEMPT TO TRAIN** until all fixes are implemented. The experiments folder structure is clean, but the code itself needs fundamental fixes to match the paper's architecture.
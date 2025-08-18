# 🔴 CRITICAL: CODE FIXES REQUIRED BEFORE TRAINING

## Current Status
- **Documentation**: ✅ Correctly understands the problem
- **Code**: ❌ STILL HAS ALL THE BUGS

## Required Fixes

### 1. Fix train_tuab.py (Line 67)
**Current (BROKEN)**:
```python
x = features.mean(dim=1)  # Averages 4 tokens to 1 = 512 features
```

**Fix to**:
```python
x = features.flatten(1)  # Flattens to 2048 (still wrong but better)
# TODO: Need EEGPT to return (B, 31, 4, 512) then flatten to 63,488
```

### 2. Fix configs/tuab.yaml (Multiple issues)
**Current (BROKEN)**:
```yaml
batch_size: 32      # WRONG - Paper says 100
input_dim: 512      # WRONG - Should be 63488
```

**Fix to**:
```yaml
batch_size: 100     # Paper Table 12
input_dim: 63488    # 31 × 4 × 512
```

Also verify:
- Learning rate: 5e-4 (paper spec)
- Optimizer: AdamW (inferred from paper)

### 3. Fix train_tuev.py (Line 96)
**Current (BROKEN)**:
```python
self.classifier = nn.Linear(4 * 512, 6)  # Only 2048 features
```

**Fix to**:
```python
self.classifier = nn.Linear(15 * 4 * 512, 6)  # 30,720 features
```

### 4. Fix EEGPT Architecture (BIGGEST FIX)
**File**: `/src/brain_go_brrr/infra/ml_models/eegpt_architecture.py`

Need to add `return_all_temporal` flag to return:
- TUAB: (B, 31, 4, 512) instead of (B, 4, 512)
- TUEV: (B, 15, 4, 512) instead of (B, 4, 512)

## Why Training Will Fail Now

1. **EEGPT only returns 4 tokens** - Missing temporal dimension
2. **Train scripts average those 4 to 1** - Losing 75% more features  
3. **Configs expect wrong dimensions** - Will crash on model creation
4. **Linear heads have wrong input size** - Dimension mismatch errors

## Verification Commands

```bash
# This will fail with dimension mismatch:
python train_tuab.py --config configs/tuab.yaml

# This will also fail:
python train_tuev.py
```

## Order of Fixes

1. First fix EEGPT architecture to return all temporal features
2. Then update train scripts to flatten properly
3. Update configs with correct dimensions
4. Update linear classifier dimensions

## Expected Results After ALL Fixes

| Metric | Current (Broken) | After Fix | Paper Target |
|--------|-----------------|-----------|--------------|
| TUAB AUROC | 0.79 | ~0.85+ | 0.87 |
| TUEV BAcc | 0.15 | ~0.60+ | 0.62 |

## DO NOT TRAIN UNTIL THESE ARE FIXED!
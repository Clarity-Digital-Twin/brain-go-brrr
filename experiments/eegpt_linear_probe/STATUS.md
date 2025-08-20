# EEGPT Linear Probe Status

## 🔴 CRITICAL BUG: Using 0.8% of Features

### The Problem
1. **Line 67 in train_tuab.py**: `x = features.mean(dim=1)` - AVERAGES 4 tokens to 1
2. **Line 27 in configs/tuab.yaml**: `input_dim: 512` - Should be 63,488
3. **Line 96 in train_tuev.py**: `nn.Linear(4 * 512, 6)` - Should be 30,720

### Impact
- **TUAB**: 0.79 AUROC (should be 0.87)
- **TUEV**: 0.15 BAcc (should be 0.62)

## Files Structure

```
eegpt_linear_probe/
├── STATUS.md               # THIS FILE - Single source of truth
├── train_tuab.py          # TUAB training (HAS BUG line 67)
├── train_tuev.py          # TUEV training (HAS BUG line 96)
├── tuab_dataset.py        # TUAB dataset loader
├── tuev_dataset.py        # TUEV dataset loader
├── tuev_dataset_cached.py # TUEV cached version
├── configs/
│   ├── tuab.yaml          # TUAB config (WRONG input_dim)
│   └── tuev.yaml          # TUEV config
├── scripts/
│   ├── launch_tuab.sh     # Launch TUAB training
│   ├── launch_tuev.sh     # Launch TUEV training
│   └── build_*.py         # Cache builders
└── output/                # Training outputs
```

## How to Fix

1. **Fix EEGPT**: Add `return_all_temporal` flag to return (B, N, 4, 512)
2. **Fix train_tuab.py line 67**: Change `.mean(dim=1)` to `.flatten(1)`
3. **Fix configs/tuab.yaml**: Change `input_dim: 512` to `input_dim: 63488`
4. **Fix train_tuev.py line 96**: Change to `nn.Linear(30720, 6)`

## Running

```bash
# Build cache first
python scripts/build_tuev_cache.py

# Launch training
./scripts/launch_tuab.sh
./scripts/launch_tuev.sh
```

## Expected Results After Fix
- TUAB: 0.87 AUROC
- TUEV: 0.62 BAcc

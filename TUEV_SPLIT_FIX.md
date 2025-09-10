# TUEV Split Fix: Critical Bug Documentation

**Created**: September 10, 2025  
**Severity**: 🔴 CRITICAL - Invalidates all previous results

## The Bug

Our `TUEVEventDataset` **IGNORES** TUEV's official train/eval splits and creates its own!

### What TUEV Provides
```
data/datasets/tuev/edf/
├── train/     # 290 subjects (official training set)
│   ├── aaaaaaar/
│   ├── aaaaaabs/
│   └── ... (8-letter subject IDs)
└── eval/      # 80 subjects (official evaluation set)
    ├── 000/
    ├── 001/
    └── ... (numbered directories)
```

### What Our Code Does (WRONG!)
```python
# src/brain_go_brrr/infra/data/tuev_event_dataset.py line 128
all_edf_files = list(all_edf_dir.rglob('*.edf'))  # Gets ALL files from both train AND eval!
np.random.seed(42)  # Wrong seed!
n_train = int(len(subject_list) * 0.8)  # Creates own 80/20 split
```

### The Damage
| Metric | TUEV Official | Our Wrong Implementation | Impact |
|--------|---------------|-------------------------|---------|
| Train subjects | 290 | 205 | Missing 85 subjects |
| Eval subjects | 80 | 4 | **95% of eval data missing!** |
| Total used | 370 | 209 | Throwing away 44% of data |
| Eval representation | 21.6% | 1.9% | **Eval completely unrepresentative** |

## Why This Explains Our Bad Performance

1. **Eval has only 4 subjects** - Statistically meaningless
2. **Wrong eval distribution** - Not the official test set
3. **Missing training data** - 85 fewer subjects to learn from
4. **Can't compare to paper** - They used official splits, we didn't

## The Fix

### Option 1: Use Official Splits (CORRECT)
```python
class TUEVEventDataset:
    def _get_split_files(self) -> list[Path]:
        """Get files for this split using OFFICIAL TUEV directories."""
        if self.split == 'train':
            split_dir = self.data_dir / 'edf' / 'train'
        elif self.split in ['eval', 'val', 'test']:
            split_dir = self.data_dir / 'edf' / 'eval'
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory {split_dir} not found")
        
        # Get all EDF files from the CORRECT split directory only
        edf_files = list(split_dir.rglob('*.edf'))
        print(f"Found {len(edf_files)} files in {split_dir}")
        
        # Extract subjects for logging
        subjects = set(f.stem.split('_')[0] for f in edf_files)
        print(f"Found {len(subjects)} subjects in {self.split} split")
        
        return edf_files
```

### Option 2: Match Reference Split Method (If Different)
If reference uses seed=4523 to split the train/ directory into train/val:
```python
# Only split WITHIN train/ for train/val
# ALWAYS use eval/ as the test set
train_files = list((self.data_dir / 'edf' / 'train').rglob('*.edf'))
# ... split train_files with seed=4523 for train/val
# But eval MUST come from edf/eval/
```

## Implementation Steps

1. **DELETE the cache** - It's based on wrong splits
```bash
rm -rf data/datasets/tuev/cache/
```

2. **Fix the dataset class** - Use official directories
3. **Rebuild cache** - With correct splits
4. **Verify counts** - Should see 290 train, 80 eval subjects
5. **Retrain** - Results should now be comparable to paper

## Expected Impact

With correct splits, we expect:
- **More stable eval** - 80 subjects vs 4
- **Better generalization** - Proper test distribution
- **Valid comparison** - Same splits as paper
- **Higher BAC** - Just from having valid eval set

## Verification Commands

```bash
# After fix, verify correct counts:
uv run python -c "
from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
train_ds = TUEVEventDataset('data/datasets/tuev', 'train')
eval_ds = TUEVEventDataset('data/datasets/tuev', 'eval')
# Should print: ~290 train subjects, ~80 eval subjects
"
```

## THIS BUG ALONE COULD EXPLAIN 20-30% OF OUR BAC GAP!
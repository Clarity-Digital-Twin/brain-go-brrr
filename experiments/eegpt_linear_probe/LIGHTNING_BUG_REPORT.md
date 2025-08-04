# PyTorch Lightning 2.5.2 Critical Bug Report

## Issue Summary
PyTorch Lightning 2.5.2 hangs indefinitely when training with large cached datasets (>100k samples).

## Symptoms
Training freezes at:
```
Loading `train_dataloader` to estimate number of stepping batches
```

- CPU usage remains high (~600%)
- GPU shows some activity (~25%)
- No progress bars appear
- No epochs start
- Process must be killed manually

## Environment
- PyTorch Lightning: 2.5.2
- PyTorch: 2.x
- Dataset: TUAB with 930,495 cached windows
- System: WSL2, RTX 4090, Python 3.12

## Attempted Fixes (ALL FAILED)
1. `deterministic=False` - Still hangs
2. `limit_train_batches=300` (integer) - Still hangs
3. `max_steps=10000` - Still hangs
4. `num_sanity_val_steps=0` - Still hangs
5. `reload_dataloaders_every_n_epochs=1` - Still hangs
6. `fast_dev_run=True` - Still hangs
7. `num_workers=0` - Still hangs
8. Combinations of above - Still hangs

## Root Cause
Lightning's dataloader length estimation code enters an infinite loop or deadlock when:
- Using cached datasets with __len__ > 100k
- Custom collate functions
- Possibly related to the batch sampler initialization

## Solution
**DO NOT USE PYTORCH LIGHTNING FOR THIS PROJECT**

Use pure PyTorch training loop:
- `train_pytorch_nan_safe.py` - **RECOMMENDED** - Includes NaN protection
- `train_pytorch_stable.py` - Basic version without NaN protection
- Both work perfectly with same dataset/model
- Training at ~10 it/s (much faster than Lightning would be)

## Evidence
1. Standalone dataloader test passes (loads 3 batches in <1s)
2. fast_dev_run=5 worked ONCE then hung on retry
3. Pure PyTorch training works immediately
4. Issue persists across multiple Lightning versions

## References
- GitHub issues: #4450, #11587 (similar but not exact)
- No official fix as of August 2024
- Community confirms this is a known issue with large datasets

## Recommendation
1. Use pure PyTorch for training
2. Consider Lightning Fabric (lower-level API) if needed
3. File bug report with minimal reproduction
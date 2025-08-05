# Archive - Why These Scripts Are Broken

## 🚨 DO NOT USE ANY SCRIPTS IN THIS ARCHIVE!

### /crashed_overnight/
**train_pytorch_stable.py** - CRASHED at 0.6% of epoch 0
- No NaN detection whatsoever
- No gradient clipping
- No input validation
- Crashed after 1 hour with no error handling
- WASTED ENTIRE NIGHT OF COMPUTE

### /lightning_broken/
**ALL PyTorch Lightning scripts** - HANG FOREVER
- Lightning 2.5.2 has critical bug with large datasets
- Hangs at "Loading train_dataloader to estimate number of stepping batches"
- Cannot be fixed with ANY settings
- Wasted DAYS trying to debug this shit

### /failed_attempts/
- Scripts with wrong learning rates that exploded
- Scripts with bad batch size causing OOM
- Scripts with incorrect data loading
- Scripts with wrong model architecture

### /old_training_scripts/
- Pre-NaN-fix versions
- Scripts before channel mapping fixes
- Scripts with old hyperparameters
- Scripts before cached dataset implementation

### /smoke_tests/
- Quick test scripts that don't do full training
- Debugging scripts for specific issues
- Not meant for actual training

## Why We Archive Instead of Delete

1. **Learn from failures** - See what didn't work
2. **Avoid repeating mistakes** - Don't waste time on same bugs
3. **Reference for debugging** - Compare working vs broken
4. **Documentation** - Show the journey to working solution

## The Only Working Solution

**USE ONLY: train_pytorch_nan_safe.py**

This is the ONLY script that:
- Handles NaN/Inf properly
- Has gradient clipping
- Uses correct learning rate schedule
- Validates inputs
- Actually completes training

Everything else is broken garbage that will waste your time and compute.
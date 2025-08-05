# EEGPT Scripts Status - WHAT WORKS AND WHAT'S BROKEN

## ✅ WORKING SCRIPTS (Keep These!)

### 1. train_pytorch_nan_safe.py ✅✅✅
**STATUS: CURRENTLY RUNNING SUCCESSFULLY**
- Full NaN protection with gradient clipping
- Learning rate warmup from 2e-6
- Input validation on every batch
- Anomaly detection for debugging
- Currently at 62.5% accuracy after 48% of epoch 0
- **USE THIS FOR ALL TRAINING**

### 2. launch_nan_safe_training.sh ✅
- Launches the nan_safe training in tmux
- Sets all environment variables correctly
- Creates proper log directories
- **USE THIS TO START TRAINING**

### 3. custom_collate_fixed.py ✅
- Handles variable channel counts properly
- Works with cached dataset
- No issues found

### 4. inference_example.py ✅
- Shows how to load trained model
- Example code for predictions
- Works correctly

## ❌ BROKEN/ARCHIVED SCRIPTS

### archive/crashed_overnight/
- **train_pytorch_stable.py** ❌ - CRASHED after 171 batches! No NaN protection
- **launch_training_slow.sh** ❌ - Launches the broken stable script

### archive/lightning_broken/
- **train_enhanced.py** ❌ - PyTorch Lightning HANGS FOREVER on large datasets
- **All Lightning-based scripts** ❌ - Lightning 2.5.2 has unfixable bug

### archive/failed_attempts/
- Various experimental scripts that didn't work
- Old training attempts with bugs

### archive/old_training_scripts/
- Outdated versions before NaN fixes
- Scripts with wrong hyperparameters

## 📁 Important Files to Keep

1. **README.md** - Main documentation
2. **TRAINING_STATUS.md** - Current training progress
3. **LIGHTNING_BUG_REPORT.md** - Why we can't use Lightning
4. **CHANNEL_MAPPING_EXPLAINED.md** - Critical channel naming info
5. **configs/** - All configuration files

## 🚨 NEVER USE THESE

1. **PyTorch Lightning** - Will hang on datasets >100k samples
2. **train_pytorch_stable.py** - No NaN protection, will crash
3. **Any script without gradient clipping** - Will explode
4. **Scripts with num_workers=0** - Causes persistent_workers bug

## 📝 Summary

**ONLY USE `train_pytorch_nan_safe.py` FOR TRAINING!**

Everything else is archived for reference only. The nan_safe version has been tested and is currently running successfully with all protections enabled.
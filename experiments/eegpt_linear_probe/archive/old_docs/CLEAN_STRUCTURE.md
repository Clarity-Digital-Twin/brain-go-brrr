# CLEAN PROJECT STRUCTURE - WHAT EVERYTHING DOES

## DIRECTORIES:

### `/data/cache/`
- `tuab_4s_final/` - **THE ONLY CACHE WE USE** (157 GB, 2 million .pt files)
  - Currently being converted to memory-mapped arrays
  - Contains 4-second windows at 256Hz 
  - 2,050,044 total windows (1.86M train, 185k eval)

### `/experiments/eegpt_linear_probe/`
Main experiment directory for EEGPT linear probe training

#### `configs/`
- `tuab_4s_paper_aligned.yaml` - **THE ONLY CONFIG** - Paper-aligned settings
  - batch_size: 32
  - num_workers: 0 (WSL limitation)
  - 4-second windows, 50% overlap
  - 50 epochs, early stopping

#### `logs/` 
- Empty (cleaned) - will contain training logs

#### `output/`
- Empty (cleaned) - will contain model outputs during training

#### `checkpoints/`
- Empty (cleaned) - will contain saved model checkpoints

#### `results/`
- Empty (cleaned) - will contain final results

## PYTHON FILES:

### Core Training Files:
- `train_paper_aligned.py` - **MAIN TRAINING SCRIPT**
  - Trains linear probe on frozen EEGPT features
  - Uses memory-mapped dataset for speed
  - Target AUROC: 0.869 (paper performance)

- `tuab_mmap_dataset.py` - **MEMORY-MAPPED DATASET**
  - Loads from .npy files (after conversion)
  - No RAM usage - OS handles paging
  - Expected speed: 50-100 it/s on WSL

- `custom_collate_fixed.py` - **DATALOADER COLLATE**
  - Ensures consistent 20 channels
  - Handles batch creation

### Build Scripts (One-time use):
- `build_4s_cache_FINAL.py` - Built the 157GB cache (DONE)
- `build_mmap_cache.py` - **CURRENTLY RUNNING** - Converting to memory-mapped arrays
  - Creates: train_data.npy (152.8 GB), train_labels.npy
  - Creates: eval_data.npy (15.1 GB), eval_labels.npy
  - Progress: ~13% after 10 minutes, ~2 hours total

### Documentation:
- `README.md` - Original readme
- `TRAINING_STATUS_FINAL.md` - Training status tracking
- `CLEAN_STRUCTURE.md` - THIS FILE - explains everything

## WHAT'S HAPPENING NOW:

1. **Memory-mapped conversion running** (tmux session: mmap_convert)
   - Converting 157GB of .pt files to 4 .npy files
   - ~2 hours total, currently at ~13%

2. **After conversion completes:**
   ```bash
   # Launch training
   export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
   export CUDA_VISIBLE_DEVICES=0
   
   python train_paper_aligned.py --config configs/tuab_4s_paper_aligned.yaml
   ```

3. **Expected performance:**
   - Training speed: 50-100 it/s (reasonable for WSL)
   - Total training time: ~8-10 hours
   - Target AUROC: 0.869 ± 0.005

## DELETED (CLEANED UP):

### Configs (8 old/experimental configs deleted):
- tuab_4s_final.yaml, tuab_8s_temp.yaml, tuab_cached.yaml, etc.

### Python files (3 failed experiments deleted):
- tuab_cached_loader.py, tuab_preloaded.py, inference_example.py

### Logs (40+ old logs deleted)

### Documentation (4 unnecessary files deleted):
- INDEX.md, PROFESSIONAL_PRACTICES.md, SETUP_COOKBOOK.md, launch_paper_aligned_training.sh

### Cache directories (3 broken caches being deleted in background):
- tuab_enhanced/, tuab_enhanced_test/, tuab_preprocessed/

## EVERYTHING IS NOW CLEAN AND DOCUMENTED!
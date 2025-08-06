# FINAL CACHE FIX - THE COMPLETE TRUTH

## THE FUCKING PROBLEM

1. **EEGPT needs 4-second windows** (was pretrained on 4s, NOT 8s)
2. **Current cache is BROKEN**: 
   - Empty (0 windows)
   - Wrong paths
   - Wrong channel matching (looking for exact names instead of handling "EEG FP1-REF" format)
3. **Training is loading raw EDFs every batch** = 124 seconds per iteration = DISASTER

## THE SOLUTION - BUILD PROPER 4S CACHE

### Step 1: Kill Everything
```bash
tmux kill-server
pkill -f train
```

### Step 2: Clean Up Broken Shit
```bash
rm -rf /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache_4s
rm -rf /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache/*
```

### Step 3: Build WORKING 4s Cache

The TUAB files have channels like "EEG FP1-REF", "EEG T3-REF" etc. We need to:
1. Strip the "EEG " prefix and "-REF" suffix
2. Map old names (T3→T7, T4→T8, T5→P7, T6→P8)
3. Cache 4-second windows at 256 Hz

### Step 4: Use EXISTING Working Infrastructure

We ALREADY have `TUABCachedDataset` that works! We just need to:
1. Point it to the right cache
2. Make sure window size is 4s
3. Use proper channel handling

## THE ACTUAL FIX

The paper-aligned training script SHOULD be using a pre-built cache with:
- 4-second windows
- 256 Hz sampling rate
- Proper channel mapping
- Fast loading (no EDF reads during training)

## VERIFICATION

A working training should show:
- **Speed**: 1-2 iterations/second (NOT 124 seconds/iteration)
- **No channel warnings** during training
- **AUROC target**: 0.869 (achievable with 4s windows)

## CRITICAL FILES

1. `/data/cache_4s/tuab_index_4s.json` - MUST exist and have entries
2. `configs/tuab_4s_paper_target.yaml` - MUST use 4s windows
3. `train_paper_aligned.py` - The working training script

## SUCCESS METRICS

- First epoch < 20 minutes (not 3 days!)
- AUROC > 0.85 by epoch 10
- Target AUROC 0.869 ± 0.005
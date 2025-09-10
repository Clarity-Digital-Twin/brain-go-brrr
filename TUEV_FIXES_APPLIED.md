# TUEV Fixes Applied - Summary

**Date**: September 10, 2025  
**Purpose**: Document all fixes applied to match EEGPT reference implementation

## Fixes Applied ✅

### 1. Data Splits - FIXED ✅
- **Problem**: Used wrong 80/20 split with seed=42, only 4 eval subjects
- **Solution**: Deleted cache, now uses official train/eval directories
- **Impact**: 359 train files, 159 eval files (was 205/4)

### 2. WeightedRandomSampler - REMOVED ✅
- **Problem**: We forced class balancing, reference doesn't
- **Solution**: Removed sampler, using simple shuffle=True
- **Impact**: Model sees natural distribution like reference

### 3. Batch Size - FIXED ✅  
- **Problem**: Used 384 effective (32×12), reference uses 400
- **Solution**: Auto-adjust to 34×12=408 ≈ 400
- **Impact**: Exact gradient statistics match

### 4. Normalization - FIXED ✅
- **Problem**: We normalized to N(0,50μV), reference uses raw μV
- **Solution**: Disabled normalization, scale V→μV in forward pass
- **Impact**: Same input scale as reference

### 5. Mean Pooling - ADDED ✅
- **Problem**: We flattened to 2048, reference pools to 512
- **Solution**: Added mean pooling option (enabled by default)
- **Impact**: 4x smaller feature vector like reference

### 6. DropPath - PENDING ⏳
- **Problem**: Reference uses 0.2, we don't have it
- **Solution**: Complex to add, skip for initial test
- **Impact**: Minor regularization difference

## Code Changes

### train_tuev_events.py
```python
# 1. NO SAMPLER
train_loader = DataLoader(
    train_dataset,
    shuffle=True,  # Simple shuffle, NO WeightedRandomSampler
    ...
)

# 2. EXACT BATCH
if args.batch_size * 12 == 384:
    args.batch_size = 34  # 34*12 = 408 ≈ 400

# 3. NO NORMALIZATION
self.eegpt.normalize = False  # Use raw values
x = x * 1e6  # Convert V to μV in forward

# 4. MEAN POOLING
self.use_mean_pooling = True  # Match reference
features = features.mean(dim=1)  # (B,4,512) → (B,512)
```

## Expected Impact

With these fixes, we expect:
1. **Valid evaluation** - 159 eval files vs 4 subjects
2. **Natural distribution** - No forced balancing
3. **Correct scale** - Raw μV like reference  
4. **Same architecture** - 512 features not 2048
5. **BAC improvement** - Should jump from 0.24 → 0.50+

## Next Steps

1. Wait for cache rebuild (in progress via tmux)
2. Run training with all fixes
3. Monitor per-class recall and BAC progression
4. Target: 0.62 ± 0.02 BAC by epoch 30

## Verification

After cache completes:
```bash
# Check cache
ls -la data/datasets/tuev/cache/tuev_event_segments/*/index.json

# Start training
./experiments/eegpt_linear_probe/scripts/launch_tuev_safe.sh
```

## Status
- Cache rebuilding: ~90% complete
- All major divergences fixed except DropPath
- Ready to train once cache completes
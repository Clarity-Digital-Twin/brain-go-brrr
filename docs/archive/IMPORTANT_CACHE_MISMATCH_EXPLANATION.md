# CRITICAL: Cache Mismatch Issue Explanation

## The Problem

We discovered why the baseline training was taking forever to load:

1. **Existing cached data**: 188,981 window files cached at **8 seconds @ 256Hz**
2. **Current config**: Expects **10.24 seconds @ 200Hz** (to match EEGPT paper)
3. **Result**: Complete cache miss - regenerating all 725,775 windows!

## Timeline

- Earlier: Ran training with 8s @ 256Hz windows, created 188,981 cached files
- Later: Updated config to match paper (10.24s @ 200Hz) 
- Now: Training stuck regenerating all windows because cache is incompatible

## Solutions

### Option 1: Use cached baseline config (FAST)
```bash
export EEGPT_CONFIG=configs/tuab_cached_baseline.yaml
```
- Uses existing 8s @ 256Hz cached data
- Will load in seconds, not hours
- Good for quick baseline comparison

### Option 2: Wait for regeneration (SLOW) 
- Continue with current training
- Will take ~2-3 hours to regenerate all windows
- Matches paper specifications exactly

### Option 3: Clear cache and restart
```bash
rm -rf data/cache/tuab_enhanced/*.pkl
# Then restart training
```

## Configs Explained

- **tuab_cached_baseline.yaml**: Matches cached data (8s @ 256Hz) - USE THIS FOR QUICK TESTS
- **tuab_enhanced_config.yaml**: Paper specs (10.24s @ 200Hz) - REGENERATES CACHE
- **DEPRECATED_tuab_cached_fast_BROKEN.yaml**: DO NOT USE - causes infinite loop

## Key Learning

The cache is tied to specific window parameters:
- Window duration
- Sampling rate  
- Preprocessing settings

Changing ANY of these invalidates the entire cache!
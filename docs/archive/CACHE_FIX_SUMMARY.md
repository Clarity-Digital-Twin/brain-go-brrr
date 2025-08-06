# Cache Fix Summary

## What We Fixed

1. **Standardized Window Specs**: All configs now use **8s @ 256Hz**
   - Was mixed: some 8s @ 256Hz, others 10.24s @ 200Hz
   - This caused cache invalidation and regeneration

2. **Added Read-Only Cache Mode**:
   - New `cache_mode` parameter: "write" (default) or "readonly"
   - In readonly mode, raises error if cache miss (prevents regeneration)
   - Updated datasets to support this

3. **Created Cache Building Script**: `scripts/build_tuab_cache.py`
   - Pre-generates all windows (930,495 for train, ~46k for eval)
   - Shows progress and uses parallel processing
   - Currently running in tmux session 'cache_build'

4. **Fixed Test Suite Issues**:
   - Added AutoReject cache isolation via env var
   - Updated pytest.ini to skip slow/GPU tests by default
   - Fixed scope issues with fixtures

5. **Cleaned Up Configs**:
   - Renamed broken config to `DEPRECATED_tuab_cached_fast_BROKEN.yaml`
   - Created `tuab_cached_baseline.yaml` for baseline training
   - Updated `tuab_enhanced_config.yaml` to standard specs

## Current Status

Cache building is running:
- Progress: Check with `tmux attach -t cache_build`
- Log: `tail -f logs/cache_build_final.log`
- ETA: ~1-2 hours based on dataset size

## Next Steps

Once cache is built:

1. **Update config to readonly**:
   ```yaml
   cache_mode: "readonly"
   ```

2. **Run baseline training**:
   ```bash
   EEGPT_CONFIG=configs/tuab_cached_baseline.yaml \
   python experiments/eegpt_linear_probe/train_enhanced.py
   ```

3. **Run with AutoReject**:
   ```yaml
   use_autoreject: true
   ```

4. **Compare AUROC** (target: 5% improvement)

## Key Learning

- Cache is tied to exact window parameters
- Partial caches cause confusion (looks stuck but is regenerating)
- Read-only mode prevents accidental regeneration
- Pre-building cache saves hours of confusion
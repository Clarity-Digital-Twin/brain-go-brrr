# Fix Summary - August 28, 2025

## ✅ Issues Fixed in src/

### Architecture Unification
- **Problem**: Had parallel implementations in experiments/ and src/
- **Solution**: experiments/ now imports from src/ (thin shims only)
- **Result**: Single source of truth for all components

### Normalization SSOT
- **Problem**: Multiple normalization points causing inconsistency
- **Solution**: Wrapper handles ALL normalization, datasets emit raw mV
- **Result**: Consistent N(0,1) data to model

### Channel Validation
- **Problem**: TUAB validated against subset instead of SSOT
- **Solution**: Direct comparison with ordered channels from SSOT
- **Result**: Enforces correct channel order

### META Schema
- **Problem**: Inconsistent keys (channels19 vs channels20)
- **Solution**: Unified to "channels" + "n_channels", backward compat with warnings
- **Result**: Consistent metadata across all datasets

## Quality Gates (All Green)
- ✅ Typecheck: 122 source files clean
- ✅ Lint: All checks passed
- ✅ Unit tests: 751 passed
- ✅ Smoke tests: 16 passed
- ✅ No Lightning imports
- ✅ No sys.path.insert in src/

## Remaining Work (experiments/)
- Remove sys.path.insert from training scripts
- Continue using src/ components everywhere

## Files Changed
- `src/brain_go_brrr/infra/data/tuab_dataset.py`: Fixed channel validation
- `src/brain_go_brrr/infra/data/tuev_dataset.py`: META schema unified
- `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py`: Normalization SSOT
- Deprecated dataset files converted to aliases with warnings

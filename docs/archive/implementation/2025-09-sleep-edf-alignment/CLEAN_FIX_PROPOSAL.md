# CLEAN FIX PROPOSAL - Cache Path Standardization

> ⚠️ **ARCHIVED DOCUMENT - DO NOT USE FOR CURRENT DEVELOPMENT**
> This document is preserved for historical reference only.
> For current documentation, see [docs/README.md](../../README.md)
> Archive date: September 2, 2025

---



## CURRENT CLUSTERFUCK STATUS

1. **Cache is building RIGHT NOW to wrong path**: `tuab_mne_v2` (24+ minutes in)
2. **Training scripts expect**: `tuab_mne_preprocessed`
3. **Two different cache builders exist**: One good (cache_builder.py), one slop (build_tuev_mne_cache.sh creates Python inline)

## THE CLEANEST SOLUTION

### STANDARDIZE ON ONE NAMING CONVENTION

**Proposal: Use descriptive, versioned names everywhere**

```
/data/cache/
├── tuab_mne_v2/      # TUAB with MNE preprocessing, version 2
└── tuev_mne_v2/      # TUEV with MNE preprocessing, version 2
```

### WHY THIS IS CLEANEST

1. **Version suffix `_v2` is GOOD** - Shows this is the new deterministic version
2. **Clear what each cache contains** - tuab/tuev, mne preprocessing
3. **No legacy baggage** - Not tied to old paths that no longer exist
4. **Future-proof** - Can have v3, v4 as preprocessing improves

## IMPLEMENTATION PLAN

### Step 1: Let current cache finish (it's 24+ min in)
- Don't waste compute
- It's building to `tuab_mne_v2` which is CORRECT

### Step 2: Update all consumers to use v2 names
```bash
# Update training launcher scripts
sed -i 's/tuab_mne_preprocessed/tuab_mne_v2/g' scripts/launch_tuab_mne.sh
sed -i 's/tuev_mne_preprocessed/tuev_mne_v2/g' scripts/launch_tuev_mne.sh

# Update any config files
find configs -name "*.yaml" -exec sed -i 's/tuab_mne_preprocessed/tuab_mne_v2/g' {} \;
find configs -name "*.yaml" -exec sed -i 's/tuev_mne_preprocessed/tuev_mne_v2/g' {} \;
```

### Step 3: Clean up redundant scripts
```bash
# Delete the slop script that creates Python inline
rm scripts/build_tuev_mne_cache.sh

# Keep only these cache-related scripts:
# - scripts/build_mne_cache.sh (update to call cache_builder.py with v2 paths)
# - scripts/launch_tuev_cache.sh (already uses v2)
# - mne_integration/cache_builder.py (the ONE TRUE cache builder)
```

### Step 4: Update build_mne_cache.sh to be consistent
```bash
# Should call cache_builder.py with:
# --cache-dir /data/cache/tuab_mne_v2 (not preprocessed)
# --data-root /data/datasets/tuab/edf (not external/)
```

## WHAT WE'RE DELETING (SLOP REMOVAL)

1. `scripts/build_tuev_mne_cache.sh` - Creates Python inline (SLOP!)
2. `datasets/tuab_mne_dataset.py` - Deprecated shim
3. `datasets/tuev_mne_dataset.py` - Deprecated shim

## WHAT WE'RE KEEPING (CLEAN)

1. `mne_integration/cache_builder.py` - ONE cache builder to rule them all
2. `train_tuab_mne.py` - Training script (already imports from src/)
3. `train_tuev_mne.py` - Training script (already imports from src/)
4. Updated launch scripts pointing to v2 caches

## SENIOR REVIEW QUESTIONS

1. **Should we use `_v2` suffix or go back to `_preprocessed`?**
   - Recommendation: Keep `_v2` - it's cleaner and version-explicit

2. **Should we wait for current cache to finish (could be hours)?**
   - Recommendation: Yes, it's already 24+ min in, don't waste it

3. **Should we update training scripts or just launch scripts?**
   - Recommendation: Just launch scripts - training scripts take cache_dir as argument

## COMMANDS TO EXECUTE (AFTER APPROVAL)

```bash
# 1. Update launch scripts to use v2
sed -i 's/tuab_mne_preprocessed/tuab_mne_v2/g' experiments/eegpt_linear_probe/scripts/launch_tuab_mne.sh
sed -i 's/tuev_mne_preprocessed/tuev_mne_v2/g' experiments/eegpt_linear_probe/scripts/launch_tuev_mne.sh

# 2. Delete slop
rm experiments/eegpt_linear_probe/scripts/build_tuev_mne_cache.sh
rm experiments/eegpt_linear_probe/datasets/tuab_mne_dataset.py
rm experiments/eegpt_linear_probe/datasets/tuev_mne_dataset.py
rmdir experiments/eegpt_linear_probe/datasets  # if empty

# 3. Update build_mne_cache.sh
cat > experiments/eegpt_linear_probe/scripts/build_mne_cache.sh << 'EOF'
#!/bin/bash
# Clean TUAB cache builder - calls the ONE TRUE cache_builder.py

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
DATA_ROOT="${BGB_DATA_ROOT:-$PROJECT_ROOT/data}"

echo "Building TUAB MNE cache v2..."
uv run python experiments/eegpt_linear_probe/mne_integration/cache_builder.py \
    --corpus TUAB \
    --data-root "$DATA_ROOT/datasets/tuab/edf" \
    --cache-dir "$DATA_ROOT/cache/tuab_mne_v2" \
    --split both
EOF
```

## END STATE

- ONE cache builder (cache_builder.py)
- CONSISTENT v2 naming everywhere
- NO symlinks
- NO inline Python generation
- NO deprecated shims
- CLEAN, maintainable structure

## WAITING FOR SENIOR APPROVAL

Please review and approve before we execute. This is the cleanest path forward.

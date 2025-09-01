# CACHE PATH ANALYSIS - CRITICAL ISSUE

## THE PROBLEM

We have a PATH MISMATCH CLUSTERFUCK caused by data deletion and re-download:

### Original Setup (Before Data Deletion)
- TUAB was at: `/data/datasets/external/tuab/`
- TUEV was at: `/data/datasets/external/tuh_eeg/TUEV/v2.0.1/`
- Cache dirs: `tuab_mne_preprocessed`, `tuev_mne_preprocessed`

### Current Setup (After Re-download)
- TUAB is at: `/data/datasets/tuab/` (DIFFERENT!)
- TUEV is at: `/data/datasets/tuev/` (DIFFERENT!)
- New cache dirs: `tuab_mne_v2`, `tuev_mne_v2` (DIFFERENT!)

## FILE INVENTORY

### Training Scripts (GOOD - Import from src/)
```
train_tuab_mne.py - EXPECTS cache at: tuab_mne_preprocessed
train_tuev_mne.py - EXPECTS cache at: tuev_mne_preprocessed  
```

### Cache Builders (MIXED STATE)
```
mne_integration/cache_builder.py - NEW, deterministic, creates *_v2 caches
scripts/build_mne_cache.sh - OLD, calls cache_builder.py, but for OLD paths
scripts/build_tuev_mne_cache.sh - SLOP, creates Python inline, wrong paths
scripts/launch_tuev_cache.sh - NEW (I just created), uses cache_builder.py
```

### Launch Scripts (BROKEN - Wrong cache paths)
```
scripts/launch_tuab_mne.sh - EXPECTS: tuab_mne_preprocessed (won't find v2)
scripts/launch_tuev_mne.sh - EXPECTS: tuev_mne_preprocessed (won't find v2)
```

### Deprecated Shims (UNUSED but sitting there)
```
datasets/tuab_mne_dataset.py - Just imports from src/
datasets/tuev_mne_dataset.py - Just imports from src/
```

## DEPENDENCIES MAP

```
launch_tuab_mne.sh 
  ├── Checks for: cache/tuab_mne_preprocessed/
  ├── References: build_mne_cache.sh (in error message)
  └── Runs: train_tuab_mne.py

launch_tuev_mne.sh
  ├── Checks for: cache/tuev_mne_preprocessed/
  ├── References: build_tuev_mne_cache.sh (in error message)
  └── Runs: train_tuev_mne.py

build_mne_cache.sh
  └── Runs: cache_builder.py --cache-dir tuab_mne_preprocessed

build_tuev_mne_cache.sh
  └── Creates & runs: tuev_cache_builder.py (INLINE SLOP!)
```

## ROOT CAUSE

The refactoring from parallel experiments/ universe to unified src/ happened BEFORE the data deletion. When we re-downloaded data to DIFFERENT paths, the scripts weren't updated to match.

## THE FIX PLAN

### Option 1: Use OLD cache names (Least disruption)
1. Change cache_builder.py to remove "_v2" suffix
2. Update build_mne_cache.sh to use `/data/datasets/tuab/edf`
3. Delete build_tuev_mne_cache.sh (redundant slop)
4. Update launch_tuev_cache.sh to use `tuev_mne_preprocessed` not `_v2`

### Option 2: Update everything to NEW names (Cleaner long-term)
1. Update launch_tuab_mne.sh to expect `tuab_mne_v2`
2. Update launch_tuev_mne.sh to expect `tuev_mne_v2`
3. Update train_*.py configs to use new cache paths
4. Delete build_tuev_mne_cache.sh (redundant slop)

### Option 3: Symlink workaround (Quick fix)
1. After cache builds, create symlinks:
   - `tuab_mne_preprocessed -> tuab_mne_v2`
   - `tuev_mne_preprocessed -> tuev_mne_v2`

## RECOMMENDATION

**GO WITH OPTION 1** - Use old cache names to minimize changes:

1. The training scripts are proven to work
2. Less risk of breaking something
3. Cache names don't really matter as long as they're consistent
4. We're already building TUAB cache - don't want to restart

## FILES TO CHANGE

```bash
# 1. Update cache_builder.py
sed -i 's/tuab_mne_v2/tuab_mne_preprocessed/g' mne_integration/cache_builder.py
sed -i 's/tuev_mne_v2/tuev_mne_preprocessed/g' mne_integration/cache_builder.py

# 2. Update build_mne_cache.sh data path
sed -i 's|external/tuab|tuab/edf|' scripts/build_mne_cache.sh

# 3. Delete redundant slop
rm scripts/build_tuev_mne_cache.sh

# 4. Update launch_tuev_cache.sh
sed -i 's/tuev_mne_v2/tuev_mne_preprocessed/' scripts/launch_tuev_cache.sh
```

## FILES TO KEEP AS-IS

- train_tuab_mne.py - Already correct
- train_tuev_mne.py - Already correct
- launch_tuab_mne.sh - Will work once cache name matches
- launch_tuev_mne.sh - Will work once cache name matches

## SENIOR REVIEW NEEDED

Please review this analysis before we make changes. Key questions:

1. Is Option 1 (use old cache names) the right approach?
2. Should we delete the deprecated shims in datasets/?
3. Any other dependencies we missed?

## CURRENT STATUS

- TUAB cache building in tmux as `tuab_mne_v2` (WRONG NAME!)
- Need to either:
  - Stop it and rebuild with correct name
  - OR let it finish and symlink
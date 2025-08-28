# CRITICAL TECHNICAL DEBT - Channel Enforcement

## ⚠️ CRITICAL CHANNEL SPECIFICATIONS
- **TUAB**: EXACTLY **19 channels** (excludes Fz which is often missing in raw data)
- **TUEV**: EXACTLY **20 channels** (includes Fz, excludes Fpz per EEGPT paper Table 13)

**DO NOT CONFUSE THESE** - They have different requirements!

## 🔴 CURRENT ISSUE: TUAB 20-Channel Cache Contamination

### Problem Summary
- **304 windows (0.081%)** have 20 channels instead of 19
- Source files: `aaaaakfo_s004_t000.edf` and `aaaaakfo_s005_t000.edf` (both abnormal)
- Current workaround: Collate function truncates 20→19 by dropping Fz (index 4)
- Risk: Assumes Fz is always at index 4, which may not be guaranteed

### Root Cause
The MNE preprocessor inconsistently handled channel selection for certain files:
- Most files: Correctly dropped Fz to get 19 channels
- 2 files: Kept all 20 channels including Fz
- Likely cause: These files may have had different initial channel configurations

### Current Workaround (TEMPORARY)
```python
# In utils/custom_collate_fixed.py
if x.shape[0] == 20:
    x = torch.cat([x[:4], x[5:]], dim=0)  # Drop channel 4 (Fz)
```

## 🟡 SHORT-TERM FIXES (After Current Training Run)

### Option 1: Clean Existing Cache (Quick)
```bash
# Remove the 304 bad windows from the index
uv run python scripts/clean_20ch_windows.py
```
- Pros: Fast, no rebuild needed
- Cons: Loses 304 training samples (negligible impact)

### Option 2: Fix and Re-save Bad Windows
```python
# Load the 304 bad windows, drop Fz, re-save
for bad_window in bad_windows:
    data = torch.load(bad_window)
    data['x'] = data['x'][[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
    torch.save(data, bad_window)
```
- Pros: Preserves all data
- Cons: Assumes channel order is consistent

## 🟢 LONG-TERM FIXES (Next Cache Build)

### Proper Solution: Enforce Channel Selection in Preprocessor
```python
# In preprocessors/mne_preprocessor.py
CANONICAL_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8',  # No Fz!
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]  # Total: 19 channels

# Ensure consistent ordering and selection
raw_clean = raw_clean.pick_channels(CANONICAL_CHANNELS, ordered=True)
```

### Cache Index Improvements
1. Store channel names in index: `"channels": ["Fp1", "Fp2", ...]`
2. Validate on load: Assert channels match expected
3. Version the cache: `"cache_version": "mne-ar-v3"`

## 📊 Impact Assessment

### Current Impact
- **Frequency**: ~1 in 1,230 batches will fail (0.081%)
- **Training**: Handled by collate workaround, no crashes
- **Performance**: Negligible overhead from channel check

### If Left Unfixed
- Random batch failures without collate workaround
- Potential inconsistency in channel ordering
- Hidden technical debt accumulation

## ✅ RESIDUAL ISSUES (Non-Critical)

### 1. Normalization Warning
```
WARNING - No normalization file found - using identity normalization
```
- Impact: None (MNE preprocessing already normalizes)
- Fix: Generate normalization stats or suppress warning

### 2. Config Cruft
- Lightning-era parameters still in config (unused)
- `early_stopping`, `gradient_clip_val` have no effect
- Fix: Clean up config file

### 3. Missing Tests
- No test for channel consistency in cache
- No test for collate function edge cases
- Fix: Add unit tests

## 🚀 Action Items

### While Training Runs (NOW)
✅ Keep collate workaround in place
✅ Monitor for any other channel count variations
✅ Document in this file

### After This Run (TOMORROW)
- [ ] Run script to identify ALL 20-channel windows
- [ ] Either remove from index OR fix and re-save
- [ ] Update collate to strict mode
- [ ] Add channel validation test

### Next Cache Build (NEXT WEEK)
- [ ] Fix preprocessor to enforce 19 channels
- [ ] Add channel names to cache index
- [ ] Version the cache format
- [ ] Run full validation before training

## 📈 Training Progress
- Started: 2025-08-27 15:08
- Status: Running with workaround
- Speed: ~1.8 batches/sec
- Target: 75-87% AUROC (paper benchmark)

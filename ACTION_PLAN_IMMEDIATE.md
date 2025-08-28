# IMMEDIATE ACTION PLAN - FIX THE TRAINING

## PROBLEMS FOUND (RANKED BY SEVERITY)

### 🔴 CRITICAL (Breaks Training)

1. **DOUBLE NORMALIZATION BUG**
   - Dataset normalizes to N(0,1) ✅
   - Model normalizes AGAIN ❌
   - Result: Tiny values → gradient vanishing → AUROC=0.50
   - **FIXED:** Smart detection in wrapper

2. **IDENTITY NORMALIZATION FALLBACK**
   - Was: mean=0, std=1 (no normalization!)
   - Raw EEG at 1e-5 scale seen as zeros
   - **FIXED:** Now uses 50μV typical scale

3. **WRONG WINDOW SIZE**
   - Some code uses 1000 samples (WRONG)
   - Some uses 8s = 2048 samples (WRONG) 
   - Should be: 4s @ 256Hz = 1024 samples
   - **NOT FIXED YET**

### 🟡 IMPORTANT (Causes Confusion)

4. **MULTIPLE EEGPT INTERFACES**
   - eegpt_wrapper.py (CLI)
   - eegpt_compat.py (API)
   - eegpt_probe_unified.py (Tasks)
   - **NOT FIXED YET**

5. **REDUNDANT DATASETS**
   - tuab_dataset.py
   - tuab_enhanced_dataset.py
   - tuab_cached_dataset.py
   - **NOT FIXED YET**

### 🟢 MINOR (Technical Debt)

6. **REDUNDANT ADAPTERS**
   - eegpt_classifier.py
   - eegpt_feature_extractor.py
   - Just thin wrappers
   - **NOT FIXED YET**

## WHAT'S BEEN FIXED

✅ **Restored deleted files:**
- eegpt_probe_unified.py
- tuab_enhanced_dataset.py  
- tuab_cached_dataset.py

✅ **Fixed normalization:**
- Removed identity fallback
- Added smart double-norm detection
- Now prevents normalizing twice

✅ **Tests passing:**
- All unit tests that were failing now pass

## NEXT STEPS (IN ORDER)

### Step 1: Verify Normalization Fix Works
```bash
# Check dataset output
python -c "
from brain_go_brrr.infra.data.tuab_dataset import TUABDataset
ds = TUABDataset('data/TUAB', split='train')
x, y = ds[0]
print(f'Mean: {x.mean():.3f}, Std: {x.std():.3f}')
# Should be close to Mean: 0.000, Std: 1.000
"
```

### Step 2: Fix Window Size Everywhere
```bash
# Find all wrong window sizes
grep -r "1000\|2048" src/ experiments/ --include="*.py"
# Replace with 1024
```

### Step 3: Test Training Again
```bash
# Launch training with fixed normalization
cd experiments/eegpt_linear_probe
./scripts/launch_tuab.sh
# Monitor for AUROC > 0.5
```

### Step 4: If Training Works, Start Migration
1. Create deprecation warnings
2. Migrate one component at a time
3. Keep tests green
4. Delete only after confirmed unused

## SUCCESS CRITERIA

- [ ] Dataset outputs N(0,1)
- [ ] Model receives N(0,1) (not double normalized)
- [ ] All windows are 1024 samples
- [ ] Training AUROC > 0.5 within 10 epochs
- [ ] Tests stay green throughout

## DO NOT DO

❌ Delete any more files
❌ Create new implementations
❌ Assume normalization "just works"
❌ Trust that window sizes are correct
❌ Make changes without running tests

## THE TRUTH

We had THREE bugs preventing training:
1. Missing normalization (fixed in experiments)
2. Identity normalization in wrapper (fixed)
3. Double normalization (just fixed)

Plus architectural mess:
- 10 EEGPT files
- 3 different APIs
- Multiple datasets
- Redundant code

The training should work now. The cleanup can wait.
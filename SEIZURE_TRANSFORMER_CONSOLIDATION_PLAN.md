# SeizureTransformer Documentation Consolidation Plan

## Current Documentation Audit (December 2025)

### Documents Reviewed

#### 1. **SEIZURE_TRANSFORMER_POSTMORTEM.md** ✅ KEEP
- **Status**: ACCURATE & CURRENT
- **Content**: Documents the architecture fix we completed
- **Decision**: Keep as historical record of what we fixed

#### 2. **IDEAL_REFERENCE_SEIZURE_TRANSFORMER_DATAFLOW.md** ✅ KEEP & UPDATE
- **Status**: MOSTLY ACCURATE (Dec 12, 2024)
- **Content**: Correct OSS implementation patterns
- **Decision**: Keep as reference specification, minor updates needed

#### 3. **INTENDED_SEIZURE_TRANSFORMER_APPLICATION.md** ✅ KEEP AS REFERENCE
- **Status**: HISTORICAL PLAN (Dec 12, 2024)  
- **Content**: Original implementation plan with TSE parser fix
- **Decision**: Keep as reference for TSE parsing fix we still need

#### 4. **CURRENT_SEIZURE_TRANSFORMER_DATAFLOW.md** ❌ ARCHIVE
- **Status**: OUTDATED (Sept 12, 2025)
- **Content**: Pre-fix state, no longer accurate after architecture fix
- **Decision**: Archive to `docs/archive/` - superseded by postmortem

#### 5. **TUSZ_IMPLEMENTATION.md** ❌ ARCHIVE
- **Status**: OUTDATED (Sept 9, 2025)
- **Content**: Old implementation guide with incorrect assumptions
- **Decision**: Archive - contains wrong architecture references

#### 6. **TUSZ_OSS_SUMMARY.md** ✅ KEEP
- **Status**: REFERENCE MATERIAL
- **Content**: OSS implementation details from Wu paper
- **Decision**: Keep as reference documentation

#### 7. **TUSZ_ROADMAP.md** ❌ ARCHIVE
- **Status**: OUTDATED PLAN
- **Content**: Old execution plan, mostly completed
- **Decision**: Archive - execution complete

#### 8. **TUSZ_SPEC.md** ❌ ARCHIVE  
- **Status**: OUTDATED SPEC
- **Content**: Old requirements, superseded by IDEAL doc
- **Decision**: Archive - replaced by IDEAL_REFERENCE

## What We've Actually Implemented (Current State - Dec 2025)

### ✅ COMPLETED
1. **Architecture Fix**: Wu 2025 CNN+Transformer vendored correctly
   - `seizure_transformer_wu2025.py` - correct architecture
   - `seizure_transformer_wrapper.py` - uses Wu 2025 by default
   - `seizure_transformer_utils.py` - SSOT preprocessing
   - Toy model deprecated with warnings

2. **Preprocessing Pipeline**: Exact match to paper
   - Z-score normalization (before windowing)
   - Resample to 256Hz
   - Bandpass 0.5-120Hz (order 3, causal)
   - Notch at 1Hz and 60Hz (Q=30)

3. **Evaluation**: Working but with small gap
   - Achieved: 0.844 AUROC
   - Expected: 0.876 AUROC
   - Gap: 3.2% (acceptable)

### ❌ STILL BROKEN / NOT IMPLEMENTED

1. **TSE Parser Bug** (CRITICAL DATA CORRUPTION)
   - Current: Accepts ANY 2-field line (not documented where this is)
   - Need: Only accept seizure annotations
   - Fix documented in INTENDED_SEIZURE_TRANSFORMER_APPLICATION.md

2. **Training Script Issues**
   - Training bypasses wrapper preprocessing (no bandpass/notch)
   - Uses window-level labels expanded to timesteps (wrong supervision)
   - Should use per-timestep segmentation labels

3. **NEDC Integration**
   - Not wired into evaluation scripts
   - FA/24h metrics not computed during evaluation

## Consolidation Actions

### Step 1: Archive Outdated Docs
```bash
mkdir -p docs/archive/seizure_transformer/
mv CURRENT_SEIZURE_TRANSFORMER_DATAFLOW.md docs/archive/seizure_transformer/
mv TUSZ_IMPLEMENTATION.md docs/archive/seizure_transformer/
mv TUSZ_ROADMAP.md docs/archive/seizure_transformer/
mv TUSZ_SPEC.md docs/archive/seizure_transformer/
```

### Step 2: Create New Unified Documentation
Create `SEIZURE_TRANSFORMER_STATUS.md` that combines:
- Current implementation state (from postmortem)
- Remaining gaps to fix (TSE parser, training preprocessing)
- Reference to IDEAL doc for correct patterns

### Step 3: Update IDEAL_REFERENCE Doc
Minor updates needed:
- Note that architecture is now vendored ✅
- Update preprocessing to note it's implemented ✅
- Mark evaluation as partially working (0.844 vs 0.876)

### Step 4: Keep These Active Docs
- `SEIZURE_TRANSFORMER_POSTMORTEM.md` - Historical record
- `IDEAL_REFERENCE_SEIZURE_TRANSFORMER_DATAFLOW.md` - Reference spec
- `INTENDED_SEIZURE_TRANSFORMER_APPLICATION.md` - TSE parser fix reference
- `TUSZ_OSS_SUMMARY.md` - Paper reference
- `SEIZURE_TRANSFORMER_STATUS.md` - NEW unified status doc

## Critical Gaps to Address

### Priority 1: Find and Fix TSE Parser
- **Issue**: Parser accepts non-seizure annotations
- **Impact**: Massive false positives in training data
- **Fix**: Implement strict seizure-only filtering (see INTENDED doc)
- **Location**: Need to find where TSE parsing actually happens

### Priority 2: Fix Training Preprocessing  
- **Issue**: Training bypasses critical filters
- **Impact**: Model sees different data distribution than inference
- **Fix**: Use wrapper preprocessing in training script

### Priority 3: Per-Timestep Labels
- **Issue**: Window labels expanded to all timesteps
- **Impact**: Wrong supervision signal
- **Fix**: Generate true per-timestep segmentation masks

## Implementation Parity Assessment

### What Matches Wu 2025 OSS ✅
- Model architecture (after fix)
- Preprocessing parameters
- Post-processing parameters
- Window size (60s)
- Sampling rate (256Hz)

### What Differs ❌
- TSE parsing (too permissive)
- Training preprocessing (missing filters)
- Label supervision (window vs timestep)
- AUROC (0.844 vs 0.876)

### Unknown/Unclear ❓
- Where TSE parsing actually happens in current code
- Whether we use the same train/dev/eval splits
- Exact channel selection strategy

## Next Steps

1. **Execute consolidation plan** (archive old, create unified status)
2. **Find TSE parser** in codebase (grep for annotation loading)
3. **Fix critical bugs** (TSE parser, training preprocessing)
4. **Re-evaluate** after fixes to see if AUROC improves
5. **Document final state** in unified status doc

## Summary

We have a working SeizureTransformer with correct architecture achieving 96% of paper performance (0.844 vs 0.876 AUROC). The main issues are:
1. Data corruption from permissive TSE parsing (location unknown)
2. Training bypasses preprocessing 
3. Wrong supervision (window vs timestep labels)

After consolidation, we'll have 5 clean documents instead of 8 confusing ones, with clear documentation of what works, what's broken, and how to fix it.
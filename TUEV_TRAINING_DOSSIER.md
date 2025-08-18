# TUEV Training Dossier - Critical Audit Report
**Date**: 2025-08-18  
**Author**: System Audit  
**Status**: ⚠️ CONCERNING - Requires Strategic Decision

## Executive Summary

The TUEV training is running but with MAJOR architectural concerns. We're using 163,840 features (full patch embeddings) vs the 2,048 features (summary tokens) that TUAB successfully used. This 80x increase in features is causing:
- Massive initial loss (~50 vs expected ~2)
- Slower convergence 
- Potential overfitting risk
- Architectural inconsistency between tasks

## Current Training Status

### Location & Commands
```bash
# Correct directory
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/experiments/eegpt_linear_probe

# Monitor training
tmux attach -t tuev_fixed_42

# Watch logs
tail -f logs/tuev_FIXED_seed42.log
```

### Metrics (Epoch 1, ~50% complete)
- Loss: Decreasing from 50 → 5-10 range (still HIGH)
- Accuracy: Fluctuating wildly (0.05 → 0.65)
- Speed: ~1.8s/batch (168 batches total)
- No validation results yet

## The Core Problem: Feature Extraction Strategy

### What We Have Now

```
TUAB (Working):
Input (23×1024) → Conv layers → EEGPT → 4 summary tokens (2048 features) → 2 classes
Result: BAcc ~0.85 ✅

TUEV (Current Attempt):
Input (23×1024) → Conv layers → EEGPT → ALL patches (163,840 features) → 6 classes
Result: Loss exploding, unstable training ❌
```

### Why This Happened

1. **Initial TUEV training failed** with summary tokens (BAcc 0.16 vs target 0.62)
2. **Hypothesis**: TUEV's 6-class problem needs more features than binary TUAB
3. **Emergency fix**: Extract ALL patch features instead of just summary
4. **Current file**: `eegpt_full_features.py` in experiments (SHOULD BE IN SRC!)

## Architectural Analysis

### EEGPT's Design Intent

EEGPT was designed to produce **summary tokens** for downstream tasks:
- 4 summary tokens capture global patterns
- Patches (16×20) contain local features
- Paper likely uses summary tokens for ALL tasks

### Why TUAB Works with Summary Tokens

- Binary classification (normal/abnormal) is simpler
- Global patterns sufficient for pathology detection
- 2048 features adequate for linear separation

### Why TUEV Might Need Different Approach

- 6-class event detection requires fine-grained temporal info
- Events (SPSW, GPED, PLED) have specific temporal signatures
- Summary tokens might lose critical local patterns

## The Architecture Dilemma

### Option 1: Keep Full Features (Current)
**Location**: `experiments/eegpt_linear_probe/eegpt_full_features.py`

**Pros**:
- Maximum information preservation
- Could capture fine temporal events

**Cons**:
- 80x parameter increase (163,840 vs 2,048)
- High overfitting risk
- Slow convergence
- Not how EEGPT was designed to be used

### Option 2: Fix Summary Token Usage
**Would require**: Proper feature extraction in `src/brain_go_brrr/models/`

**Pros**:
- Consistent with EEGPT design
- Matches TUAB success
- Fewer parameters

**Cons**:
- Already failed (BAcc 0.16)
- May genuinely lack features for 6-class

### Option 3: Hybrid Approach (Recommended)
**Proposal**: Use patch features but with pooling/attention

```python
# In src/brain_go_brrr/models/eegpt_features.py
class EEGPTFeatureExtractor:
    def get_summary_features(self):  # Original 4×512
    def get_patch_features(self):    # Full 16×20×512  
    def get_pooled_features(self):   # Pooled patches 20×512
    def get_attention_features(self): # Attention-weighted
```

## File Organization Issues

### Current (MESSY):
```
experiments/eegpt_linear_probe/
  ├── eegpt_full_features.py        # Should be in src!
  ├── train_tuev_aligned.py         # Uses wrong features
  └── train_tuev_aligned_fixed.py   # Emergency fix
```

### Should Be:
```
src/brain_go_brrr/models/
  ├── eegpt_wrapper.py              # Base wrapper
  ├── eegpt_feature_extractor.py   # Feature extraction strategies
  └── eegpt_task_heads.py          # Task-specific heads

experiments/eegpt_linear_probe/
  └── train_tuev.py                # Clean training script
```

## Critical Questions

1. **Why didn't the paper mention this?**
   - They might use different feature extraction per task
   - TUEV might have custom architecture not disclosed
   - We might be missing normalization/preprocessing

2. **Is 163,840 features sustainable?**
   - NO - will overfit with 83k training samples
   - Need dimensionality reduction or regularization

3. **Should we continue current training?**
   - Monitor for 5 epochs
   - If loss doesn't drop below 2.0, STOP
   - If validation BAcc < 0.4 by epoch 10, STOP

## Recommendations

### Immediate (Today):
1. **Let current training run** to epoch 5 for data
2. **Check validation metrics** - if BAcc < 0.35, abort
3. **DO NOT launch seeds 123, 456** until architecture resolved

### Short-term (This Week):
1. **Refactor feature extraction** to `src/brain_go_brrr/models/`
2. **Try pooled features** (20×512 instead of 163,840)
3. **Add L2 regularization** if keeping full features

### Long-term (Architecture):
1. **Create proper feature extraction API** in src
2. **Benchmark different strategies** systematically
3. **Document which tasks need which features**

## Risk Assessment

### Current Risks:
- ⚠️ **Overfitting**: 163k parameters for 83k samples
- ⚠️ **Memory**: Could OOM on larger batches
- ⚠️ **Inconsistency**: Different architectures per task
- ⚠️ **Technical debt**: Emergency fixes in experiments/

### Mitigation:
- Add dropout (currently 0.5, maybe increase)
- Use weight decay (currently 0.01)
- Early stopping (patience=20)
- Gradient clipping (currently 1.0)

## Bottom Line

**We're using a sledgehammer (163k features) to crack a nut (6-class problem).**

The correct solution is likely:
1. Better feature extraction (attention over patches)
2. Task-specific feature selection
3. Proper architectural design in src/

**Current training is a diagnostic experiment, not a production solution.**

## Monitoring Checklist

Watch for these red flags:
- [ ] Loss stays above 5.0 after epoch 3
- [ ] Validation BAcc < 0.30 after epoch 5
- [ ] Memory usage > 20GB
- [ ] Training time > 30min/epoch

If ANY occur, abort and redesign.

---

**Next Update**: After epoch 5 completes (~30 minutes)
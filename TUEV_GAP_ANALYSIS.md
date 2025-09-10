# TUEV Gap Analysis: Our Implementation vs Reference

**Created**: September 10, 2025  
**Purpose**: Identify concrete differences between our implementation and EEGPT reference  
**Impact**: Our BAC=0.19-0.24 vs Reference BAC=0.62

## Pre-Flight Critical Issues (Fix FIRST)

### 0. Data Split Mismatch ⚠️ **CRITICAL - Could explain 20-30% BAC gap**
| Aspect | Reference | Ours | Impact |
|--------|-----------|------|--------|
| Split method | Subject-based | Pre-split dirs or fallback | Different subjects in eval |
| Random seed | 4523 | 42 or undefined | Non-reproducible splits |
| Result | Consistent eval set | Potentially leaked/shifted eval | **Invalid comparison** |

**Action**: Verify our splits are subject-based with NO leakage between train/eval.

## Critical Divergences (Likely Causing Performance Gap)

### 1. Class Balancing ⚠️ **MAJOR DIFFERENCE**
| Aspect | Reference | Ours | Impact |
|--------|-----------|------|--------|
| Sampling | **NO balancing** | WeightedRandomSampler | Model sees different distribution |
| Class weights | None | 1/class_count weights | Changes gradient magnitudes |
| Result | 62% BAC with imbalance | 19-24% BAC with balancing | **OPPOSITE of expected** |

**Hypothesis**: WeightedRandomSampler may be **hurting** performance by oversampling rare classes too aggressively.

### 2. Data Scale/Normalization ⚠️ **MAJOR DIFFERENCE**
| Aspect | Reference | Ours | Impact |
|--------|-----------|------|--------|
| Data scale | Raw microvolts (μV) | Volts → normalized to N(0, 50μV) | Different input distributions |
| Range | ~[-100, +100] μV typical | [-2, +2] after normalization | Model sees different magnitudes |
| Normalization | None mentioned | mean=0, std=50μV default | May affect learned features |

### 3. Batch Size & Accumulation
| Aspect | Reference | Ours | Impact |
|--------|-----------|------|--------|
| Total batch | 400 | 32 × 12 steps = 384 | Slightly smaller effective batch |
| Distribution | 2 GPUs, DDP | Single GPU | Different gradient statistics |

## Moderate Divergences (Could Contribute 5-10% Each)

### 4. Mean Pooling Strategy
| Aspect | Reference | Ours | Impact |
|--------|-----------|------|--------|
| Feature reduction | use_mean_pooling flag | Flatten 4 tokens | Different feature dims |
| Output size | 512 (if pooled) | 2048 (flattened) | 4x larger feature vector |
| Classifier input | Variable | Always 2048 | May need different head |

### 5. Training Infrastructure
| Aspect | Reference | Ours | Impact |
|--------|-----------|------|--------|
| GPUs | 2 with DDP/DeepSpeed | 1 with accumulation | Different gradient stats |
| Batch sync | Every step | Every N steps | Stale gradients |

## Minor Divergences (Less Impact)

### 4. Implementation Details
| Aspect | Reference | Ours | Status |
|--------|-----------|------|--------|
| Window extraction | Fixed 1000 samples | Fixed 1000 samples | ✅ Same |
| Channel mapping | 23→20 Conv2d | 23→20 TUEVChannelMapper | ✅ Equivalent |
| Label mapping | Subtract 1 (1-6→0-5) | Explicit dict {spsw:0...} | ✅ Same semantics |
| Label smoothing | 0.1 | 0.1 | ✅ Same |
| Learning rate | 5e-4 | 5e-4 | ✅ Same |
| Layer decay | 0.65 | 0.65 | ✅ Same |
| Warmup | 5 epochs | 5 epochs | ✅ Same |
| DropPath | 0.2 | Not implemented | ⚠️ Missing regularization |

### 5. Data Pipeline
| Aspect | Reference | Ours | Status |
|--------|-----------|------|--------|
| Filtering | 0.1-75Hz + 50Hz notch | 0.1-75Hz + 50Hz notch | ✅ Same |
| Sampling rate | 200 Hz | 200 Hz | ✅ Same |
| Segment length | 5s (1000 samples) | 5s (1000 samples) | ✅ Same |
| Channels | 23 referential | 23 referential | ✅ Same |

## Revised Action Plan (Pre-Flight Checks FIRST)

### 0. 🔴 PRE-FLIGHT: Verify Data Splits
```bash
# Check if we're using subject-based splits
uv run python -c "
from brain_go_brrr.infra.data.tuev_event_dataset import TUEVEventDataset
import json
# Check train subjects
train_idx = json.load(open('data/datasets/tuev/cache/tuev_event_segments/train/index.json'))
train_subjects = set([seg['file_path'].split('/')[-1].split('_')[0] for seg in train_idx['segments'][:100]])
# Check eval subjects  
eval_idx = json.load(open('data/datasets/tuev/cache/tuev_event_segments/eval/index.json'))
eval_subjects = set([seg['file_path'].split('/')[-1].split('_')[0] for seg in eval_idx['segments'][:100]])
print(f'Train subjects sample: {list(train_subjects)[:5]}')
print(f'Eval subjects sample: {list(eval_subjects)[:5]}')
print(f'Overlap: {train_subjects & eval_subjects}')  # Should be empty set!
"
```
**If overlap exists**: Data leakage! Must rebuild cache with proper subject splits.

### 1. 🔴 THEN: Disable WeightedRandomSampler
```python
# REMOVE THIS:
train_sampler = WeightedRandomSampler(...)
train_loader = DataLoader(..., sampler=train_sampler)

# REPLACE WITH:
train_loader = DataLoader(..., shuffle=True)  # Standard random sampling
```
**Rationale**: Reference achieves 62% WITHOUT balancing. Our balancing may be overcorrecting.

### 2. 🟡 Test Different Normalization
```python
# Option A: Match reference (no normalization)
wrapper = EEGPTWrapper(..., normalize=False)  # If supported

# Option B: Use corpus statistics
# Compute mean/std from entire TUEV dataset
stats = compute_corpus_stats()
wrapper = EEGPTWrapper(..., stats_file=stats)
```

### 3. 🟡 Match Exact Batch Size
```python
# Increase to exactly 400 effective batch
batch_size = 34  # 34 × 12 = 408 ≈ 400
# OR
accumulate_steps = 13  # 32 × 13 = 416 ≈ 400
```

### 4. 🟢 Add DropPath Regularization
```python
# In model definition
self.drop_path = DropPath(drop_prob=0.2) if training else nn.Identity()
```

## Validation Metrics

After each change, monitor:
1. **Per-class recall** in confusion matrix (especially Class 0 spsw)
2. **Training distribution** (print batch class counts)
3. **BAC progression** by epoch

## Expected Outcomes

1. **Remove sampler** → Expect BAC to jump from 0.24 → 0.35-0.45
2. **Fix normalization** → Additional +0.10-0.15 BAC
3. **Match batch size** → Minor improvement +0.02-0.05
4. **Add DropPath** → Stabilization, not necessarily higher BAC

## The Smoking Guns 🔫

### 1. Wrong Data Splits
**If train/eval subjects overlap or use different seeds, we're not even testing the same problem!**

### 2. Class Balancing Paradox  
**The reference achieves 62% BAC with severe imbalance WITHOUT any balancing.**
- They let the model see natural distribution
- We force equal representation  
- Result: We're teaching a false reality

### 3. Normalization Mismatch
**Reference uses raw μV, we normalize to N(0,50μV)**
- Changes feature magnitudes by orders of magnitude
- EEGPT was pretrained on specific scales

## Refined Hypothesis

It's NOT just the sampler. Multiple issues compound:
1. **Wrong splits** → Testing on wrong data (20-30% impact)
2. **Wrong sampler** → False distribution (10-20% impact)  
3. **Wrong normalization** → Miscalibrated features (5-15% impact)
4. **Wrong pooling** → Different feature dims (5-10% impact)
5. **Missing DropPath** → Less regularization (2-5% impact)

**Total**: These could explain the full 40% BAC gap!

## Next Steps (In Order)

1. **Verify splits** - Ensure NO subject leakage
2. **Remove sampler** - Test natural distribution
3. **Fix normalization** - Try raw μV
4. **Match pooling** - Test mean vs flatten
5. **Add DropPath** - Stabilize training

**Only by fixing ALL of these will we reach 62% BAC.**
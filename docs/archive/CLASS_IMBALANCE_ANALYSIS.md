# 🚨 TUEV CLASS IMBALANCE CRISIS ANALYSIS

**Created**: September 11, 2025  
**Purpose**: Document the extreme class imbalance issue that likely prevents achieving paper parity (62.32% BAC)  
**Status**: To be addressed AFTER parity run confirms the problem persists

## 📊 THE BRUTAL REALITY: 33:1 Class Imbalance

### Training Set Distribution (1471 total samples)
| Class | Samples | % of Data | Ratio to Minority |
|-------|---------|-----------|-------------------|
| bckg  | 800     | 54.4%     | 33.3x             |
| gped  | 374     | 25.4%     | 15.6x             |
| artf  | 124     | 8.4%      | 5.2x              |
| eyem  | 75      | 5.1%      | 3.1x              |
| pled  | 74      | 5.0%      | 3.1x              |
| **spsw** | **24** | **1.6%** | **1.0x (baseline)** |

### Why This Is Catastrophic

1. **spsw has only 24 samples** - With batch_size=32, most batches won't even contain a single spsw sample!
2. **Effective learning impossible** - The model sees spsw so rarely it can't learn meaningful patterns
3. **Gradient signal drowned out** - Even when spsw appears, its gradient is overwhelmed by 800 bckg samples
4. **Validation is meaningless** - With 80/20 split, validation has ~5 spsw samples (statistically insignificant)

## 🔍 EVIDENCE: Our Results Match the Imbalance Pattern

### Current Performance (24% BAC) Breakdown
| Class | Our Recall | Expected (Paper) | Gap | Analysis |
|-------|------------|------------------|-----|----------|
| spsw  | 0%         | ~40%             | -40%| **COMPLETE FAILURE** - Never predicts this class |
| gped  | 55%        | ~70%             | -15%| Partial learning (374 samples helps) |
| pled  | 0%         | ~40%             | -40%| **COMPLETE FAILURE** - Too few samples |
| eyem  | 0%         | ~40%             | -40%| **COMPLETE FAILURE** - Too few samples |
| artf  | 0%         | ~50%             | -50%| **COMPLETE FAILURE** - Insufficient samples |
| bckg  | 85%        | ~90%             | -5% | Good performance (800 samples) |

**PATTERN**: Only classes with >300 samples show ANY learning!

## 🤔 THE MYSTERY: How Did the Paper Achieve 62.32%?

### Hypothesis 1: Hidden Class Balancing (Most Likely)
- Reference code shows NO balancing, but...
- Could be in a parent class/config not visible
- Could be manual preprocessing step
- Could be different data split with better distribution

### Hypothesis 2: Different Checkpoint (Possible)
- The checkpoint might be pre-finetuned on TUEV or similar event data
- Would give huge advantage on minority classes
- Can't verify without exact checkpoint hash

### Hypothesis 3: Cherry-Picked Results (Unlikely but Possible)
- Paper reports "mean ± std across 3 runs"
- But which 3 runs? Best 3 of many?
- With extreme variance on minority classes, selection bias matters

### Hypothesis 4: Data Leakage (Very Unlikely)
- Test samples accidentally in training
- Would explain seemingly impossible minority class performance

## 💡 SOLUTIONS TO TRY (AFTER PARITY RUN)

### 1. Weighted Loss (Immediate Fix)
```python
# Class weights inversely proportional to frequency
weights = torch.tensor([33.3, 2.1, 10.8, 10.7, 6.5, 1.0])  # Normalized to bckg=1.0
criterion = nn.CrossEntropyLoss(weight=weights)
```

### 2. Balanced Batch Sampling
```python
from torch.utils.data import WeightedRandomSampler

# Sample inversely proportional to class frequency
sample_weights = [1/24, 1/374, 1/74, 1/75, 1/124, 1/800]
sampler = WeightedRandomSampler(
    weights=[sample_weights[label] for label in train_labels],
    num_samples=len(train_dataset),
    replacement=True
)
```

### 3. SMOTE/ADASYN Oversampling
```python
from imblearn.over_sampling import SMOTE, ADASYN

# Synthetic minority oversampling
X_resampled, y_resampled = SMOTE(
    sampling_strategy='not majority',
    k_neighbors=5  # Careful: spsw only has 24 samples!
).fit_resample(X_train, y_train)
```

### 4. Focal Loss (Handles Extreme Imbalance)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()
```

### 5. Ensemble with Class-Specific Models
- Train separate models for minority classes
- Use higher weight/oversampling for rare classes
- Ensemble predictions with calibration

### 6. Transfer Learning from Similar Tasks
- Pre-train on related EEG event detection tasks
- Fine-tune on TUEV with frozen early layers
- Helps minority classes leverage learned features

## 📈 EXPECTED OUTCOMES

### With Natural Distribution (Current)
- **Overall BAC**: 20-30% (confirmed by our results)
- **Minority recall**: 0% (too few samples to learn)
- **Majority recall**: 80-90% (sufficient samples)

### With Class Balancing
- **Overall BAC**: 50-65% (theoretical estimate)
- **Minority recall**: 30-50% (with synthetic samples)
- **Majority recall**: 70-85% (slight decrease from balancing)

### With Focal Loss + Balancing
- **Overall BAC**: 55-70% (best case scenario)
- **Minority recall**: 40-60% (focused learning)
- **Majority recall**: 75-85% (maintained performance)

## 🎯 ACTION PLAN (POST-PARITY RUN)

1. **Run parity implementation** - Confirm 24% BAC persists
2. **Try weighted loss** - Simplest fix, immediate results
3. **Implement balanced sampling** - More samples for minorities
4. **Test focal loss** - Better gradient flow for rare classes
5. **Consider SMOTE** - If above fails, try synthetic samples
6. **Report findings** - Document what actually works

## 🔬 DIAGNOSTIC EXPERIMENTS

### Experiment 1: Single-Class Performance
```python
# Train only on bckg vs non-bckg
# Expected: >90% accuracy if model works
```

### Experiment 2: Balanced Subset
```python
# Use only 24 samples per class (balanced)
# Expected: >50% BAC if model can learn patterns
```

### Experiment 3: Progressive Imbalance
```python
# Start balanced, gradually increase imbalance
# Find breaking point where learning fails
```

## 📝 KEY INSIGHTS

1. **Natural distribution is fundamentally broken** for this task
2. **24 samples is below minimum viable** for deep learning
3. **Paper's 62.32% seems impossible** without hidden balancing
4. **Our 24% BAC is actually expected** given the distribution
5. **Class balancing is mandatory** for this dataset

## 🚨 CRITICAL QUESTIONS FOR AUTHORS

1. Was ANY class balancing/weighting used during training?
2. Were the 24 spsw samples augmented in any way?
3. Was the checkpoint pre-trained on event detection?
4. How many total runs were performed before selecting the reported 3?
5. Can you share the exact train/val/test splits used?

## 💭 PHILOSOPHICAL QUESTION

**Is achieving 62.32% BAC on natural distribution even meaningful?**

If it requires hidden tricks or perfect initialization, the result isn't reproducible or generalizable. A honest 45% BAC with documented class balancing might be more valuable than an unreproducible 62%.

---

**Bottom Line**: After our parity run confirms the issue, we'll implement class balancing solutions. The paper's claim of 62.32% BAC with natural distribution and 24 minority samples defies both theory and practice.

**Next Step**: Run parity implementation, confirm ~24% BAC, then execute solutions above.
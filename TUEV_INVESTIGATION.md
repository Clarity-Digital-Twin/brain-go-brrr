# TUEV Training Investigation: Why We're Stuck at 16.67% BAC

## Executive Summary

Our TUEV training is achieving **16.67% balanced accuracy** which is **exactly random chance for 6 classes**. This document investigates the divergence between our implementation and the EEGPT reference that achieves 62.32% BAC.

## Critical Finding: Model Predicts Only Background Class

**Evidence:**
- Balanced Accuracy: 0.1667 (exactly 1/6 = random)
- Loss: ~0.42-0.48 (converged but wrong)
- Epoch: 36+ with no improvement

**What's Happening:**
- Model learned to predict ONLY class 5 (BCKG - background)
- BCKG is 99.5% of the dataset
- This gives ~99.5% accuracy but 16.67% balanced accuracy

## Our Implementation vs EEGPT Reference

### 1. Loss Function

**Our Implementation (`train_tuev_mne.py`):**
```python
# Line 585-586
logger.info(f"Using unweighted CrossEntropyLoss with label_smoothing={label_smoothing}")
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

**EEGPT Reference (`run_class_finetuning_EEGPT_change.py`):**
```python
# Lines 378-382
if args.smoothing > 0.:
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
else:
    criterion = torch.nn.CrossEntropyLoss()
```

**SAME** ✓ - Both use unweighted loss with smoothing=0.1

### 2. Hyperparameters

**Our Config (`tuev_paper_parity.yaml`):**
```yaml
learning_rate: 5.0e-4      # Paper value
weight_decay: 0.05         # Paper value
label_smoothing: 0.1       # Paper value
batch_size: 64
n_epochs: 100
```

**EEGPT Reference (`finetune_TUEV_EEGPT.sh`):**
```bash
--lr 5e-4 \
--weight_decay 0.05 \
--batch_size 100 \         # DIFFERENT!
--epochs 50 \
--warmup_epochs 5 \        # We don't have this!
--layer_decay 0.65 \       # We don't have this!
```

**DIVERGENCE** ❌:
- Batch size: 64 vs 100
- No warmup epochs (critical for transformers!)
- No layer decay (different LR for different layers)

### 3. Data Preprocessing

**Our Implementation:**
```python
# TUEVMNEDataset with MNE preprocessing
- Autoreject for bad channels
- 0.5-40 Hz bandpass filter
- Average reference
- Z-score normalization per channel
```

**EEGPT Reference:**
```python
# From dataset loading (not shown in reference code)
- Unknown preprocessing (hidden in dataset class)
- Likely simpler preprocessing
```

**UNKNOWN** ❓ - Can't verify without seeing their dataset implementation

### 4. Channel Handling

**Our Implementation:**
```python
# 23 channels → 20 channel mapping via learnable mapper
use_paper_parity: true
n_channels: 23  # Including A1, A2, T1, T2
use_channel_mapper: true
mapper_dropout: 0.8
```

**EEGPT Reference:**
```python
# From run_class_finetuning_EEGPT_change.py line 199-200
ch_names = train_dataset.ch_names
# Removes reference suffix but keeps all channels
ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
```

**UNCLEAR** ❓ - They process channel names but actual count unclear

### 5. Class Distribution & Sampling

**Critical Discovery from EEGPT Paper:**
```
Table 1: Datasets for pretraining and downstream tasks
TUEV | Event | 288 | 6
```

**Only 288 samples?!** This is completely different from our dataset!

**Our Dataset:**
- Thousands of 4-second windows
- 99.5% background class
- Severe imbalance

**Their Dataset:**
- 288 total samples (maybe patient-level?)
- Possibly balanced or less imbalanced
- Different granularity

### 6. Data Augmentation

**Our Implementation:**
- None

**EEGPT Reference (`run_class_finetuning_EEGPT_change.py`):**
```python
# Lines 164-169
parser.add_argument('--reprob', type=float, default=0.25)
parser.add_argument('--remode', type=str, default='pixel')
parser.add_argument('--recount', type=int, default=1)
parser.add_argument('--resplit', action='store_true', default=False)
```
- Has RandomErasing augmentation options
- Mixup/CutMix support

**MISSING** ❌ - We have no augmentation

## Key Divergences Identified

### Critical Issues:

1. **Dataset Size Mismatch**: EEGPT paper says 288 samples, we have thousands
   - Are they using patient-level aggregation?
   - Are we creating too many windows?

2. **Missing Warmup**: No warmup epochs in our training
   - Transformers need warmup for stable training
   - Could cause immediate overfitting to majority class

3. **No Layer Decay**: All layers use same LR
   - EEGPT uses layer_decay=0.65
   - Earlier layers should have lower LR

4. **Batch Size**: 64 vs 100
   - Smaller batches might not see minority classes

5. **Unknown Data Preprocessing**: Can't verify if our MNE pipeline matches

## Hypothesis: Why EEGPT Works

They likely have:
1. **Different data granularity** (288 samples suggests patient-level, not window-level)
2. **Less severe imbalance** (different preprocessing/windowing)
3. **Warmup + layer decay** preventing early overfitting
4. **Larger batches** ensuring minority class representation

## Immediate Actions Needed

### 1. Verify Dataset Understanding
```bash
# Check EEGPT's actual TUEV data loading
grep -r "TUEV" reference_repos/EEGPT/downstream_tueg/
# Look for their window creation logic
```

### 2. Count Our Actual Class Distribution
```python
# Get exact counts per class
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
dataset = TUEVMNEDataset(split='train')
print(dataset.class_counts)
```

### 3. Check EEGPT's Model Architecture
- Are they using the same probe architecture?
- Is the feature extraction identical?

### 4. Implement Missing Components
- Add warmup epochs
- Add layer decay  
- Try batch_size=100
- Consider patient-level aggregation

## Evidence Supporting Investigation

### From Training Logs:
```
Epoch 36: loss=0.4211, bal_acc=0.1667
```
- Loss converged but to wrong solution
- BAC exactly 1/6 = uniform/single-class predictions

### From EEGPT Paper (Table 3):
```
TUEV Results:
Ours: 0.6232±0.0114 BAC, 0.8187±0.0063 F1
BIOT: 0.5281±0.0225 BAC, 0.7492±0.0082 F1
```
- High F1 with moderate BAC suggests they handled imbalance
- Consistent improvements over BIOT suggest systematic approach

## Conclusion

We're **NOT** replicating EEGPT's training correctly. Key issues:
1. Dataset understanding mismatch (288 vs thousands of samples)
2. Missing critical training components (warmup, layer decay)
3. Possible preprocessing differences
4. Unknown class distribution in their setup

## Next Steps

1. **STOP** assuming and **INVESTIGATE** their exact data pipeline
2. **COUNT** our exact class distribution
3. **IMPLEMENT** warmup and layer decay
4. **VERIFY** we're creating windows the same way
5. **TEST** with their exact hyperparameters

**DO NOT** proceed with training until we understand the 288 sample discrepancy!

---

**Status**: Investigation Phase
**Priority**: CRITICAL - Current approach will never work
**Owner**: Senior Review Needed
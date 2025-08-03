# Phase 5: Epileptiform Detection Head

## Overview

After the abnormality detection head identifies abnormal EEG segments, we need a specialized head to categorize specific epileptiform events.

## Architecture

```
Abnormal EEG Windows
    ↓
EEGPT Features (frozen)
    ↓
Epileptiform Detection Head
    ├─ Binary: Has epileptiform activity? 
    └─ Multi-class: Spike / Sharp Wave / Spike-Wave / Polyspike
```

## Data Preparation

### 1. Generate Abnormal Dataset
```python
# From best abnormality checkpoint
abnormal_windows = []
for window in validation_set:
    if model.predict(window) > 0.8:  # High confidence abnormal
        abnormal_windows.append(window)
```

### 2. Label Requirements
Need expert annotations for:
- **Spike**: 20-70ms sharp transient
- **Sharp Wave**: 70-200ms transient
- **Spike-Wave Complex**: Spike followed by slow wave
- **Polyspike**: Multiple spikes in succession
- **Other Abnormal**: Non-epileptiform abnormalities

### 3. Expected Distribution
From literature (TUAB subset analysis):
- ~30% of abnormal windows contain epileptiform activity
- Distribution: Spikes (40%), Sharp waves (30%), Spike-wave (20%), Polyspike (10%)

## Model Architecture

### Option 1: Binary Cascade
```python
class EpileptiformBinaryHead(nn.Module):
    def __init__(self):
        self.has_epileptiform = nn.Linear(768, 2)  # Yes/No
        
class EpileptiformTypeHead(nn.Module):
    def __init__(self):
        self.event_type = nn.Linear(768, 4)  # 4 types
```

### Option 2: Multi-label
```python
class EpileptiformMultiLabelHead(nn.Module):
    def __init__(self):
        self.classifier = nn.Linear(768, 5)  # 4 types + none
        # Use BCEWithLogitsLoss for multi-label
```

### Option 3: Hierarchical Softmax
```python
class HierarchicalEpileptiformHead(nn.Module):
    def __init__(self):
        self.level1 = nn.Linear(768, 2)  # Has epileptiform?
        self.level2 = nn.Linear(768, 4)  # Which type?
```

## Training Strategy

### 1. Data Split
- Use only windows with abnormality_score > 0.7
- 70/15/15 train/val/test split
- Stratify by event type

### 2. Loss Function
```python
# For imbalanced classes
class_weights = compute_class_weight('balanced', 
                                    classes=np.unique(y_train),
                                    y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

### 3. Metrics
- Primary: F1-score (macro-averaged)
- Secondary: Per-class precision/recall
- Clinical: False negative rate for spikes

### 4. Hyperparameters
```yaml
training:
  learning_rate: 1e-4  # Lower than abnormality head
  weight_decay: 0.1
  epochs: 30
  early_stopping_patience: 5
  batch_size: 32  # Smaller due to fewer samples
```

## Implementation Plan

### Phase 5.1: Data Generation
1. Load best abnormality checkpoint
2. Process full TUAB dataset
3. Extract high-confidence abnormal windows
4. Save as separate dataset

### Phase 5.2: Annotation
1. Create annotation tool/interface
2. Get expert labels (or use TUAB annotations if available)
3. Validate inter-rater agreement

### Phase 5.3: Model Training
1. Implement hierarchical softmax head
2. Add class balancing
3. Train with frozen EEGPT backbone
4. Evaluate on held-out test set

### Phase 5.4: Integration
1. Add to hierarchical pipeline
2. Update confidence thresholds
3. Add temporal clustering (merge nearby events)
4. Clinical validation

## Expected Performance

Based on literature:
- Binary detection: F1 > 0.75
- Multi-class: F1 > 0.65
- Clinical utility: >90% sensitivity for spikes

## Code Structure

```
src/brain_go_brrr/
├── tasks/
│   └── epileptiform_detection.py
├── models/
│   └── epileptiform_heads.py
└── data/
    └── epileptiform_dataset.py

scripts/
├── generate_epileptiform_dataset.py
└── train_epileptiform_detector.py

tests/
└── unit/
    └── test_epileptiform_detection.py
```

## Dependencies

- Existing: EEGPT backbone, abnormality head
- New: None (reuses same infrastructure)
- External: Expert annotations (or TUAB labels)

## Timeline

1. Week 1: Generate abnormal dataset, implement heads
2. Week 2: Obtain/create annotations
3. Week 3: Train and evaluate models
4. Week 4: Integration and clinical validation

## Notes

- Start simple with binary detection
- Multi-class only if binary works well
- Consider temporal context (nearby windows)
- May need post-processing for clinical use
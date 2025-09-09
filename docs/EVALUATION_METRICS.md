# 📊 Evaluation Metrics - Single Source of Truth

**Purpose**: Define EXACTLY which metrics to use for each dataset. No confusion, no mixing tasks.

---

## Quick Reference Table

Reporting: TUEV uses 3 runs (seeds: 42, 123, 456); TUAB currently single seed (42).
Monitors: TUAB → AUROC (binary); TUEV → Kappa per paper (but our code monitors BAC†).

†*Implementation note: train_tuev_mne.py saves best model by BAC, not Kappa*

| Dataset | Task Type | Primary Metrics | Secondary Metrics | Threshold Policy |
|---------|-----------|-----------------|-------------------|------------------|
| **TUAB** | Binary Classification | AUROC, Balanced Accuracy | Spec@Sens={0.90,0.95}, Kappa | Choose τ on VAL for target Sens, report on TEST |
| **TUEV** | 6-Class Classification | Weighted F1, Balanced Accuracy, Kappa | Confusion Matrix (secondary) | Softmax; monitor Kappa |
| **TUSZ** | Temporal Detection (Future) | FA/24h@Sens, TAES | Latency, DET curve | Threshold + post-proc on VAL, freeze for TEST |

---

## TUAB: Abnormal vs Normal (Binary Classification)

### Task Definition
- **Type**: Binary classification per EEG recording
- **Classes**: Normal (0) vs Abnormal (1)
- **NOT a temporal task** - no FA/24h needed!

### Metrics (What EEGPT Paper Reports)
From our local EEGPT markdown (see `literature/markdown/EEGPT/EEGPT.md`, Table 11):
- **AUROC**: 87.18% ± 0.5%
- **Balanced Accuracy**: 79.83% ± 0.4%  
- **Cohen's Kappa**: 0.60 ± 0.01

*Note: Paper reports mean ± std from 3 runs; our TUAB config currently uses single seed*

### Additional Clinical Metrics (We Add)
- **Specificity @ Sensitivity = 0.90**: How many normals correctly identified when catching 90% of abnormals
- **Specificity @ Sensitivity = 0.95**: How many normals correctly identified when catching 95% of abnormals

### Evaluation Protocol
```python
# 1. Train linear probe on TRAIN set
# 2. Find threshold on VAL set for target sensitivity
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, scores_val)
target = 0.95
meets = np.where(tpr >= target)[0]
if len(meets) == 0:
    best_idx = int(np.argmax(tpr))  # fallback if not achievable
else:
    best_idx = int(meets[0])  # largest τ meeting target
threshold = thresholds[best_idx]

# 3. Apply threshold ONCE on TEST set
y_test_pred = (scores_test >= threshold).astype(int)

# 4. Report metrics
spec90, _ = spec_at_sens(y_test, scores_test, 0.90)
spec95, _ = spec_at_sens(y_test, scores_test, 0.95)
metrics = {
    'auroc': roc_auc_score(y_test, scores_test),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'spec_at_sens_90': spec90,
    'spec_at_sens_95': spec95,
    'kappa': cohen_kappa_score(y_test, y_test_pred)
}
```

---

## TUEV: Event Classification (6-Class)

**📊 Reference**: See [TUEV_METRICS_SSOT.md](../TUEV_METRICS_SSOT.md) for exact target values and thresholds.

**⚠️ Architecture Note**:
- **Paper parity (IMPLEMENTED)**: 23‑channel raw input + learned Conv2d(23→20) mapper (BN/GELU + depthwise 1×55 + BN/Dropout 0.8). No channel synthesis.
- Legacy (archived): 20‑channel preprocessing with Fpz interpolation; not paper‑parity and not used for current experiments.

### Task Definition
- **Type**: 6-class classification per event window
- **Classes**: SPSW (spike), GPED, PLED, EYEM (eye movement), ARTF (artifact), BCKG (background)
- **NOT a temporal detection task** - just classify each window

### Metrics (What EEGPT Paper Reports)
Paper-aligned multi-class metrics (targets from TUEV_METRICS_SSOT.md):
- **Weighted F1**: 81.87% ± 0.63% (misleading due to 99.5% class imbalance)
- **Balanced Accuracy (BAC)**: 62.32% ± 1.14% (true performance metric)
- **Cohen's Kappa**: 0.635 ± 0.013 (paper's monitor metric)

### Evaluation Protocol
```python
# 1. Train linear probe with 6-class output
# 2. No threshold needed - use softmax for class selection
y_pred = np.argmax(model.predict_proba(X_test), axis=1)

# 3. Report metrics (paper-aligned)
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

metrics = {
    'weighted_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    'kappa': cohen_kappa_score(y_test, y_pred),
    'confusion_matrix': confusion_matrix(y_test, y_pred),  # secondary
}
```

### Channel Note
- TUEV parity keeps all 23 raw channels, including A1/A2/T1/T2, with mixed‑case canonical names.
- A learnable mapper converts 23→20 before EEGPT; we do not synthesize channels in parity mode.

---

## TUSZ: Seizure Detection (Future Work - Temporal)

### ⚠️ DIFFERENT TASK TYPE
This is **temporal event detection**, NOT classification. Requires different metrics!

### Task Definition
- **Type**: Detect seizure start/end times in continuous EEG
- **Output**: List of (start_sec, end_sec) events
- **Challenge**: Very imbalanced (99%+ non-seizure)

### Metrics (Clinical Standard)
- **FA/24h @ Sensitivity**: False alarms per 24 hours at fixed sensitivity (e.g., 95%)
- **Sensitivity @ FA/24h**: What sensitivity achieved at acceptable FA rate (e.g., <10 FA/24h)
- **TAES**: Time-Aligned Event Scoring (Jaccard index ≥ 0.5 for match)
- **Latency**: Seconds from true onset to detection

### Post-Processing Parameters (Tune on VAL)
- **min_duration**: Minimum event length (e.g., 2-5 seconds)
- **merge_gap**: Gap to merge nearby detections (e.g., 3-5 seconds)
- **hysteresis**: Different thresholds for onset/offset

### Why Different from TUAB/TUEV
- **TUAB/TUEV**: "Is this recording/window abnormal?" → Classification metrics
- **TUSZ**: "When do seizures start/stop?" → Temporal metrics with FA/24h

---

## Key Principles

### 1. Patient-Level Splits (No Leakage!)
```python
# CORRECT: Split by patient ID
train_patients, test_patients = split_patient_ids(patient_ids, test_size=0.2)

# WRONG: Split by samples (leaks patient data across sets)
X_train, X_test = train_test_split(X, test_size=0.2)  # ❌ DON'T DO THIS
```

### 2. Threshold Selection Protocol
- **TUAB**: Choose threshold on VAL to achieve target sensitivity
- **TUEV**: No threshold (multiclass uses argmax)
- **TUSZ**: Optimize FA/24h vs sensitivity trade-off on VAL

### 3. Single Evaluation on TEST
- Tune ALL parameters on VAL set
- Freeze everything (threshold, post-processing, etc.)
- Evaluate ONCE on TEST set
- Never iterate on TEST results!

---

## Common Mistakes to Avoid

❌ **Using FA/24h for TUAB/TUEV** - These aren't temporal tasks!
❌ **Choosing threshold on TEST set** - This is data leakage
❌ **Mixing patient data across splits** - Use patient-level splitting
❌ **Reporting "accuracy" alone** - Use balanced metrics for imbalanced data
❌ **Forgetting confidence intervals** - Report mean ± std from multiple runs

---

## Implementation Checklist

### For TUAB
- [ ] Load TUAB v3.0.1 with patient-level splits
- [ ] Extract EEGPT features: (B, 4, 512) with summary=False → flatten to (B, 2048)
- [ ] Train linear probe on TRAIN
- [ ] Find threshold for 90% and 95% sensitivity on VAL
- [ ] Report AUROC, BAC, Spec@Sens on TEST

### For TUEV
- [ ] Handle Fpz synthesis (zeros or learned adapter)
- [ ] Train 6-class linear probe
- [ ] Report Weighted F1, BAC, and Kappa metrics
- [ ] Generate confusion matrix

### For TUSZ (Future)
- [ ] Design sliding window approach
- [ ] Implement post-processing pipeline
- [ ] Add TAES scoring
- [ ] Optimize FA/24h trade-off

---

*Last Updated: September 6, 2025*

# 🎯 ROADMAP: From AUROC to Clinical Reality

## 📧 Canonical Expert Feedback (September 2, 2025)

**From TUH/NEDC Leadership:**
> "I think you need input from clinicians. The evaluation metric depends on the application. 
> For years, we have distributed our own scoring software that we believe addresses a problem 
> like seizure detection where segmentation and false positive rates are very important:
> https://isip.piconepress.com/publications/book_sections/2021/springer/metrics/
> 
> In terms of pipelines, I think what we lack is adequate annotated data... There is obviously 
> use for a portal that can analyze data without a need to train models, but moving these big 
> EEG files to/from such a portal is a problem."

**Mission**: Ship a clinically-useful EEG pipeline addressing these concerns:
1. ✅ Pick specific clinical application (TUAB abnormal detection first, TUSZ seizures next)
2. ⚡ Use the RIGHT metrics for EACH task (AUROC for TUAB, FA/24h for seizures)
3. 📦 Bring compute to data (local container, no cloud BS)

**Target**: Email expert reviewer in 60 days with working container + clinical metrics

---

## Phase 1: Foundation (Week 1-2) ✅ DONE
- [x] EEGPT integrated and working
- [x] TUAB dataset loading
- [x] Basic AUROC: 86.9% (EEGPT paper baseline)
- [x] Sleep staging with YASA: 87% accuracy
- [x] 899+ tests passing

## 🔗 Technical Bridge: EEGPT Paper → Working Pipeline

### What We Have (From EEGPT Paper)
- **Model checkpoint**: `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Architecture**: 10M params, 8 transformer layers, 512 embedding dim
- **Feature extraction**: 512-dim per 4s window (or 2048 flattened from 4×512)
- **Linear probe approach**: Freeze encoder, train only 1x1 conv + linear layer
- **Training details from paper**:
  - Optimizer: AdamW with OneCycle LR (2.5e-4 → 5e-4 → 3.13e-5)
  - Batch size: 64
  - Epochs: 200 for pretraining (but we only need ~10 for linear probe)
  - Data split: Patient-level, no leakage
- **Paper results on TUAB**: 
  - 86.9% ± 0.6% AUROC
  - 76.9% ± 0.4% Balanced Accuracy
  - Linear probe OUTPERFORMED full fine-tuning!

### What's Actually Working Now
```python
# This already works in our codebase:
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt
from brain_go_brrr.infra.data.tuab_dataset import TUABDataset

# Load model (WORKS ✅)
model = create_normalized_eegpt()

# Load data (WORKS ✅)
dataset = TUABDataset(root="/data/datasets/tuab")

# Extract features (WORKS ✅)
features = model.extract_features(eeg_window, summary=True)  # → (B, 512)
```

### The Gap to Bridge
1. **EEGPT gives**: Raw predictions (0-1 probabilities) at ONE operating point
2. **Clinicians need**: Multiple operating points with trade-offs documented
3. **Missing piece**: Threshold sweep + clinical metric calculation
4. **KEY INSIGHT**: EEGPT never reported FA/24h or Spec@Sens - we're the FIRST to bridge this gap!

### Concrete Implementation Path

#### Step 1: Linear Probe Training (FROM PAPER)
```python
# EEGPT paper approach: Freeze encoder, train linear head only
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt

# 1. Extract features with frozen EEGPT (they used this)
model = create_normalized_eegpt()
model.eval()  # Freeze the encoder

# 2. Add linear probe (1x1 conv + linear layer)
probe = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=1),  # Adaptive spatial filter
    nn.Flatten(),
    nn.Linear(512, 2)  # Binary classification
)

# 3. Train ONLY the probe (not the encoder!)
optimizer = AdamW(probe.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()
```

#### Step 2: Get Predictions (WE HAVE THIS)
```python
# After training probe, get predictions
with torch.no_grad():
    features = model.extract_features(test_data, summary=True)
    logits = probe(features)
    predictions = torch.sigmoid(logits).numpy()  # 0-1 scores
```

#### Step 3: Add Clinical Metrics (NEED TO ADD)
```python
from brain_go_brrr.domain.metrics import clinical_metrics

# For TUAB (classification)
results = {
    'auroc': roc_auc_score(labels, predictions),
    'balanced_acc': balanced_accuracy_score(labels, predictions > 0.5),
    'spec_at_95_sens': calculate_specificity_at_sensitivity(labels, predictions, 0.95)
}

# For TUSZ (temporal - future)
results = {
    'fa_per_24h': calculate_fa_per_24h(predictions, labels, threshold, total_hours),
    'taes': calculate_taes(predictions, labels, timestamps)
}
```

#### Step 4: Package for Deployment (NEED TO ADD)
```bash
# Create reproducible container
docker build -t brain-go-brrr:latest .
docker save brain-go-brrr:latest | gzip > bgb-container.tar.gz

# Or offline wheelhouse
pip wheel . --wheel-dir=wheelhouse
tar -czf bgb-offline.tar.gz wheelhouse/ models/
```

## Phase 2: Clinical Metrics (Week 3-4) 🚧 CURRENT

### Operating Point Selection (THE CRITICAL BRIDGE 🌉)

#### DECISION POLICY (EXPLICIT)
**TUAB Classification**: Pick the SMALLEST threshold τ achieving target sensitivity on VALIDATION set, FREEZE it, evaluate ONCE on test
**TUSZ Seizures**: MINIMIZE FA/24h subject to sensitivity ≥ target on VALIDATION set, FREEZE all params, evaluate ONCE on test

#### For TUAB (Binary Classification)
```python
# 1. Compute ROC on VALIDATION set
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, scores_val)

# 2. Find threshold for target sensitivity
target_sens = 0.95
idx = np.argmax(tpr >= target_sens)  # First threshold achieving target
threshold = thresholds[idx]

# 3. Calculate specificity at this threshold
tn = ((scores_val < threshold) & (y_val == 0)).sum()
fp = ((scores_val >= threshold) & (y_val == 0)).sum()
specificity = tn / (tn + fp)

# 4. FREEZE threshold, evaluate ONCE on test
y_test_pred = (scores_test >= threshold).astype(int)
```

#### For TUSZ (Temporal Events with FA/24h)
```python
# 1. Convert frame scores to events with post-processing
def scores_to_events(scores, threshold, gap_s=3, min_s=2):
    """Post-processing parameters:
    - gap_s: 3-5 seconds (tune on VAL, freeze for TEST)
    - min_s: 2-5 seconds (tune on VAL, freeze for TEST)
    """
    mask = scores >= threshold
    # Morphological operations to merge nearby detections
    mask = binary_closing(mask, structure=np.ones(gap_s * sample_rate))
    # Remove short events
    events = extract_events(mask)
    return [e for e in events if e.duration >= min_s]

# 2. Time-aligned matching using TAES (Picone 2021 methodology)
def match_events(pred_events, ref_events, overlap_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    for pred in pred_events:
        if any(overlap(pred, ref) >= overlap_threshold for ref in ref_events):
            tp += 1
        else:
            fp += 1
    fn = len(ref_events) - tp
    return tp, fp, fn

# 3. Sweep thresholds on VALIDATION to minimize FA/24h
best_threshold = None
min_fa_per_24h = float('inf')

# CRITICAL: Calculate total_hours correctly
total_hours_val = sum(recording.duration_seconds for recording in val_set) / 3600
if total_hours_val < 0.1:  # Guard against divide-by-zero
    raise ValueError(f"Insufficient validation data: {total_hours_val:.2f} hours")

for threshold in np.arange(0.3, 0.9, 0.05):
    events = scores_to_events(scores_val, threshold)
    tp, fp, fn = match_events(events, ref_events_val)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    fa_per_24h = (fp / total_hours_val) * 24
    
    if sensitivity >= 0.95 and fa_per_24h < min_fa_per_24h:
        min_fa_per_24h = fa_per_24h
        best_threshold = threshold

# 4. FREEZE all params, evaluate ONCE on test
```

### Implementation Checklist

#### For TUAB (Abnormal Detection - Classification)
- [ ] Implement ROC curve generation on validation set
- [ ] Find threshold for 90% and 95% sensitivity (pick SMALLEST τ achieving target)
- [ ] Calculate specificity at chosen thresholds
- [ ] Generate confusion matrix at each operating point
- [ ] Report metrics: `{auroc, balanced_accuracy, spec_at_sens=[0.90,0.95]}`
- [ ] Create `spec_at_sens(y_true, y_score, sens=0.95)` function

#### For TUSZ (Seizure Detection - Temporal Events)
- [ ] Implement post-processing pipeline (gap_s=3-5, min_s=2-5, tune on VAL)
- [ ] Add TAES (Time-Aligned Event Scoring) using Jaccard index
- [ ] Optional: Implement ATWV (NIST F4DE) for comparison
- [ ] Calculate FA/24h using ACTUAL recording duration (not 24h × file_count)
- [ ] Find threshold MINIMIZING FA/24h subject to sensitivity ≥ target
- [ ] Generate DET curves showing operating point trade-offs
- [ ] Report metrics: `{sens_at_fa_per_24h=[1,5,10], fa_per_24h_at_sens=[0.90,0.95], det_curve, taes}`

### Key Code to Add:
```python
# brain_go_brrr/domain/metrics/classification.py (TUAB)
def calculate_specificity_at_sensitivity(y_true, y_score, target_sensitivity=0.95):
    """For abnormal/normal classification - with proper confusion matrix"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Find threshold for target sensitivity
    idx = np.argmax(tpr >= target_sensitivity)
    threshold = thresholds[idx]
    
    # Calculate confusion matrix at this threshold
    y_pred = (y_score >= threshold).astype(int)
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    
    specificity = tn / (tn + fp)
    return specificity, threshold

# brain_go_brrr/domain/metrics/temporal.py (TUSZ)
def calculate_fa_per_24h(predictions, labels, threshold, total_hours):
    """For seizure detection - the ONE metric clinicians care about"""
    false_positives = count_temporal_false_alarms(predictions, labels, threshold)
    return (false_positives / total_hours) * 24
```

## Phase 3: Local Deployment (Week 5-6)

### 🔒 Core Principle: "Data Never Leaves This Machine"
- Bring compute TO the data (not data to compute)
- Support air-gapped environments (no internet required)
- All processing happens locally on hospital/lab infrastructure

- [ ] Create CLI command: `bgb eval tuab --metrics clinical`
- [ ] Build Docker container with all dependencies
- [ ] Create offline wheelhouse bundle for air-gapped install
- [ ] Write DEPLOY_LOCAL.md guide with bold **"data never leaves"** guarantee
- [ ] Test on fresh machine with no internet

### Deployment Targets:
```bash
# For TUAB (abnormal detection) - Binary Classification
docker pull ghcr.io/clarity-digital-twin/brain-go-brrr:latest
docker run -v /data/TUAB:/data:ro brain-go-brrr eval tuab \
  --data-root /data --out results/ \
  --metrics auroc,balanced_accuracy,spec_at_sens=0.90,spec_at_sens=0.95

# For TUSZ (seizure detection) - Temporal Events with TAES
docker run -v /data/TUSZ:/data:ro brain-go-brrr eval tusz \
  --data-root /data --out results/ \
  --metrics sens_at_fa_per_24h=1,sens_at_fa_per_24h=5,sens_at_fa_per_24h=10,\
            fa_per_24h_at_sens=0.90,fa_per_24h_at_sens=0.95,det_curve,taes
```

## Phase 4: Clinical Validation (Week 7-8)

### ⚠️ CRITICAL: Validation Methodology (NO DATA LEAKAGE!)
```python
# The ONLY correct way to evaluate
from sklearn.model_selection import train_test_split

# 1. Split data at PATIENT level (not sample level!)
train_patients, test_patients = split_patients(patient_ids, test_size=0.2)
train_patients, val_patients = split_patients(train_patients, test_size=0.25)

# 2. Train linear probe on TRAIN only
probe = train_linear_probe(train_data)

# 3. Find threshold on VAL (never touch TEST!)
best_threshold = find_threshold_on_validation(val_data, probe)

# 4. Evaluate ONCE on TEST with frozen threshold
test_results = evaluate(test_data, probe, threshold=best_threshold)

# THIS IS WRONG (data leakage):
# threshold = find_threshold(test_data)  # ❌ NEVER DO THIS
```
- [ ] Test on full TUAB canonical split (patient-level, no leakage)
- [ ] Document specificity at multiple sensitivity levels
- [ ] Create comparison table vs classical methods
- [ ] Generate reproducible results bundle with provenance.json
- [ ] Package as one-line install script
- [ ] Ensure deterministic results (seed all RNG)
- [ ] Create results bundle with:
  - **TUAB**: `metrics.json` (AUROC, BAC, Spec@Sens), `roc_curve.csv`
  - **TUSZ**: `metrics.json` (FA/24h, TAES scores), `events.csv`, `det_curve.csv`
  - **Both**: `provenance.json` (git SHA, image tag, seeds, CLI args, threshold values)
  - **Both**: `confusion_matrix.csv` (at each operating point)

### Success Metrics Tables:

#### TUAB (Abnormal Detection - Classification)
| Method | AUROC | Balanced Acc | Spec@95% Sens | Status |
|--------|-------|--------------|---------------|--------|
| Classical | ~75% | ~70% | ~60% | Baseline |
| EEGPT (paper) | 86.9% | 76.9% | ??? | Literature |
| **EEGPT (ours)** | **Target: 86%+** | **Target: 75%+** | **Target: 70%+** | **TODO** |

#### TUSZ (Seizure Detection - Temporal) [Future Work]
| Method | Sensitivity | FA/24h | TAES | Status |
|--------|-------------|--------|------|--------|
| Classical | 80% | 10-15 | ~0.6 | Baseline |
| **EEGPT (tuned)** | **95%** | **<10** | **>0.7** | **TARGET** |

## Phase 5: Expert Follow-up (Day 60)

Send follow-up email with concrete results and working container.
See `docs/internal/email-templates.md` for templates.

---

## Stretch Goals (If Time Permits)
- [ ] Add TUSZ seizure detection with TAES/ATWV
- [ ] Implement TUEV event classification
- [ ] Create watch-folder daemon mode
- [ ] Build Apptainer/Singularity image for HPC

## Anti-Goals (What NOT to Do)
- ❌ NO cloud/SaaS features yet
- ❌ NO fancy UI/frontend
- ❌ NO authentication/user management
- ❌ NO trying to solve data annotation problem
- ❌ NO scope creep beyond clinical metrics

---

## Daily Check-in Questions
1. Are we using the RIGHT metric for the RIGHT task?
2. Can this run on a hospital workstation?
3. Would an expert reviewer be impressed by the rigor?

## Resources
- Picone's metrics paper: `/literature/markdown/evaluation-metrics/picone-2021-objective-evaluation-metrics.md`
  - **KEY QUOTE**: "A low false alarm rate... is the single most important criterion for user acceptance"
  - Commercial systems fail due to high FA rates despite good accuracy
  - TAES (Time-Aligned Event Scoring) uses Jaccard index for overlap
- EEGPT paper baseline: 86.9% AUROC, 76.9% BAC on TUAB
- Key metric distinctions:
  - **TUAB (abnormal)**: AUROC, BAC, Specificity@Sensitivity
  - **TUSZ (seizures)**: FA/24h, TAES, ATWV, time-aligned scoring
- Clinical acceptance thresholds:
  - **Seizure detection**: <10 FA/24h at >95% sensitivity (inferred from literature)
  - **Why systems fail**: Even 90%+ accuracy is rejected if FA/24h too high

## Implementation Quality Bar
- **Single-responsibility modules** - Pure functions for metrics
- **Determinism** - Seed all RNG, log versions  
- **No raw data in artifacts** - Only JSON/CSV/plots
- **Patient-level splits** - No data leakage between train/test
- **Test coverage** - Unit tests for each metric function
- **No hidden I/O** - All file operations explicit

### Critical Edge Cases to Handle
- **Short recordings**: Guard divide-by-zero in FA/24h calculation
- **Threshold ties**: Many identical scores → use first index achieving target
- **Overlapping seizures**: Coalesce reference events before matching
- **Parameter leakage**: Fit thresholds on VAL, freeze for TEST (no peeking!)
- **Empty predictions**: Handle recordings with no detected events gracefully

### Unit Test Requirements
```python
# Test TUAB threshold selection
def test_spec_at_sensitivity():
    # Synthetic scores where we KNOW the answer
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9]
    spec, threshold = spec_at_sens(y_true, y_score, sens=0.67)
    assert threshold == 0.6  # Should pick this threshold
    assert spec == 0.67  # 2/3 true negatives correctly identified

# Test seizure FA/24h calculation
def test_fa_per_24h():
    # 1-hour recording with known events
    pred_events = [(10, 20), (100, 110), (500, 510)]  # 3 predictions
    ref_events = [(12, 18), (600, 610)]  # 2 actual seizures
    tp, fp, fn = match_events(pred_events, ref_events)
    assert tp == 1  # First pred matches first ref
    assert fp == 2  # Two false alarms
    assert calculate_fa_per_24h(fp, total_hours=1) == 48  # 2 FA/hr * 24
```

## Next 3 Concrete Commits (When Ready to Code)
1. `feat(metrics): classification Spec@Sens + tests`
   - Pure function implementation
   - Unit tests with synthetic scores
   - Assert monotonicity & edge cases

2. `feat(cli): bgb eval tuab command`
   - CLI command with metrics selection
   - Writes metrics.json, roc.csv, provenance.json
   - Deterministic and reproducible

3. `docs: DEPLOY_LOCAL.md`
   - Docker one-liners
   - Offline wheelhouse instructions
   - Bold "**data never leaves this machine**"

---

**Remember**: We're not building "AI for EEG" - we're building "clinically-useful tools that solve real problems"

*Last Updated: September 2, 2025*
*Target Completion: November 1, 2025*
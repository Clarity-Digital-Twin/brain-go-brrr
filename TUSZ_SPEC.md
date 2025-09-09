# TUSZ Temporal Seizure Detection Specification

## Executive Summary

TUSZ (Temple University Seizure) corpus temporal detection is the next critical milestone for Brain-Go-Brrr after achieving 83% AUROC on TUAB abnormality detection. This specification defines our approach to implementing time-step level seizure detection, targeting clinical deployment readiness with industry-standard evaluation metrics.

## Mission-Critical Requirements

### Clinical Performance Targets
- **Primary Metric**: ≤5 FA/24h at 90% sensitivity (clinical deployment threshold)
- **ATWV Score**: ≥0.40 (indicates practical utility)
- **Detection Latency**: ≤10 seconds from seizure onset
- **Inference Speed**: Real-time (≥4x faster than recording duration)

### Technical Requirements
- **Sampling Rate**: 256 Hz (EEGPT native rate)
- **Window Size**: 4 seconds with 2-second overlap
- **Channel Support**: 19-21 channels (TCP montage)
- **Model Architecture**: Dual approach (SeizureTransformer + EEGPT+BiLSTM)
- **Evaluation Framework**: NEDC Eval v6.0.0 compliance

## Architecture Decision: Dual-Model Approach

### Primary: SeizureTransformer Wrapper
```python
# Why SeizureTransformer?
- State-of-the-art performance on TUSZ (Wu et al., 2025)
- Direct time-step predictions (no window aggregation needed)
- Transformer architecture aligns with our EEGPT backbone
- Published benchmarks we can validate against
```

### Secondary: EEGPT + BiLSTM
```python
# Why EEGPT + BiLSTM?
- Leverages our existing EEGPT infrastructure
- BiLSTM captures temporal dynamics seizures exhibit
- Allows direct comparison of foundation model vs specialized model
- Potential for ensemble improvements
```

## Dataset Specifications

### TUSZ v2.0.1 Structure
```
Size: ~60GB compressed, ~150GB uncompressed
Subjects: 675 patients
Sessions: 1,478 recording sessions
Duration: ~1,500 hours total
Seizures: ~3,000 seizure events
Classes: Focal (FNSZ), Generalized (GNSZ), Combined (CNSZ), Unknown (UNSZ)
```

### Data Split Strategy
```python
# Official NEDC splits (must use for comparison)
train_set = "01_tcp_ar"  # 80% of data
dev_set = "02_tcp_ar"    # 10% of data
test_set = "03_tcp_ar"   # 10% of data (held out)

# Critical: Never train on dev/test sets
# Critical: Use patient-level splits (no data leakage)
```

## Evaluation Metrics (NEDC Eval v6.0.0)

### Clinical Metrics Hierarchy

#### 1. False Alarms per 24 Hours (FA/24h)
```python
# The gold standard metric for clinical acceptance
fa_per_24h = (false_positives / total_hours) * 24

# Clinical thresholds:
# <1 FA/24h: Excellent (ICU deployment ready)
# 1-5 FA/24h: Good (general ward deployment)
# 5-10 FA/24h: Acceptable (with human review)
# >10 FA/24h: Research only
```

#### 2. Time-Aligned Event Scoring (TAES)
```python
# Measures temporal overlap quality
sensitivity = TP / (TP + FN)  # Detection rate
precision = TP / (TP + FP)    # False alarm rate
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

# Report at multiple operating points:
sensitivity_levels = [0.80, 0.85, 0.90, 0.95]
```

#### 3. Any-Overlap Event Scoring (AOES)
```python
# Binary detection (less strict than TAES)
# Event is TP if ANY overlap exists with ground truth
# Used for high-level detection capability assessment
```

#### 4. Epoch-Based Scoring
```python
# Traditional window-level metrics
# 1-second epochs for fine-grained analysis
# Reports AUROC, AUPRC, accuracy, Cohen's kappa
```

### Advanced Metrics

#### Detection Latency
```python
latency = detection_time - true_onset_time
# Target: <10 seconds for clinical utility
# Critical for early intervention
```

#### Actual Term-Weighted Value (ATWV)
```python
# Balances sensitivity and false alarm rate
atwv = sensitivity - beta * fa_rate
# beta typically = 0.1 for seizure detection
# ATWV > 0.4 indicates practical utility
```

## Post-Processing Pipeline

### 1. Hysteresis Thresholding
```python
# Dual thresholds for robust detection
high_threshold = 0.7  # Start seizure
low_threshold = 0.3   # Continue seizure
# Reduces flickering in predictions
```

### 2. Gap Merging
```python
# Merge events separated by <10 seconds
min_gap_seconds = 10
# Handles brief prediction dropouts
```

### 3. Duration Filtering
```python
# Remove events shorter than threshold
min_seizure_duration = 5  # seconds
# Reduces false positives from artifacts
```

### 4. Collar Expansion
```python
# Extend boundaries for clinical safety
pre_seizure_collar = 5   # seconds before
post_seizure_collar = 10  # seconds after
# Ensures complete seizure capture
```

## Performance Optimization Strategy

### Computational Targets
```python
# Real-time constraint
processing_time = recording_duration / 4  # 4x speedup minimum

# Memory constraint
max_memory_gb = 8  # For edge deployment

# Batch processing
optimal_batch_size = 32  # Balance speed/memory
```

### Model Optimization
1. **Quantization**: INT8 for 2x speedup
2. **Pruning**: Remove 30% parameters without accuracy loss
3. **Knowledge Distillation**: Student model at 10% size
4. **ONNX Export**: Cross-platform deployment

## Clinical Integration Requirements

### Safety Protocols
```python
# Never suppress high-confidence detections
if confidence > 0.9:
    alert_immediately()

# Always flag for review
if confidence < 0.6:
    mark_for_human_review()

# Maintain audit trail
log_all_predictions_with_timestamp()
```

### Interpretability Features
1. **Attention Maps**: Which channels/times triggered detection
2. **Confidence Bands**: Uncertainty quantification
3. **Feature Attribution**: SHAP/LIME explanations
4. **Clinical Reports**: Automated summary generation

## Implementation Phases

### Phase 1: Baseline Establishment (Week 1-2)
- [ ] Set up TUSZ v2.0.1 dataset pipeline
- [ ] Implement NEDC Eval v6.0.0 metrics
- [ ] Train SeizureTransformer baseline
- [ ] Validate against published results

### Phase 2: EEGPT Integration (Week 3-4)
- [ ] Extract EEGPT features for TUSZ
- [ ] Train BiLSTM temporal model
- [ ] Implement post-processing pipeline
- [ ] Compare with SeizureTransformer

### Phase 3: Optimization (Week 5-6)
- [ ] Hyperparameter tuning
- [ ] Ensemble experiments
- [ ] Speed optimization
- [ ] Edge deployment testing

### Phase 4: Clinical Validation (Week 7-8)
- [ ] Cross-dataset evaluation
- [ ] Failure mode analysis
- [ ] Clinical metric reporting
- [ ] Documentation and deployment

## Risk Mitigation

### Known Challenges
1. **Class Imbalance**: ~2% seizure, 98% background
   - Solution: Focal loss, weighted sampling
   
2. **Patient Variability**: Seizures vary dramatically
   - Solution: Patient-specific fine-tuning option
   
3. **Artifact Confusion**: Movement/noise mimics seizures
   - Solution: Artifact rejection preprocessing
   
4. **Computational Cost**: Real-time constraint
   - Solution: Model optimization techniques

### Fallback Options
1. If SeizureTransformer fails: Use EEGPT+BiLSTM
2. If both fail targets: Ensemble approach
3. If speed too slow: Deploy quantized model
4. If accuracy too low: Require human review

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] FA/24h ≤ 10 at 85% sensitivity
- [ ] ATWV ≥ 0.30
- [ ] Real-time processing
- [ ] NEDC Eval compliance

### Production Ready
- [ ] FA/24h ≤ 5 at 90% sensitivity
- [ ] ATWV ≥ 0.40
- [ ] 4x real-time processing
- [ ] Clinical interpretability
- [ ] Cross-dataset validation

### Gold Standard
- [ ] FA/24h ≤ 1 at 95% sensitivity
- [ ] ATWV ≥ 0.50
- [ ] 10x real-time processing
- [ ] Patient-specific adaptation
- [ ] Multi-modal integration

## Key References

1. **SeizureTransformer Paper**: Wu et al., 2025 - "Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection"
2. **NEDC Eval Framework**: Picone et al., 2021 - "The Temple University Hospital EEG Corpus: Annotation Guidelines"
3. **Clinical Requirements**: Haider et al., 2016 - "Sensitivity of quantitative EEG for seizure identification"
4. **EEGPT Foundation**: Our TUAB implementation achieving 83% AUROC

## Approval Checklist

- [ ] Senior engineering review
- [ ] Clinical advisor validation
- [ ] Computational resource approval
- [ ] Timeline agreement
- [ ] Risk assessment sign-off

---

**Document Status**: AWAITING SENIOR REVIEW
**Last Updated**: 2025-01-09
**Next Review**: Upon senior approval
**Location**: Root directory (temporary) → docs/tusz/ (after approval)
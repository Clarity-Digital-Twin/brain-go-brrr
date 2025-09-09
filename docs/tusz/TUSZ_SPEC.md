# 🎯 TUSZ Temporal Seizure Detection - Specification & Planning

**Created**: September 9, 2025  
**Status**: 🔥 READY TO IMPLEMENT  
**Type**: Single Source of Truth for TUSZ Implementation  
**Priority**: HIGH - Next major feature after TUEV

---

## 📋 Executive Summary

TUSZ (Temple University Hospital Seizure Corpus) temporal detection is fundamentally different from TUAB/TUEV classification tasks. This specification defines the complete implementation plan for seizure onset/offset detection using a **dual-model approach**: SeizureTransformer wrapper for immediate baseline and EEGPT+BiLSTM for comparative analysis.

**Key Innovation**: We'll be the FIRST to evaluate April 2025 SOTA (SeizureTransformer) with proper clinical metrics (FA/24h, TAES, ATWV) that the original paper omitted.

---

## 🎓 Background & Requirements

### Dataset Specifications
- **Corpus**: TUSZ v1.1.1 (Temple University Hospital)
- **Train/Eval Split**: 196 train patients, 50 eval patients
- **Channels**: 22 channels (standard 10-20 montage)
- **Sampling Rate**: 256 Hz (matches EEGPT requirements)
- **Class Distribution**: ~8% seizure, ~92% background (massive imbalance)
- **Annotations**: Precise onset/offset times in seconds
- **File Format**: EDF files with TSE/CSV annotation files

### Task Definition
- **Type**: Temporal event detection (not classification)
- **Input**: Continuous EEG recordings (variable length)
- **Output**: List of seizure events with (start_time, end_time, confidence)
- **Challenge**: Precise temporal localization with low false alarm rate

### Clinical Requirements
- **Primary Metric**: FA/24h < 10 at 95% sensitivity
- **Secondary Metrics**: TAES > 0.5, ATWV > 0.5
- **Latency**: Detection within 10 seconds of onset
- **Processing Speed**: Real-time capability (future goal)

---

## 📊 Evaluation Metrics (SSOT)

### Primary Metrics

#### 1. **FA/24h @ Sensitivity Levels**
```python
# Clinical gold standard - what actually matters
fa_per_24h = (false_positives / total_hours) * 24
# Report at multiple sensitivity levels: [0.80, 0.85, 0.90, 0.95]
```
- **Target**: < 10 FA/24h at 95% sensitivity
- **Clinical Threshold**: < 5 FA/24h ideal

#### 2. **TAES (Time-Aligned Event Scoring)**
```python
# Temporal alignment quality using Jaccard index
jaccard = intersection_duration / union_duration
match = jaccard >= 0.5  # Standard threshold per Picone 2021
```
- **Target**: > 0.5 (50% sensitivity with good alignment)
- **Key**: Penalizes poor temporal alignment

#### 3. **ATWV (Actual Term-Weighted Value)**
```python
# Balances detection vs false alarms
ATWV = P(seizure) * P(correct|seizure) - β * P(FA|no_seizure)
# β = 999.9 for seizure detection (per NIST standards)
```
- **Target**: > 0.5 (indicates useful system)
- **Reality**: Most systems achieve < 0.45

### Evaluation Protocol (Critical)
```python
# NEVER tune on test data!
1. Train model on TRAIN set
2. Tune thresholds/post-processing on VAL set
3. Freeze ALL parameters
4. Evaluate ONCE on TEST set
5. Report metrics with confidence intervals
```

---

## 🏗️ Architecture Design

### Strategy: Dual-Model Approach

#### Phase 1: SeizureTransformer Wrapper (Days 1-3)
```
EEG → SeizureTransformer → Time-step Probabilities → Post-Processing → Events
     ↓
  NEDC Eval v6.0.0 → Clinical Metrics (FA/24h, TAES, ATWV)
```

**Advantages**:
- Immediate baseline with existing weights (169MB)
- First clinical validation of April 2025 SOTA
- Establishes evaluation infrastructure

**Architecture Details**:
- CNN Encoder: 32→512 filters
- Transformer: 8 layers, 4 heads, global attention
- U-Net Decoder: Skip connections for multi-scale
- Output: Per-sample probability (no windows!)

#### Phase 2: EEGPT + BiLSTM (Week 2)
```
EEG → Sliding Windows → EEGPT Features → BiLSTM → Post-Processing → Events
                                          ↓
                                   Same NEDC Eval Infrastructure
```

**Advantages**:
- Leverages existing EEGPT infrastructure
- Direct comparison using same evaluation
- Potentially better with our post-processing

**Architecture Details**:
- Windows: 4s with 2s overlap
- Features: 2048-dim EEGPT embeddings
- Temporal: BiLSTM with 256 hidden units
- Context: 30 windows (60s sequences)

### Reusable Infrastructure
```python
class TemporalSeizureWrapper:
    """Universal wrapper for any temporal detection model"""
    
    def __init__(self, backend='seizure_transformer'):
        self.backend = self._load_backend(backend)
        self.post_processor = AdvancedPostProcessor()
        self.evaluator = NEDCClinicalEvaluator()
    
    def evaluate(self, data, ground_truth):
        predictions = self.backend.predict(data)
        processed = self.post_processor.apply(predictions)
        metrics = self.evaluator.compute_all_metrics(processed, ground_truth)
        return metrics  # FA/24h, TAES, ATWV - proper clinical metrics!
```

---

## 🔧 Preprocessing Pipeline

### Signal Processing (Standardized)
```python
class TUSZPreprocessor:
    """
    Consistent preprocessing for all models.
    CRITICAL: Match SeizureTransformer's expectations.
    """
    
    def preprocess(self, raw_eeg):
        # 1. Channel selection (22 standard channels)
        selected = self.select_channels(raw_eeg, TUSZ_CHANNELS)
        
        # 2. Resampling (ensure 256 Hz)
        resampled = self.resample(selected, target_fs=256)
        
        # 3. Filtering
        filtered = self.bandpass_filter(resampled, low=0.5, high=50)
        notched = self.notch_filter(filtered, freq=60)  # US power line
        
        # 4. Normalization (Z-score per channel)
        normalized = self.z_score_normalize(filtered)
        
        # 5. Windowing (model-specific)
        if self.model_type == 'seizure_transformer':
            # 60-second chunks for transformer
            windows = self.create_windows(normalized, window_sec=60, hop_sec=30)
        else:  # EEGPT
            # 4-second windows with overlap
            windows = self.create_windows(normalized, window_sec=4, hop_sec=2)
        
        return windows
```

### Data Augmentation (Optional)
- Time shifting: ±0.5 seconds
- Amplitude scaling: 0.8-1.2x
- Gaussian noise: SNR > 20dB
- Channel dropout: Max 2 channels

---

## 🎮 Post-Processing Pipeline

### Three-Stage Approach (Critical for Performance)

#### Stage 1: Hysteresis Thresholding
```python
def hysteresis_threshold(probabilities, low=0.3, high=0.7):
    """
    Dual threshold for stability (reduces flickering).
    Start detection at high threshold, maintain until low.
    """
    in_event = False
    events = []
    
    for t, prob in enumerate(probabilities):
        if not in_event and prob > high:
            in_event = True
            event_start = t
        elif in_event and prob < low:
            in_event = False
            events.append((event_start, t))
    
    return events
```

#### Stage 2: Gap Merging
```python
def merge_gaps(events, max_gap_sec=2.0, fs=256):
    """
    Merge events separated by small gaps.
    Critical for reducing false alarms.
    """
    max_gap_samples = int(max_gap_sec * fs)
    merged = []
    
    for start, end in events:
        if merged and start - merged[-1][1] < max_gap_samples:
            merged[-1] = (merged[-1][0], end)  # Extend previous
        else:
            merged.append((start, end))
    
    return merged
```

#### Stage 3: Duration Filtering
```python
def filter_duration(events, min_sec=1.0, max_sec=600.0, fs=256):
    """
    Remove spurious short detections and unrealistic long ones.
    """
    min_samples = int(min_sec * fs)
    max_samples = int(max_sec * fs)
    
    return [(s, e) for s, e in events 
            if min_samples <= (e - s) <= max_samples]
```

### Tunable Parameters (Optimize on VAL)
- Hysteresis: (τ_low, τ_high) ∈ [(0.2, 0.6), (0.3, 0.7), (0.4, 0.8)]
- Gap merge: [1.0, 2.0, 3.0, 5.0] seconds
- Min duration: [0.5, 1.0, 2.0] seconds
- Max duration: [300, 600, 1200] seconds

---

## 📈 Performance Targets

### Minimum Viable (Week 1)
- FA/24h < 30 at 90% sensitivity
- TAES > 0.3
- ATWV > 0.3
- Process 1 hour in < 10 minutes

### Production Ready (Month 1)
- FA/24h < 10 at 95% sensitivity
- TAES > 0.5
- ATWV > 0.5
- Process 1 hour in < 2 minutes

### SOTA Competitive (Future)
- FA/24h < 5 at 95% sensitivity
- TAES > 0.7
- ATWV > 0.6
- Real-time processing

### Reality Check (Literature Baselines)
| System | Architecture | FA/24h | TAES | ATWV |
|--------|-------------|--------|------|------|
| CNN/LSTM (Picone 2021) | Best known | ~20 | 0.35 | 0.45 |
| SeizureTransformer | Unknown | Not reported | Not reported | Not reported |
| **Our Target** | Wrapper + Post-proc | <10 | >0.5 | >0.5 |

---

## 🚀 Implementation Phases

### Phase 1: Infrastructure (Days 1-2)
- [ ] Set up NEDC Eval v6.0.0 integration
- [ ] Create universal wrapper class
- [ ] Implement post-processing pipeline
- [ ] Build evaluation harness

### Phase 2: SeizureTransformer (Days 3-4)
- [ ] Load pretrained weights
- [ ] Create model wrapper
- [ ] Run inference on test samples
- [ ] Compute clinical metrics

### Phase 3: EEGPT + BiLSTM (Week 2)
- [ ] Design BiLSTM architecture
- [ ] Extract EEGPT features
- [ ] Train temporal head
- [ ] Compare with SeizureTransformer

### Phase 4: Optimization (Week 3)
- [ ] Hyperparameter tuning on VAL
- [ ] Post-processing optimization
- [ ] Ensemble methods (optional)
- [ ] Publication preparation

---

## 🎯 Key Design Decisions

### Why Dual Approach?
1. **SeizureTransformer**: Immediate baseline, validate April 2025 SOTA
2. **EEGPT + BiLSTM**: Leverage existing infrastructure, potentially better
3. **Same evaluation**: Fair comparison, reusable components

### Why NEDC Eval v6.0.0?
- Industry standard for TUSZ evaluation
- Implements TAES, OVLP, EPOCH correctly
- Used by all major papers
- Ensures reproducibility

### Why These Metrics?
- **FA/24h**: Clinical acceptance criterion
- **TAES**: Temporal alignment quality
- **ATWV**: Balances detection vs false alarms
- **NOT F1**: Misleading for temporal detection

---

## ⚠️ Risks & Mitigations

### Risk 1: SeizureTransformer Underperforms
- **Mitigation**: Have EEGPT + BiLSTM as backup
- **Reality**: Likely since no FA/24h reported

### Risk 2: Class Imbalance Issues
- **Mitigation**: Weighted loss, focal loss, SMOTE
- **Reality**: 8% seizure is challenging

### Risk 3: Poor Temporal Alignment
- **Mitigation**: Advanced post-processing
- **Reality**: Post-processing > model architecture

### Risk 4: High False Alarm Rate
- **Mitigation**: Ensemble methods, conservative thresholds
- **Reality**: FA/24h < 10 is very hard

---

## 📚 References

### Key Papers
- Picone 2021: "Objective Evaluation Metrics" - TAES/ATWV definitions
- Wu 2025: "SeizureTransformer" - April 2025 SOTA (missing metrics)
- Shah 2018: "TUSZ Corpus" - Dataset description
- NEDC Eval: https://github.com/TUH-NEDC/nedc_eval_eeg

### Our Contributions
1. First clinical evaluation of SeizureTransformer
2. Reusable temporal detection infrastructure
3. Direct comparison framework for future models
4. Bridge between competition metrics and clinical needs

---

## 🔍 Success Criteria

### Technical Success
- [ ] Both models running and producing predictions
- [ ] NEDC Eval integrated and computing metrics
- [ ] Post-processing improving raw predictions
- [ ] Reproducible results with seeds

### Clinical Success
- [ ] FA/24h < 10 at 95% sensitivity
- [ ] TAES > 0.5
- [ ] Physician feedback positive
- [ ] Ready for prospective validation

### Research Success
- [ ] Paper accepted to conference/journal
- [ ] Code released open-source
- [ ] Community adoption of wrapper
- [ ] New SOTA on proper metrics

---

**THIS SPECIFICATION IS THE SINGLE SOURCE OF TRUTH FOR TUSZ IMPLEMENTATION**

*Next Document: TUSZ_IMPLEMENTATION.md for execution details*
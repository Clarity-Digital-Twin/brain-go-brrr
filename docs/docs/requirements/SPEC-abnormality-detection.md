# EEG Abnormality Detection Specification

## Executive Summary

This document specifies the implementation of a production-ready EEG abnormality detection system for the Brain-Go-Brrr platform. Based on comprehensive literature review and clinical requirements, we will implement a foundation model approach using EEGPT with fine-tuning capabilities that achieves >80% balanced accuracy as required by the PRD.

## Clinical Significance

EEG abnormality detection is critical for:
- **Reducing diagnostic backlogs**: Current 48+ hour turnaround times
- **Improving detection rates**: 20% of subtle abnormalities currently missed
- **Enabling triage**: Urgent cases can be prioritized
- **Standardizing interpretation**: Reduces inter-reader variability

## Technical Approach

### Foundation Model Strategy

Based on literature review, foundation models significantly outperform models trained from scratch:

![Model Performance Comparison](../img/abnormality/_page_12_Figure_1.jpeg)

Performance targets by dataset:
- **BioSerenity-E1**: 94.63% balanced accuracy (3-expert consensus, best in class)
- **EEGPT (our model)**: 79.83% accuracy on TUAB dataset
- **CNN-LSTM from scratch**: 86.34% balanced accuracy
- **Transformer from scratch**: 88.72% balanced accuracy

Key findings from BioSerenity-E1 study:
- **89% balanced accuracy** on large private dataset
- **82% accuracy** on TUAB evaluation set (generalization)
- **94.6% accuracy** on 3-expert consensus subset

We will use EEGPT as our foundation model with the following advantages:
1. Pre-trained on diverse EEG data
2. Requires less labeled training data
3. Better generalization across equipment types
4. Faster training/fine-tuning

### Model Architecture

![BioSerenity-E1 Architecture](../img/abnormality/_page_6_Figure_2.jpeg)

The architecture consists of:
1. **Preprocessing Pipeline**: 0.5 Hz HPF → 45 Hz LPF → notch filter → resample to 128 Hz
2. **Foundation Model**: Frozen encoder with 16-channel subset
3. **Classification Head**: 3 conv layers + GAP + FC layers with dropout 0.4
4. **Window Aggregation**: 16s non-overlapping windows, mean-pooled to recording level

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw EEG       │────▶│  Preprocessing   │────▶│ Window Splitter │
│  (20 minutes)   │     │  & QC Check      │     │  (4s windows)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Triage Decision │◀────│ Ensemble Voting  │◀────│ EEGPT Encoder   │
│ & Confidence    │     │ & Aggregation    │     │ (frozen/tunable)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Detailed Specifications

### Input Requirements

1. **EEG Format**:
   - File format: EDF/BDF
   - Duration: 20-40 minutes typical (no hard limit)
   - Channels: 19-23 channels (10-20 system)
   - Sampling rate: Any (will be resampled to 256 Hz)

2. **Quality Requirements**:
   - Minimum 19 functional channels
   - Maximum 30% bad channels (else reject)
   - Recording must pass basic QC checks

### Preprocessing Pipeline

```python
# Preprocessing steps in order:
1. Load EDF/BDF file
2. Channel validation and mapping to 10-20 system
3. Bandpass filter: 0.5-50 Hz (5th order Butterworth)
4. Notch filter: 50/60 Hz based on region
5. Resample to 256 Hz
6. Average re-referencing
7. Z-score normalization per channel
8. Autoreject for artifact detection
9. Window extraction (4s with 50% overlap)
```

### Model Architecture

#### EEGPT Encoder (Foundation Model)
- **Parameters**: 10M (Large variant)
- **Input**: 256 Hz × 4 seconds × up to 58 channels
- **Patch size**: 64 samples (250ms)
- **Architecture**: Vision Transformer with masked autoencoding
- **Output**: Feature embeddings per window

#### Classification Head
- **Input**: Aggregated window features
- **Architecture**:
  ```
  Linear(512, 256) → BatchNorm → ReLU → Dropout(0.3)
  Linear(256, 128) → BatchNorm → ReLU → Dropout(0.3)
  Linear(128, 2) → Softmax
  ```

### Window-Level Processing

1. **Sliding Window Approach**:
   - Window size: 4 seconds (1024 samples)
   - Overlap: 50% (2 seconds)
   - Padding: Mirror padding for edge windows

2. **Per-Window Features**:
   - EEGPT embeddings: 512-dimensional
   - Quality score: 0-1 (from autoreject)
   - Position encoding: Relative position in recording

### Aggregation Strategy

Multiple aggregation methods for robustness:

1. **Weighted Average**:
   ```python
   weights = quality_scores * position_weights
   final_score = sum(window_scores * weights) / sum(weights)
   ```

2. **Voting Ensemble**:
   - Each window votes (threshold = 0.5)
   - Final = majority vote with confidence

3. **Attention-based Aggregation**:
   - Self-attention over all windows
   - Learns which windows are most informative

### Output Specifications

```python
{
    "abnormality_score": 0.0-1.0,      # Probability of abnormality
    "classification": "normal/abnormal", # Binary decision
    "confidence": 0.0-1.0,              # Model confidence
    "triage_flag": "NORMAL/ROUTINE/EXPEDITE/URGENT",
    "window_scores": [...],             # Per-window scores
    "quality_metrics": {
        "bad_channels": ["T3", "O2"],
        "quality_grade": "GOOD",
        "artifacts_detected": 12
    },
    "processing_time": 23.4,            # Seconds
    "model_version": "eegpt-v1.0"
}
```

### Triage Logic

Based on clinical requirements and abnormality score:

```python
if abnormality_score > 0.8 or quality_grade == "POOR":
    triage = "URGENT"      # Immediate review needed
elif abnormality_score > 0.6 or quality_grade == "FAIR":
    triage = "EXPEDITE"    # Priority review (< 4 hours)
elif abnormality_score > 0.4:
    triage = "ROUTINE"     # Standard workflow (< 48 hours)
else:
    triage = "NORMAL"      # Low priority
```

## Performance Requirements

### Accuracy Targets
- **Balanced Accuracy**: > 80% (PRD requirement)
- **Sensitivity**: > 85% (minimize false negatives)
- **Specificity**: > 75% (acceptable false positive rate)
- **AUROC**: > 0.85

### Processing Speed
- **Single recording**: < 30 seconds
- **Batch processing**: 50 concurrent analyses
- **GPU utilization**: < 4GB VRAM
- **CPU fallback**: < 2 minutes per recording

### Robustness Requirements
- Handle variable recording lengths (10-120 minutes)
- Work with 19-58 channels
- Robust to different equipment manufacturers
- Graceful degradation with poor quality data

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create `AbnormalityDetector` service class
2. Implement preprocessing pipeline
3. Add window extraction with overlap
4. Create aggregation strategies

### Phase 2: Model Integration (Week 2)
1. Integrate EEGPT encoder
2. Implement classification head
3. Add fine-tuning capabilities
4. Create model versioning system

### Phase 3: Clinical Features (Week 3)
1. Implement triage logic
2. Add confidence calibration
3. Create detailed reporting
4. Add explainability features

### Phase 4: Production Hardening (Week 4)
1. Performance optimization
2. Error handling and recovery
3. Monitoring and logging
4. API integration

## Testing Strategy

### Unit Tests
- Preprocessing functions
- Window extraction logic
- Aggregation methods
- Triage decisions

### Integration Tests
- End-to-end pipeline
- Model loading and inference
- Multi-file batch processing
- Error scenarios

### Performance Tests
- Processing speed benchmarks
- Memory usage profiling
- Concurrent request handling
- GPU/CPU switching

### Clinical Validation
- Test on TUAB dataset (n=276)
- Internal validation set (n=500)
- Edge cases (short recordings, artifacts)
- Cross-equipment validation

## API Endpoints

### Primary Endpoint
```
POST /api/v1/eeg/analyze/abnormality
Content-Type: multipart/form-data

Request:
- file: EDF/BDF file
- options: {
    "use_gpu": true,
    "return_window_scores": false,
    "confidence_threshold": 0.7
  }

Response:
{
    "status": "success",
    "results": { ... },  # As specified above
    "request_id": "uuid"
}
```

### Batch Endpoint
```
POST /api/v1/eeg/analyze/abnormality/batch
Content-Type: application/json

Request:
{
    "file_ids": ["s3://...", "s3://..."],
    "callback_url": "https://..."
}
```

## Monitoring & Metrics

### Key Metrics to Track
- Model accuracy (daily validation)
- Processing times (p50, p95, p99)
- Triage distribution
- Error rates by type
- Resource utilization

### Alerts
- Accuracy drop > 5%
- Processing time > 60s
- Error rate > 1%
- Memory usage > 80%

## Safety Considerations

1. **Never skip physician review** - This is a decision support tool
2. **Log all predictions** with timestamps for audit
3. **Flag low confidence** predictions (< 0.7)
4. **Fail safely** - Return "URGENT" triage on errors
5. **Version tracking** - Know which model made each prediction

## Future Enhancements

1. **Multi-class Classification**:
   - Normal
   - Epileptiform
   - Slowing
   - Artifact
   - Other

2. **Localization**:
   - Identify which channels show abnormalities
   - Temporal localization of events

3. **Fine-tuning Pipeline**:
   - Continuous learning from physician feedback
   - Hospital-specific adaptation

4. **Explainability**:
   - Attention visualization
   - Saliency maps
   - Natural language explanations

## References

1. Bussalb et al. (2025). "Automatic detection of abnormal clinical EEG: comparison of a finetuned foundation model with two deep learning models"
2. EEGPT Paper - Foundation model for universal EEG representation
3. PRD - Product Requirements Document v1.0.0
4. TRD - Technical Requirements Document

## Appendix: Configuration Parameters

```yaml
abnormality_detection:
  model:
    checkpoint_path: "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    device: "auto"  # auto, cuda, cpu
    batch_size: 16

  preprocessing:
    target_sampling_rate: 256
    bandpass_low: 0.5
    bandpass_high: 50.0
    notch_freq: 50  # or 60 for US

  windowing:
    window_duration: 4.0  # seconds
    overlap_ratio: 0.5
    min_windows: 10  # minimum for valid prediction

  thresholds:
    abnormal_threshold: 0.5
    urgent_threshold: 0.8
    expedite_threshold: 0.6
    routine_threshold: 0.4

  performance:
    max_processing_time: 30  # seconds
    max_concurrent: 50
    cache_ttl: 3600  # 1 hour
```

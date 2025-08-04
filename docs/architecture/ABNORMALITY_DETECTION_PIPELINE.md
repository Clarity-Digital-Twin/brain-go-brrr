# Abnormality Detection Pipeline

## Overview

This document details the implementation of EEG abnormality detection based on the BioSerenity-E1 paper and our EEGPT adaptation. The pipeline achieves state-of-the-art performance for binary classification of EEG recordings as normal or abnormal.

## Key Performance Metrics

### Target Benchmarks
- **BioSerenity-E1**: 94.63% balanced accuracy (closed source)
- **EEGPT (Our Implementation)**: Target >80% balanced accuracy, AUROC ≥0.93
- **Baseline CNN-LSTM**: 86% balanced accuracy
- **Processing Time**: <30 seconds for 20-minute EEG

## Data Specifications

### Input Requirements
- **Duration**: 20 minutes standard (flexible)
- **Sampling Rate**: 256Hz (resampled if needed)
- **Channels**: 19 channels (10-20 system)
- **Format**: EDF/BDF files

### Preprocessing Pipeline
```python
def preprocess_eeg(raw_eeg):
    # 1. Channel selection (19 channels)
    channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
                'T3', 'C3', 'CZ', 'C4', 'T4',
                'T5', 'P3', 'PZ', 'P4', 'T6', 
                'O1', 'O2']
    
    # 2. Re-referencing to average
    raw_eeg.set_eeg_reference('average')
    
    # 3. Filtering
    raw_eeg.filter(l_freq=0.5, h_freq=50.0, method='iir')
    raw_eeg.notch_filter(freqs=50.0)  # or 60Hz for US
    
    # 4. Resampling
    raw_eeg.resample(256)
    
    # 5. Windowing (8s windows, 50% overlap)
    windows = create_windows(raw_eeg, window_size=8.0, overlap=0.5)
    
    return windows
```

## Model Architecture

### 1. EEGPT Feature Extraction
```python
class AbnormalityDetectionPipeline:
    def __init__(self, checkpoint_path):
        # Load pretrained EEGPT
        self.backbone = create_normalized_eegpt(checkpoint_path)
        self.backbone.eval()  # Freeze during inference
        
        # Two-layer probe (matching paper)
        self.probe = EEGPTTwoLayerProbe(
            backbone_dim=768,
            n_input_channels=20,
            n_adapted_channels=19,
            hidden_dim=16,
            n_classes=2,
            dropout=0.5
        )
```

### 2. Two-Layer Probe Architecture
Based on the paper's CNN-LSTM approach, adapted for EEGPT features:

```python
# Layer 1: Channel adaptation + feature projection
conv1: 20 -> 22 channels (1x1 conv, max_norm=1.0)
conv2: 22 -> 19 channels (1x1 conv, max_norm=1.0)
linear1: 2048 -> 16 (max_norm=1.0)

# Layer 2: Classification
dropout: p=0.5
linear2: 256 -> 2 (max_norm=0.25)
```

### 3. Training Configuration
```yaml
# Hyperparameters from paper
learning_rate: 5e-4
batch_size: 32
epochs: 50
optimizer: AdamW
weight_decay: 0.05
scheduler: OneCycle
warmup_epochs: 5
layer_decay: 0.65
label_smoothing: 0.1
```

## Implementation Details

### 1. Window-Level Processing
```python
def process_recording(eeg_file_path):
    # Load and preprocess
    raw = mne.io.read_raw_edf(eeg_file_path)
    windows = preprocess_eeg(raw)
    
    # Extract features for each window
    window_predictions = []
    for window in windows:
        features = eegpt_model(window)
        pred = probe(features)
        window_predictions.append(pred)
    
    # Aggregate predictions
    recording_pred = aggregate_predictions(window_predictions)
    return recording_pred
```

### 2. Prediction Aggregation
```python
def aggregate_predictions(window_preds):
    # Option 1: Average probabilities
    avg_prob = torch.stack(window_preds).mean(dim=0)
    
    # Option 2: Voting (paper approach)
    votes = (torch.stack(window_preds) > 0.5).float()
    final_pred = votes.mean() > 0.5
    
    # Option 3: Weighted by confidence
    confidences = torch.abs(window_preds - 0.5) * 2
    weighted_pred = (window_preds * confidences).sum() / confidences.sum()
    
    return weighted_pred
```

### 3. Quality Control Integration
```python
def enhanced_abnormality_detection(eeg_file_path):
    # Step 1: Quality check with AutoReject
    raw = mne.io.read_raw_edf(eeg_file_path)
    epochs = create_epochs(raw)
    
    ar = AutoReject(n_interpolate=[1, 2, 3, 4])
    epochs_clean, reject_log = ar.fit_transform(epochs)
    
    # Step 2: Only process clean segments
    clean_windows = epochs_clean[~reject_log.bad_epochs]
    
    # Step 3: EEGPT feature extraction
    features = extract_eegpt_features(clean_windows)
    
    # Step 4: Classification
    prediction = classify_abnormal(features)
    
    return {
        'classification': 'abnormal' if prediction > 0.5 else 'normal',
        'confidence': float(prediction),
        'quality_score': 1 - reject_log.bad_epochs.mean(),
        'n_clean_windows': len(clean_windows)
    }
```

## Dataset Specifications

### TUAB Dataset (Our Training Data)
- **Total Files**: 2,717 (train) + 251 (eval)
- **Classes**: Normal (50.4%), Abnormal (49.6%)
- **Window Count**: 930,495 (train), 85,834 (eval)
- **Channel Mapping**: OLD naming (T3→T7, T4→T8, T5→P7, T6→P8)

### Data Augmentation
```python
# Time-domain augmentations
- Gaussian noise: σ = 0.01 * signal_std
- Time shifting: ±0.5s random shift
- Amplitude scaling: 0.8-1.2x random scale

# Frequency-domain augmentations  
- Band-limited noise injection
- Phase jittering in non-critical bands
```

## Performance Optimization

### 1. Caching Strategy
```python
# Pre-compute and cache:
- Preprocessed windows
- EEGPT features
- AutoReject masks

# Cache structure:
cache/
├── windows/        # Preprocessed 8s windows
├── features/       # EEGPT embeddings
└── quality/        # AutoReject results
```

### 2. Batch Processing
```python
def batch_process_recordings(file_paths, batch_size=16):
    dataloader = DataLoader(
        EEGDataset(file_paths),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            features = eegpt_model(batch)
            preds = probe(features)
            predictions.extend(preds)
    
    return predictions
```

## Clinical Integration

### 1. Output Format
```json
{
    "recording_id": "EEG_2024_001",
    "classification": "abnormal",
    "confidence": 0.87,
    "abnormality_segments": [
        {"start": 120.0, "end": 128.0, "confidence": 0.92},
        {"start": 456.0, "end": 464.0, "confidence": 0.85}
    ],
    "quality_metrics": {
        "bad_channels": ["T3", "O2"],
        "artifact_percentage": 15.2,
        "usable_duration": 17.4
    },
    "processing_time": 28.3
}
```

### 2. Confidence Calibration
```python
# Temperature scaling for better calibrated probabilities
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
```

## Evaluation Metrics

### 1. Primary Metrics
- **Balanced Accuracy**: (Sensitivity + Specificity) / 2
- **AUROC**: Area Under ROC Curve
- **F1 Score**: Harmonic mean of precision and recall

### 2. Clinical Metrics
- **Sensitivity**: True Positive Rate (catching abnormalities)
- **Specificity**: True Negative Rate (avoiding false alarms)
- **NPV**: Negative Predictive Value
- **PPV**: Positive Predictive Value

### 3. Performance Requirements
```python
# Minimum thresholds
assert balanced_accuracy >= 0.80
assert auroc >= 0.93
assert sensitivity >= 0.85  # Don't miss abnormalities
assert processing_time < 30  # seconds
```

## Common Issues and Solutions

### 1. Channel Mismatch
```python
# Handle different montages
def adapt_channels(raw_eeg, target_channels):
    available = raw_eeg.ch_names
    missing = set(target_channels) - set(available)
    
    if missing:
        # Interpolate missing channels
        raw_eeg = interpolate_missing_channels(raw_eeg, missing)
    
    return raw_eeg.pick_channels(target_channels)
```

### 2. Variable Recording Length
```python
# Handle recordings shorter than 20 minutes
def handle_short_recording(raw_eeg):
    duration = raw_eeg.times[-1]
    
    if duration < 600:  # Less than 10 minutes
        logger.warning(f"Short recording: {duration:.1f}s")
        # Adjust window overlap for more samples
        return create_windows(raw_eeg, overlap=0.75)
    
    return create_windows(raw_eeg, overlap=0.5)
```

### 3. Artifact Contamination
```python
# Robust to artifacts using AutoReject
def robust_prediction(raw_eeg):
    # Multiple strategies
    predictions = []
    
    # 1. With artifact rejection
    clean_pred = predict_with_autoreject(raw_eeg)
    predictions.append(clean_pred)
    
    # 2. Raw prediction (more data)
    raw_pred = predict_without_cleaning(raw_eeg)
    predictions.append(raw_pred * 0.8)  # Lower weight
    
    # 3. Ensemble
    final_pred = np.mean(predictions)
    return final_pred
```

## Future Enhancements

### 1. Multi-Stage Classification
```python
# Hierarchical approach (as discussed)
Stage 1: Normal vs Abnormal (current)
Stage 2: If abnormal → Type of abnormality
         - Epileptiform
         - Slowing
         - Asymmetry
         - Other
```

### 2. Explainability
```python
# Attention visualization
def get_abnormality_heatmap(recording):
    attention_weights = extract_attention_maps(recording)
    temporal_importance = attention_weights.mean(axis=1)
    channel_importance = attention_weights.mean(axis=2)
    
    return {
        'temporal_heatmap': temporal_importance,
        'channel_heatmap': channel_importance,
        'peak_abnormality_time': temporal_importance.argmax()
    }
```

### 3. Real-Time Processing
```python
# Streaming analysis for long-term monitoring
class StreamingAbnormalityDetector:
    def __init__(self, window_size=8.0, stride=2.0):
        self.buffer = RingBuffer(window_size)
        self.stride = stride
        
    def process_chunk(self, new_data):
        self.buffer.append(new_data)
        
        if self.buffer.is_full():
            window = self.buffer.get_window()
            prediction = self.model(window)
            
            if prediction > 0.8:  # High confidence
                self.trigger_alert()
            
            self.buffer.advance(self.stride)
```

## References

- BioSerenity Paper: "Automatic detection of abnormal clinical EEG"
- EEGPT Paper: Implementation details for feature extraction
- TUAB Dataset: Temple University Abnormal EEG Corpus
- Our Implementation: brain_go_brrr/tasks/enhanced_abnormality_detection.py
# Event Detection Architecture

## Overview

This document details the implementation of epileptiform discharge detection using state-of-the-art time-series classification methods (InceptionTime and MiniRocket) integrated with EEGPT features. The system detects 6 types of events in EEG recordings.

## Event Classes

1. **SPSW** - Spike and Sharp Wave (epileptiform)
2. **GPED** - Generalized Periodic Epileptiform Discharges
3. **PLED** - Periodic Lateralized Epileptiform Discharges
4. **EYEM** - Eye Movement (artifact)
5. **ARTF** - Other Artifacts
6. **BCKG** - Background (normal activity)

## Performance Benchmarks

### Target Metrics (from paper)
- **AUC**: 0.99
- **AUPRC**: 0.99
- **F1 Score**: 0.97
- **Processing**: Real-time capable (<100ms per window)

### Model Comparisons
| Model | AUC | AUPRC | F1 | Training Time |
|-------|-----|-------|-----|---------------|
| InceptionTime | 0.98 | 0.80 | 0.77 | ~2 hours |
| MiniRocket | 0.99 | 0.99 | 0.97 | <10 minutes |
| Traditional CNN | 0.95 | 0.75 | 0.72 | ~4 hours |

## Architecture Design

### 1. Hierarchical Detection Pipeline
```python
class HierarchicalEventDetector:
    def __init__(self):
        # Stage 1: Binary classification
        self.abnormal_detector = EEGPTAbnormalDetector()
        
        # Stage 2: Event classification (only for abnormal)
        self.event_classifier = EventClassifier(
            backbone='minirocket',  # or 'inceptiontime'
            n_classes=6
        )
    
    def detect(self, eeg_segment):
        # First check if abnormal
        abnormal_prob = self.abnormal_detector(eeg_segment)
        
        if abnormal_prob > 0.5:
            # Classify specific event type
            event_probs = self.event_classifier(eeg_segment)
            return {
                'is_abnormal': True,
                'abnormal_confidence': abnormal_prob,
                'event_type': event_probs.argmax(),
                'event_probabilities': event_probs
            }
        else:
            return {
                'is_abnormal': False,
                'abnormal_confidence': abnormal_prob,
                'event_type': 'BCKG',
                'event_probabilities': None
            }
```

### 2. Feature Extraction Pipeline
```python
def extract_features_for_event_detection(eeg_window):
    """
    Extract multi-scale features for event detection
    
    Args:
        eeg_window: [n_channels, n_samples] (e.g., 19 x 256 for 1s @ 256Hz)
    
    Returns:
        features: Dictionary of different feature representations
    """
    features = {}
    
    # 1. EEGPT features (global context)
    eegpt_features = extract_eegpt_features(eeg_window)
    features['eegpt'] = eegpt_features  # [768,]
    
    # 2. Raw time-series (for MiniRocket/InceptionTime)
    features['raw'] = eeg_window  # [19, 256]
    
    # 3. Spectral features (optional enhancement)
    features['spectral'] = compute_spectral_features(eeg_window)
    
    # 4. Spatial features (channel relationships)
    features['spatial'] = compute_spatial_features(eeg_window)
    
    return features
```

### 3. MiniRocket Implementation
```python
class MiniRocketEventDetector:
    def __init__(self, n_channels=19, n_classes=6):
        self.n_kernels = 10000  # Fixed in MiniRocket
        self.kernels = self._generate_kernels()
        
        # Classifier head
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        
    def _generate_kernels(self):
        """Generate fixed random kernels as per MiniRocket"""
        kernels = []
        kernel_sizes = [7, 9, 11]  # Key insight: multiple scales
        
        for size in kernel_sizes:
            for _ in range(self.n_kernels // len(kernel_sizes)):
                # Random weights from {-1, 0, 1}
                weights = np.random.choice([-1, 0, 1], size)
                bias = np.random.uniform(-1, 1)
                dilation = np.random.choice([1, 2, 3])
                
                kernels.append({
                    'weights': weights,
                    'bias': bias,
                    'dilation': dilation
                })
        
        return kernels
    
    def transform(self, X):
        """Apply kernels to extract features"""
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self.n_kernels * 2))
        
        for i, x in enumerate(X):
            for j, kernel in enumerate(self.kernels):
                # Dilated convolution
                conv_result = self._dilated_conv(x, kernel)
                
                # PPV and MPV features
                ppv = np.mean(conv_result > kernel['bias'])
                mpv = np.mean(np.maximum(0, conv_result))
                
                features[i, j*2] = ppv
                features[i, j*2 + 1] = mpv
        
        return features
```

### 4. InceptionTime Implementation
```python
class InceptionTimeEventDetector(nn.Module):
    def __init__(self, n_channels=19, n_classes=6):
        super().__init__()
        
        # Inception modules with multiple kernel sizes
        self.inception_modules = nn.ModuleList([
            InceptionModule(n_channels if i == 0 else 128, 128)
            for i in range(6)  # 6 inception blocks
        ])
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # x: [batch, channels, time]
        for module in self.inception_modules:
            x = module(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        return self.classifier(x)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Key: Multiple kernel sizes in parallel
        self.conv_39 = nn.Conv1d(in_channels, out_channels//4, 39, padding=19)
        self.conv_19 = nn.Conv1d(in_channels, out_channels//4, 19, padding=9)
        self.conv_9 = nn.Conv1d(in_channels, out_channels//4, 9, padding=4)
        self.conv_bottleneck = nn.Conv1d(in_channels, out_channels//4, 1)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Parallel convolutions
        x1 = self.conv_39(x)
        x2 = self.conv_19(x)
        x3 = self.conv_9(x)
        x4 = self.conv_bottleneck(x)
        
        # Concatenate
        x = torch.cat([x1, x2, x3, x4], dim=1)
        
        return self.relu(self.bn(x))
```

## Data Preprocessing

### 1. Window Extraction
```python
def extract_event_windows(eeg_recording, window_size=1.0, stride=0.5):
    """
    Extract overlapping windows for event detection
    
    Args:
        eeg_recording: MNE Raw object
        window_size: Window duration in seconds (paper uses 1s)
        stride: Stride in seconds (0.5s for 50% overlap)
    
    Returns:
        windows: List of [n_channels, n_samples] arrays
    """
    sfreq = eeg_recording.info['sfreq']
    window_samples = int(window_size * sfreq)
    stride_samples = int(stride * sfreq)
    
    windows = []
    timestamps = []
    
    for start in range(0, len(eeg_recording.times) - window_samples, stride_samples):
        end = start + window_samples
        window = eeg_recording.get_data(start=start, stop=end)
        
        # Apply preprocessing
        window = preprocess_window(window, sfreq)
        
        windows.append(window)
        timestamps.append(eeg_recording.times[start])
    
    return windows, timestamps
```

### 2. Channel-Specific Processing
```python
def preprocess_window(window, sfreq=256):
    """
    Preprocess window for event detection
    
    Key insight from paper: Process involved channels only
    """
    # Bandpass filter (0.5-50Hz)
    window = bandpass_filter(window, 0.5, 50, sfreq)
    
    # Z-score normalization per channel
    window = (window - window.mean(axis=1, keepdims=True)) / window.std(axis=1, keepdims=True)
    
    # Clip extreme values
    window = np.clip(window, -5, 5)
    
    return window
```

## Training Strategy

### 1. Multi-Label Handling
```python
class MultiLabelEventDataset(Dataset):
    """Handle overlapping events and multi-channel labels"""
    
    def __init__(self, eeg_files, annotations):
        self.samples = []
        
        for file, annot in zip(eeg_files, annotations):
            raw = mne.io.read_raw_edf(file)
            
            for event in annot:
                # Extract window around event
                start_time = event['start'] - 0.5  # Context
                end_time = event['end'] + 0.5
                
                window = raw.get_data(
                    start=int(start_time * raw.info['sfreq']),
                    stop=int(end_time * raw.info['sfreq'])
                )
                
                # Create multi-label target
                label = self._create_multilabel(event['types'])
                
                # Store involved channels
                channels = event['channels']
                
                self.samples.append({
                    'data': window[channels],  # Only involved channels
                    'label': label,
                    'channels': channels
                })
    
    def _create_multilabel(self, event_types):
        """Convert event types to multi-label vector"""
        label = np.zeros(6)
        type_map = {'SPSW': 0, 'GPED': 1, 'PLED': 2, 
                    'EYEM': 3, 'ARTF': 4, 'BCKG': 5}
        
        for event_type in event_types:
            label[type_map[event_type]] = 1
        
        return label
```

### 2. Class Imbalance Handling
```python
def compute_class_weights(dataset):
    """Compute weights for imbalanced classes"""
    # Count occurrences
    class_counts = np.zeros(6)
    for sample in dataset:
        class_counts += sample['label']
    
    # Inverse frequency weighting
    total = len(dataset)
    weights = total / (6 * class_counts + 1)
    
    # Normalize
    weights = weights / weights.sum() * 6
    
    return torch.tensor(weights, dtype=torch.float32)
```

### 3. Training Configuration
```yaml
# MiniRocket Training
minirocket:
  n_kernels: 10000
  classifier: RidgeCV
  cv_folds: 5
  alphas: [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# InceptionTime Training  
inceptiontime:
  learning_rate: 0.001
  batch_size: 64
  epochs: 1500
  optimizer: Adam
  scheduler: ReduceLROnPlateau
  early_stopping_patience: 50
  depth: 6
  kernel_sizes: [9, 19, 39]
  filters: 128
```

## Integration with EEGPT

### 1. Feature Fusion
```python
class EEGPTEventDetector(nn.Module):
    def __init__(self, eegpt_checkpoint, n_classes=6):
        super().__init__()
        
        # EEGPT for global features
        self.eegpt = load_eegpt(eegpt_checkpoint)
        self.eegpt.eval()
        
        # InceptionTime for temporal patterns
        self.inception = InceptionTimeEventDetector(n_channels=19, n_classes=128)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        # EEGPT features
        with torch.no_grad():
            eegpt_features = self.eegpt.extract_features(x)
            eegpt_features = eegpt_features.mean(dim=1)  # [B, 768]
        
        # InceptionTime features
        inception_features = self.inception.get_features(x)  # [B, 128]
        
        # Fusion
        combined = torch.cat([eegpt_features, inception_features], dim=1)
        
        return self.fusion(combined)
```

### 2. Hierarchical Pipeline Integration
```python
def integrated_event_detection_pipeline(eeg_file):
    """Complete pipeline with abnormal detection + event classification"""
    
    # Load EEG
    raw = mne.io.read_raw_edf(eeg_file)
    
    # Extract windows
    windows, timestamps = extract_event_windows(raw)
    
    results = []
    for window, timestamp in zip(windows, timestamps):
        # Stage 1: Abnormal detection
        abnormal_prob = abnormal_detector(window)
        
        if abnormal_prob > 0.5:
            # Stage 2: Event classification
            event_probs = event_detector(window)
            event_type = EVENT_CLASSES[event_probs.argmax()]
            
            # Stage 3: Localization (which channels)
            channel_importance = compute_channel_importance(window, event_type)
            
            results.append({
                'timestamp': timestamp,
                'duration': 1.0,  # seconds
                'abnormal_prob': float(abnormal_prob),
                'event_type': event_type,
                'event_confidence': float(event_probs.max()),
                'involved_channels': channel_importance.argsort()[-5:].tolist()
            })
    
    return results
```

## Evaluation Metrics

### 1. Event-Level Metrics
```python
def evaluate_event_detection(predictions, ground_truth):
    """Compute comprehensive metrics for event detection"""
    
    metrics = {}
    
    # Per-class metrics
    for i, event_class in enumerate(EVENT_CLASSES):
        y_true = (ground_truth == i)
        y_pred = (predictions == i)
        y_score = prediction_probs[:, i]
        
        metrics[event_class] = {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_score),
            'auprc': average_precision_score(y_true, y_score)
        }
    
    # Overall metrics
    metrics['overall'] = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'f1_macro': f1_score(ground_truth, predictions, average='macro'),
        'f1_weighted': f1_score(ground_truth, predictions, average='weighted'),
        'cohen_kappa': cohen_kappa_score(ground_truth, predictions)
    }
    
    return metrics
```

### 2. Clinical Metrics
```python
def compute_clinical_metrics(detections, expert_annotations):
    """Metrics relevant for clinical use"""
    
    # Detection latency
    latencies = []
    for detection in detections:
        closest_annotation = find_closest_annotation(
            detection['timestamp'], 
            expert_annotations
        )
        if closest_annotation:
            latency = abs(detection['timestamp'] - closest_annotation['start'])
            latencies.append(latency)
    
    # False positive rate per hour
    recording_duration_hours = get_recording_duration() / 3600
    false_positives = count_false_positives(detections, expert_annotations)
    fp_per_hour = false_positives / recording_duration_hours
    
    return {
        'mean_detection_latency': np.mean(latencies),
        'false_positives_per_hour': fp_per_hour,
        'sensitivity': compute_sensitivity(detections, expert_annotations),
        'precision': compute_precision(detections, expert_annotations)
    }
```

## Deployment Considerations

### 1. Real-Time Processing
```python
class StreamingEventDetector:
    def __init__(self, model, buffer_size=2.0, stride=0.1):
        self.model = model
        self.buffer = CircularBuffer(buffer_size)
        self.stride = stride
        self.last_detection = 0
        
    def process_chunk(self, new_data):
        self.buffer.append(new_data)
        
        # Check if enough data
        if self.buffer.size() < 1.0:
            return None
        
        # Extract window
        window = self.buffer.get_latest(1.0)
        
        # Detect events
        with torch.no_grad():
            prediction = self.model(window)
        
        # Debounce detections
        current_time = self.buffer.current_time()
        if prediction.max() > 0.8 and (current_time - self.last_detection) > 0.5:
            self.last_detection = current_time
            return {
                'time': current_time,
                'event': EVENT_CLASSES[prediction.argmax()],
                'confidence': float(prediction.max())
            }
        
        return None
```

### 2. GPU Optimization
```python
# Batch processing for efficiency
def batch_detect_events(windows, model, batch_size=32):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.stack(windows[i:i+batch_size])
            batch = batch.to('cuda')
            
            preds = model(batch)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)
```

## Future Enhancements

1. **Multi-Modal Integration**: Combine with video for better artifact detection
2. **Continuous Learning**: Update model with clinician feedback
3. **Explainability**: Attention maps showing which parts triggered detection
4. **Cross-Dataset Generalization**: Train on multiple public datasets
5. **Subcategory Detection**: Further classify SPSW into spike vs sharp wave

## References

- Original Paper: "Automated Interictal Epileptiform Discharge Detection From Scalp EEG"
- InceptionTime: https://github.com/hfawaz/InceptionTime
- MiniRocket: https://github.com/angus924/minirocket
- TUEV Dataset: Temple University Events Corpus
- Integration: brain_go_brrr/tasks/event_detection.py
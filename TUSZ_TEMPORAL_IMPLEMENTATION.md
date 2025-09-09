# 🎯 TUSZ TEMPORAL DETECTION IMPLEMENTATION

**Created**: September 8, 2025  
**Status**: 🔵 Planning Phase  
**Priority**: HIGH - Next major feature after TUEV  
**Type**: Reference Implementation Document

---

## 📋 EXECUTIVE SUMMARY

This document serves as the **Single Source of Truth** for implementing TUSZ (TUH Seizure) temporal detection. Unlike TUAB/TUEV which are classification tasks, TUSZ requires **temporal event detection** with onset/offset prediction, making it fundamentally more complex.

**Key Difference**: TUSZ is about WHEN seizures occur, not just IF they occur.

---

## 🎓 BACKGROUND & CONTEXT

### What is TUSZ?
- **Full Name**: Temple University Hospital Seizure Corpus
- **Task Type**: Temporal seizure detection (onset/offset times)
- **Dataset Version**: v1.1.1 (196 train patients, 50 eval patients)
- **Channels**: 22 channels (standard 10-20 montage)
- **Sampling Rate**: 256 Hz (matches EEGPT requirements)
- **Classes**: Seizure vs Background (~8% seizure, ~92% background)
- **Related**: TUSL (TUH Slowing) - 3-class: seizure/slowing/background

### Why is TUSZ Hard?
1. **Temporal Alignment**: Need precise onset/offset times
2. **Class Imbalance**: Seizures are rare events (~1-5% of recording time)
3. **Evaluation Metrics**: FA/24h, TAES, latency (not simple accuracy)
4. **Post-Processing**: Critical for performance (merge gaps, minimum duration)

---

## 📊 METRICS & EVALUATION PROTOCOL

### Primary Metrics (Per Literature)

#### 1. **TAES (Time-Aligned Event Scoring)**
- Accounts for temporal alignment of hypothesis to reference
- Jaccard index ≥ 0.5 for event match
- Penalizes poor time alignment heavily
- More stringent than overlap-based metrics

#### 2. **ATWV (Actual Term-Weighted Value)**
- Borrowed from spoken term detection
- Balances reward for correct detections vs penalty for false alarms
- Formula: `ATWV = P(term) * P(correct|term) - β * P(FA|¬term)`
- Typical β = 999.9 for seizure detection
- Values < 0.5 indicate poor performance

#### 3. **FA/24h @ Sensitivity**
- Clinical gold standard metric
- False alarms per 24 hours at fixed sensitivity (e.g., 95%)
- Clinically acceptable: < 10 FA/24h

#### 4. **Epoch-Based Metrics**
- Fixed time windows (e.g., 0.25s epochs)
- Less forgiving of alignment errors
- Good for diagnostic insights

### Scoring Variants (From Picone 2021)
1. **OVLP**: Any overlap counts as match (most lenient)
2. **DPALIGN**: Dynamic programming alignment
3. **TAES**: Time-aligned with Jaccard threshold
4. **EPOCH**: Fixed epoch-based scoring
5. **ATWV**: Weighted value scoring

### Evaluation Strategy
```python
# CRITICAL: Tune on VAL, freeze for TEST
# Per Picone et al. 2021 best practices
post_proc_params = tune_post_processing(val_predictions, val_labels)
test_metrics = evaluate(test_predictions, test_labels, frozen_params=post_proc_params)

# Report multiple metrics for diagnostic insight
metrics = {
    'taes': compute_taes(pred, ref, jaccard_thresh=0.5),
    'atwv': compute_atwv(pred, ref, beta=999.9),
    'fa_24h': compute_fa_per_24h(pred, ref, sensitivity=0.95),
    'epoch': compute_epoch_metrics(pred, ref, epoch_sec=0.25)
}
```

---

## 🏗️ ARCHITECTURE DESIGN

### Overview
```
EDF Files → Sliding Windows → EEGPT Features → Temporal Head → Post-Processing → Detections
```

### Components Location
```
src/brain_go_brrr/
├── infra/
│   ├── data/
│   │   └── tusz_dataset.py          # Dataset class
│   ├── preprocessing/
│   │   └── tusz_preprocessor.py     # Window generation
│   └── ml_models/
│       └── temporal_head.py         # Detection head
├── domain/
│   ├── metrics/
│   │   └── temporal_metrics.py      # FA/24h, TAES
│   └── post_processing/
│       └── temporal_smoothing.py    # Merge, threshold
└── utils/
    └── tusz_cache_builder.py        # Feature caching

experiments/tusz_temporal/
├── cache_embeddings.py               # Extract & cache
├── train_temporal.py                 # Train detection head
├── eval_temporal.py                  # Compute metrics
├── configs/
│   └── default.yaml                  # Hyperparameters
└── README.md                         # Usage instructions
```

---

## 💾 CACHING STRATEGY

### Cache Schema
```
artifacts/tusz_embeddings/
├── manifest.json                     # Global metadata
├── train/
│   └── {patient_id}/
│       ├── windows.npy              # [N, 2048] embeddings
│       ├── times.npy                # [N, 2] start/end times
│       ├── labels.npy               # [N] binary labels
│       └── meta.json                # Patient metadata
├── val/
└── test/
```

### Storage Calculations
- Window size: 2s @ 256Hz = 512 samples
- Hop size: 1s = 256 samples overlap
- Windows per hour: 3600 windows
- Embedding size: 2048 * 4 bytes = 8KB per window
- **1000 hours**: ~3.6M windows ≈ 29GB embeddings

---

## 🚀 IMPLEMENTATION PHASES

### Phase 1: Local POC (Week 1)
- [ ] Dataset exploration & patient splits
- [ ] Window generation pipeline
- [ ] EEGPT feature extraction
- [ ] Simple logistic head
- [ ] Basic metrics computation

### Phase 2: Optimization (Week 2)
- [ ] Post-processing tuning
- [ ] TCN/LSTM temporal head
- [ ] Multi-seed evaluation
- [ ] Performance profiling

### Phase 3: Scale (If Needed)
- [ ] Cloud infrastructure
- [ ] Distributed extraction
- [ ] Hyperparameter sweeps
- [ ] Production deployment

---

## 🔬 TECHNICAL SPECIFICATIONS

### Feature Extraction (Per Literature)

#### Traditional Approach (Picone 2021 - PROVEN TO WORK)
- **LFCC Features**: Linear Frequency Cepstral Coefficients
- **Frame**: 0.1 sec duration (critical for temporal resolution)
- **Window**: 0.2 sec analysis window
- **Features**: 7 cepstral coefficients + 1st/2nd derivatives = 26 dims/channel
- **Context**: 7-41 frame temporal window (0.7-4.1 seconds context)
- **Result**: This simple approach powers ALL their systems

#### Our EEGPT Approach (PROBABLY WRONG FOR TUSZ)
- **Window**: 4 seconds @ 256Hz = 1024 samples
- **Hop**: 1-2 seconds (75-50% overlap)
- **Features**: EEGPT encoder → 2048 dims (overkill)
- **Problem**: No temporal continuity between windows

#### RECOMMENDED Approach (Based on Evidence)
- **Features**: Simple CNN or LFCC (26-64 dims)
- **Frame**: 0.1-0.25 sec (fine temporal resolution)
- **Context**: Sliding window with heavy overlap
- **Temporal**: BiLSTM with hidden state continuity
- **Post-process**: 3-stage (threshold, merge, smooth)

### Window Generation
```python
def generate_windows(raw_eeg, window_sec=4.0, hop_sec=2.0):
    """
    Generate sliding windows from continuous EEG.
    EEGPT requires 4-second windows.
    
    Args:
        raw_eeg: MNE Raw object
        window_sec: Window duration in seconds (4.0 for EEGPT)
        hop_sec: Hop size in seconds (1-2 recommended)
    
    Returns:
        windows: [N, channels, samples]  # (N, 22, 1024)
        times: [N, 2] with start/end times
    """
    assert window_sec == 4.0, "EEGPT requires 4-second windows"
    assert raw_eeg.info['sfreq'] == 256, "EEGPT requires 256Hz"
    # Implementation here
```

### Post-Processing Parameters
```yaml
post_processing:
  threshold: 0.5          # Probability threshold
  min_duration_sec: 10    # Minimum seizure duration
  merge_gap_sec: 30       # Merge detections within gap
  hysteresis:
    high: 0.7            # Start detection
    low: 0.3             # Continue detection
```

---

## 📚 LITERATURE REFERENCES & SOTA RESULTS

### Key Papers

#### 1. Picone et al. 2021 - "Objective Evaluation Metrics"
- Comprehensive TUSZ evaluation framework
- Introduces TAES metric
- 5 ML systems compared (HMM/SdA, HMM/LSTM, IPCA/LSTM, CNN/MLP, CNN/LSTM)
- **Best Result**: CNN/LSTM with ATWV ~0.5, lowest FA rate

#### 2. Shah et al. 2018 - "TUH EEG Seizure Corpus"
- Original TUSZ dataset paper
- v1.1.1: 196 train, 50 eval patients
- ~8% seizure prevalence

#### 3. EEGPT Paper (2023)
- Mentions seizure detection but no TUSZ results
- References CHB-MIT dataset improvements

### State-of-the-Art Performance (Picone 2021 - ACTUAL NUMBERS)

| System | TAES Sens | OVLP Sens | FA/24h | ATWV | Key Insight |
|--------|-----------|-----------|--------|------|-------------|
| CNN/LSTM | 16.66% | 100% | Lowest | 0.45 | Best overall, low FA |
| HMM/LSTM | Variable | High | Higher | 0.35 | Detects longer events |
| IPCA/LSTM | Higher on EPOCH | Variable | Medium | ~0.35 | Good for long seizures |
| CNN/MLP | Medium | High | Medium | 0.40 | Simple but effective |
| HMM/SdA | Lowest | High | Highest | <0.35 | Baseline system |

**CRITICAL**: The sensitivity numbers vary WILDLY by metric:
- **OVLP**: Gives ~100% sensitivity (misleading!)
- **TAES**: Gives ~17% sensitivity (realistic)
- **Clinical reality**: Need <10 FA/24h, none achieve this

### Temporal Head Architectures

#### From Literature:
1. **HMM + Postprocessor**: Classical approach with language model smoothing
2. **CNN + LSTM**: Best overall (low FA rate)
3. **Pure LSTM**: Good for long-term dependencies
4. **CNN + MLP**: Simpler but effective

#### Our Options:
1. **Linear Probe**: Baseline (like TUAB/TUEV)
2. **TCN**: Temporal convolutions over embeddings
3. **LSTM/GRU**: Sequential modeling
4. **Transformer**: Attention over window sequence

---

## ⚠️ CRITICAL CONSIDERATIONS

### Must-Haves
1. **Patient-level splits** - Never mix same patient across splits
2. **Fixed seeds** - Reproducibility is mandatory
3. **Frozen post-processing** - Tune on VAL only
4. **Memory efficiency** - Stream processing, not load all

### Common Pitfalls
1. **Data leakage** - Patient contamination across splits
2. **Metric confusion** - FA/24h not FA/patient
3. **Post-proc overfitting** - Tuning on TEST
4. **Memory explosion** - Loading full dataset

---

## 🎯 SUCCESS CRITERIA

### Minimum Viable Product
- [ ] Extract features for 50 patients
- [ ] Train simple detection head
- [ ] Compute FA/24h and sensitivity
- [ ] Generate detection plots

### Production Ready
- [ ] Full dataset processing
- [ ] Multi-seed evaluation (3+ seeds)
- [ ] Matches/exceeds paper performance
- [ ] Deployment-ready code

---

## 🛠️ DETAILED IMPLEMENTATION PLAN

### Phase 0: Research & Design (2 days)
- [x] Study Picone 2021 evaluation framework
- [x] Understand TAES, ATWV, FA/24h metrics
- [ ] Download TUSZ v1.1.1 dataset
- [ ] Explore data structure and annotations
- [ ] Design patient-level splits (196 train → 156 train, 40 val)

### Phase 1: Data Pipeline (Week 1)

#### Day 1-2: Dataset Implementation
```python
# src/brain_go_brrr/infra/data/tusz_dataset.py
class TUSZDataset(Dataset):
    """TUSZ temporal dataset with patient-level splits."""
    
    def __init__(self, 
                 root_dir: Path,
                 split: Literal['train', 'val', 'test'],
                 window_sec: float = 4.0,
                 hop_sec: float = 2.0):
        self.patient_ids = self._load_split(split)
        self.annotations = self._load_annotations()
        
    def _generate_windows(self, patient_id: str):
        """Generate sliding windows with labels."""
        # Load EDF
        # Apply sliding window
        # Generate per-window labels (0/1)
        # Return windows, times, labels
```

#### Day 3-4: Feature Extraction
```python
# experiments/tusz_temporal/cache_embeddings.py
def extract_and_cache(patient_ids: List[str], 
                      output_dir: Path,
                      model_path: Path):
    """Extract EEGPT embeddings and cache to disk."""
    
    model = load_eegpt(model_path)
    
    for patient_id in tqdm(patient_ids):
        windows, times, labels = dataset.get_patient_windows(patient_id)
        
        # Extract features in batches
        embeddings = []
        for batch in batch_generator(windows, batch_size=32):
            with torch.no_grad():
                emb = model.encode(batch)  # [B, 2048]
                embeddings.append(emb.cpu().numpy())
        
        # Save to disk
        save_patient_cache(patient_id, embeddings, times, labels)
```

#### Day 5: Metrics Implementation
```python
# src/brain_go_brrr/domain/metrics/temporal_metrics.py
def compute_taes(pred_events, ref_events, jaccard_thresh=0.5):
    """Time-Aligned Event Scoring."""
    # For each reference event
    # Find best matching prediction
    # Compute Jaccard overlap
    # Count if >= threshold
    
def compute_fa_per_24h(pred_events, ref_events, total_hours):
    """False alarms per 24 hours."""
    # Count non-overlapping predictions
    # Normalize by recording duration
    
def compute_atwv(pred_events, ref_events, beta=999.9):
    """Actual Term-Weighted Value."""
    # P(term) * P(correct|term) - beta * P(FA|~term)
```

### Phase 2: Model Training (Week 2)

#### Day 1-2: Baseline Head
```python
# experiments/tusz_temporal/train_temporal.py
class TemporalHead(nn.Module):
    """Simple logistic head over EEGPT embeddings."""
    
    def __init__(self, input_dim=2048, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
```

#### Day 3-4: Post-Processing
```python
# src/brain_go_brrr/domain/post_processing/temporal_smoothing.py
def tune_post_processing(window_probs, window_times, ref_events):
    """Grid search post-processing parameters on VAL set."""
    
    best_params = None
    best_score = -float('inf')
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for min_dur in [2, 5, 10, 15]:
            for merge_gap in [3, 5, 10, 20]:
                events = apply_post_proc(window_probs, window_times,
                                        threshold, min_dur, merge_gap)
                score = compute_atwv(events, ref_events)
                if score > best_score:
                    best_score = score
                    best_params = (threshold, min_dur, merge_gap)
    
    return best_params
```

#### Day 5: Evaluation
```python
# experiments/tusz_temporal/eval_temporal.py
def evaluate_model(model, test_loader, post_proc_params):
    """Full evaluation with all metrics."""
    
    all_predictions = []
    all_references = []
    
    for patient_id in test_patients:
        # Load cached embeddings
        # Run through model
        # Apply frozen post-processing
        # Collect events
        
    # Compute all metrics
    metrics = {
        'taes': compute_taes(all_predictions, all_references),
        'atwv': compute_atwv(all_predictions, all_references),
        'fa_24h': compute_fa_per_24h(all_predictions, all_references),
        'sensitivity': compute_sensitivity(all_predictions, all_references),
        'det_curve': compute_det_curve(all_predictions, all_references)
    }
    
    return metrics
```

### Phase 3: Advanced Models (Optional)

#### TCN Head
```python
class TCNHead(nn.Module):
    """Temporal Convolutional Network over window sequence."""
    
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=3):
        self.tcn = TemporalConvNet(input_dim, [hidden_dim]*num_layers)
        self.output = nn.Linear(hidden_dim, 1)
```

#### LSTM with Context
```python  
class LSTMHead(nn.Module):
    """LSTM processing window sequence."""
    
    def __init__(self, input_dim=2048, hidden_dim=256):
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                           num_layers=2, bidirectional=True)
        self.output = nn.Linear(hidden_dim*2, 1)
```

### Validation Checklist
- [ ] Patient IDs never cross splits
- [ ] Post-proc tuned on VAL only
- [ ] Seeds fixed for reproducibility
- [ ] Memory usage < 24GB
- [ ] Metrics match literature formulas
- [ ] DET curves generated
- [ ] Results documented

---

## 📝 NOTES & DECISIONS

### Decision Log
- **Date**: _______
- **Decision**: _______
- **Rationale**: _______

### Open Questions
1. Optimal window/hop size?
2. Which temporal head architecture?
3. Post-processing strategy?
4. Cloud vs local trade-offs?

---

## 🔗 RELATED DOCUMENTS

- [TUEV_METRICS_SSOT.md](TUEV_METRICS_SSOT.md) - For metric examples
- [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md) - General metrics
- [experiments/README.md](experiments/README.md) - Experiment guidelines

---

**THIS DOCUMENT WILL BE CONTINUOUSLY UPDATED AS WE RESEARCH AND IMPLEMENT**
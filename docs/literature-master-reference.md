# Literature Master Reference Document

## Overview
This document serves as the comprehensive reference guide for all literature and research papers in the brain-go-brrr project. It provides AI agents and developers with exact locations of key information, technical specifications, and implementation details.

## Literature Repository Structure

```
literature/
├── markdown/
│   ├── EEGPT/
│   ├── autoreject/
│   ├── yasa/
│   ├── abnormal-eeg/
│   └── epileptiform-discharge-detection/
└── pdfs/
```

## 1. EEGPT: Foundation Model for Universal EEG Representation

### Paper Location
- **Markdown**: `/literature/markdown/EEGPT/EEGPT.md`
- **Original PDF**: `/literature/pdfs/EEGPT.pdf`

### Key Figures and Their Contents

| Figure | Path | Content |
|--------|------|---------|
| Architecture Overview | `_page_3_Figure_0.jpeg` | Shows dual self-supervised learning with masked prediction and reconstruction |
| Patching Method | `_page_4_Figure_7.jpeg` | Illustrates channel embedding and patch creation process |
| Fine-tuning Pipeline | `_page_5_Figure_0.jpeg` | Demonstrates frozen encoder with task-specific heads and electrode mapping |
| Model Scaling | `_page_8_Figure_6.jpeg` | Performance vs model size plots |
| Training Curves | `_page_13_Figure_2.jpeg` | Validation loss during training |
| Results Tables | `_page_22_Figure_0.jpeg` | Comprehensive performance metrics across tasks |

### Technical Specifications
- **Model Size**: 10M parameters (Large variant preferred)
- **Input Requirements**:
  - Sampling rate: 256 Hz
  - Window size: 4 seconds (1024 samples)
  - Channels: Up to 58 electrodes
  - Patch size: 64 samples (250ms)
- **Architecture**:
  - Vision Transformer backbone
  - Dual self-supervised objectives
  - Channel-wise and temporal patching
  - Rotary position embeddings
- **Performance**: 65-87.5% accuracy across tasks (TUEV: 65.42%, TUAB: 79.83%)

### Implementation Notes
- Requires electrode position mapping (see `_page_5_Figure_0.jpeg`)
- Supports variable channel counts through adaptive spatial filtering
- Pre-trained weights available (to be downloaded to `/data/models/eegpt/`)

## 2. Autoreject: Automated Artifact Rejection

### Paper Location
- **Markdown**: `/literature/markdown/autoreject/autoreject.md`
- **Original PDF**: `/literature/pdfs/autoreject.pdf`

### Key Figures

| Figure | Path | Content |
|--------|------|---------|
| Algorithm Overview | `_page_0_Picture_0.jpeg` | Flowchart of rejection pipeline |
| Validation Curves | `_page_22_Figure_2.jpeg` | Cross-validation performance |
| Threshold Selection | `_page_23_Figure_2.jpeg` | Bayesian optimization process |
| Comparison Results | `_page_27_Figure_2.jpeg` | Performance vs other methods |

### Technical Specifications
- **Algorithm**: Cross-validation based threshold estimation
- **Input**: Raw epochs (any sampling rate)
- **Output**: Clean epochs with interpolated bad channels
- **Key Parameters**:
  - `n_interpolate`: Max channels to interpolate
  - `consensus`: Fraction for consensus
  - `cv`: Number of CV folds
- **Performance**: Matches human expert agreement

### Implementation Code Location
- Reference implementation: `/reference_repos/autoreject/`
- Integration example: `/services/qc_flagger.py`

## 3. YASA: Yet Another Spindle Algorithm

### Paper Location
- **Markdown**: `/literature/markdown/yasa/yasa.md`
- **Original PDF**: `/literature/pdfs/yasa.pdf`

### Key Figures

| Figure | Path | Content |
|--------|------|---------|
| Architecture | `_page_0_Picture_0.jpeg` | YASA pipeline overview |
| Hypnogram Example | `_page_4_Figure_2.jpeg` | Sleep stage visualization |
| Performance Metrics | `_page_5_Figure_8.jpeg` | Confusion matrices |
| Feature Importance | `_page_9_Figure_2.jpeg` | Top features for classification |

### Technical Specifications
- **Algorithm**: LightGBM ensemble
- **Training Data**: 30,000+ hours of PSG
- **Input Requirements**:
  - EEG channels (C3-M2, C4-M1)
  - EOG channels (optional)
  - 30-second epochs
  - 100-256 Hz sampling
- **Output**: 5-class sleep stages (Wake, N1, N2, N3, REM)
- **Performance**: 87.46% median accuracy

### Integration Points
- Reference implementation: `/reference_repos/yasa/`
- Service wrapper: `/services/sleep_metrics.py`

## 4. Abnormal EEG Detection Comparison

### Paper Location
- **Markdown**: `/literature/markdown/abnormal-eeg/abnormal-eeg.md`
- **Original PDF**: `/literature/pdfs/abnormal-eeg.pdf`

### Key Figures

| Figure | Path | Content |
|--------|------|---------|
| Model Comparison | `_page_0_Picture_0.jpeg` | Performance overview |
| CNN Architecture | `_page_4_Figure_5.jpeg` | CNN-LSTM model structure |
| Transformer Design | `_page_6_Figure_2.jpeg` | STFT-transformer approach |
| Results Table | `_page_10_Figure_1.jpeg` | Comprehensive metrics |

### Key Finding
**BioSerenity-E1 Foundation Model** achieves best performance:
- 94.63% balanced accuracy [92.32-98.12 CI]
- Uses frozen pretrained encoder
- Resamples to 128 Hz (from 250/256/500 Hz originals)
- 16-second input windows
- Note: CNN-LSTM and Transformer models use 256 Hz

### Implementation Strategy
For abnormal detection, prioritize foundation models (EEGPT or BioSerenity-E1) over training from scratch.

## 5. Epileptiform Discharge Detection

### Paper Location
- **Markdown**: `/literature/markdown/epileptiform-discharge-detection/epileptiform-discharge-detection.md`
- **Original PDF**: `/literature/pdfs/epileptiform-discharge-detection.pdf`

### Key Figures

| Figure | Path | Content |
|--------|------|---------|
| IED Examples | `_page_1_Figure_3.jpeg` | Visual examples of different IED types |
| InceptionTime | `_page_4_Figure_11.jpeg` | Architecture diagram |
| Performance | `_page_10_Figure_2.jpeg` | ROC curves |
| Results | `_page_12_Figure_1.jpeg` | Performance comparison table |

### Technical Specifications
- **Best Models**: InceptionTime, Minirocket
- **Window Size**: 1 second
- **Classes**: SPSW, GPED, PLED, EYEM, ARTF, BCKG
- **Performance**: Up to 97% F1 score
- **Key Innovation**: Multi-scale temporal convolutions

## Integration Pipeline

### Recommended Processing Flow
```
1. Raw EEG → Autoreject (QC/Cleaning)
2. Clean EEG → EEGPT (Feature Extraction)
3. Features → Task-Specific Heads:
   - Abnormal Detection
   - Sleep Staging (YASA)
   - Event Detection (InceptionTime)
```

### Common Preprocessing Standards
- **Sampling Rate**: 256 Hz (EEGPT), 128 Hz (BioSerenity), 100-256 Hz (YASA)
- **Filtering**: 0.5-50 Hz bandpass typical
- **Reference**: Average reference or bipolar montages
- **Normalization**: Z-score or robust scaling

## Performance Benchmarks Summary

| Task | Best Model | Accuracy/F1 | Dataset |
|------|------------|-------------|----------|
| Abnormal Detection | BioSerenity-E1 | 94.63% | Private |
| Sleep Staging | YASA | 87.46% | NSRR |
| IED Detection | InceptionTime | 97% F1 | Temple |
| Artifact Rejection | Autoreject | 87.5% | Multiple |
| General Features | EEGPT | 65-87.5% | 6 datasets |

## Data Requirements

### Minimum Dataset Sizes
- Pre-training: 1000+ hours (for foundation models)
- Fine-tuning: 100+ subjects
- Evaluation: 20+ subjects (held out)

### Storage Estimates
- Raw EEG: ~100 MB/hour/subject
- Processed features: ~10 MB/hour/subject
- Model weights: 50-500 MB per model

## Implementation Checklist

### For AI Agents Building on This:

1. **Data Loading**
   - Use MNE-Python for EDF/BDF files
   - Apply montage from `/data/electrode_positions/`
   - Resample to target frequency

2. **Preprocessing**
   - Apply Autoreject from `/reference_repos/autoreject/`
   - Use parameters from paper figures
   - Save QC metrics

3. **Feature Extraction**
   - Load EEGPT weights from `/data/models/eegpt/`
   - Use frozen encoder mode
   - Extract embeddings

4. **Task-Specific Analysis**
   - For sleep: Use YASA service
   - For abnormal: Apply threshold on EEGPT output
   - For events: Run InceptionTime detector

5. **Output Generation**
   - Follow JSON schema from `/docs/api-specification.md`
   - Generate PDF reports with matplotlib
   - Log predictions for future training

## References to Original Papers

1. **EEGPT**: "A General Brain Network Model Based on EEG for Stronger AI" (2024)
2. **Autoreject**: "Autoreject: Automated artifact rejection for MEG and EEG data" (2017)
3. **YASA**: "YASA: Yet Another Spindle Algorithm" (2021)
4. **Abnormal EEG**: "Foundation models for abnormal EEG detection" (2024)
5. **IED Detection**: "Deep learning for interictal epileptiform discharge detection" (2023)

## Notes for Future Development

### Priority Implementation Order
1. Basic preprocessing with Autoreject
2. EEGPT feature extraction
3. Abnormal/normal classification
4. Sleep staging with YASA
5. Advanced event detection

### Model Weight Locations
- EEGPT: Download to `/data/models/eegpt/eegpt_large.pt`
- YASA: Included in pip package
- Autoreject: No weights needed (algorithmic)
- InceptionTime: To be trained or downloaded

### API Endpoints to Implement
Based on literature capabilities:
- `/api/v1/preprocess` - Autoreject cleaning
- `/api/v1/analyze/abnormal` - Binary classification  
- `/api/v1/analyze/sleep` - Sleep staging
- `/api/v1/analyze/events` - IED detection
- `/api/v1/extract/features` - EEGPT embeddings

This document should be updated as new papers are added or implementation details change.
# Comprehensive Literature Summary: EEG Analysis Tools and Models

This document summarizes the key technical details, architectures, performance metrics, and implementation details from five major papers/tools in the EEG analysis domain.

## 1. EEGPT: Pretrained Transformer for Universal EEG Representation

**Paper Path**: `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/literature/markdown/EEGPT/EEGPT.md`

### Overview
EEGPT is a 10-million-parameter pretrained transformer model designed for universal EEG feature extraction using masked self-supervised learning.

### Technical Architecture
- **Model Type**: Transformer-based with hierarchical structure
- **Key Innovation**: Dual self-supervised learning combining:
  - Spatio-temporal representation alignment
  - Mask-based reconstruction
- **Architecture Components**:
  - Encoder: Integrates spatial information from masked patches
  - Predictor: Predicts complete encoded features using temporal position information
  - Momentum Encoder: Accumulates encoder parameters with τ = 0.01
  - Reconstructor: Generates reconstructed patches from encoded features

### Key Technical Details
- **Input Format**: EEG patches of size (M channels × T timepoints)
- **Patch Duration**: 250ms (64 samples at 256 Hz)
- **Masking Strategy**: 50% time patches, 80% channel patches
- **Embedding**: Local spatio-temporal embedding with channel-specific codebook
- **Training Loss**: L = L_A + L_R (alignment + reconstruction losses)

### Performance Metrics
- **BCIC-2A (Motor Imagery)**: 58.46% accuracy
- **BCIC-2B**: 80.59% AUROC
- **Sleep-EDFx**: 76.54% weighted F1
- **Scaling Law**: ACC = (33.6 × N)^0.029 where N is parameter count

### Implementation Requirements
- **Preprocessing**: 
  - Bandpass filter: 0-38 Hz for motor imagery tasks
  - Resampling: 256 Hz
  - Channels: 58 electrodes (M=58)
- **Training**: 200 epochs, batch size 64, AdamW optimizer
- **Hardware**: 8 Nvidia 3090 GPUs for training

**Key Figures**:
- Architecture diagram: `_page_3_Figure_0.jpeg`
- Scaling laws: `_page_8_Figure_6.jpeg`

---

## 2. Autoreject: Automated Artifact Rejection for MEG and EEG

**Paper Path**: `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/literature/markdown/autoreject/autoreject.md`

### Overview
Autoreject provides automated, data-driven detection and repair of bad segments in M/EEG data using cross-validation and robust metrics.

### Technical Approach
- **Global Autoreject**: Sets single peak-to-peak threshold for all sensors
- **Local Autoreject**: Sets sensor-specific thresholds
- **Key Innovation**: Cross-validation with median-based validation metric robust to outliers

### Key Formulas
1. **Peak-to-peak amplitude**:
   ```
   A_ij = max(X_it) - min(X_it)
   ```
   
2. **Good trials selection**:
   ```
   G_k = {i ∈ T_k | max(A_ij) ≤ τ}
   ```

3. **Validation metric (RMSE)**:
   ```
   e_k(τ = ||mean(X_G_k) - median(X_V_k)||_Fro
   ```

### Algorithm Parameters
- **κ**: Maximum number of bad sensors in non-rejected trial
- **ρ**: Maximum number of sensors that can be interpolated
- **Data Augmentation**: Leave-one-sensor-out interpolation doubles trial count

### Performance
- **Median accuracy**: 87.5% on 585 testing nights
- **Cohen's kappa**: 0.819 (excellent agreement)
- **Processing time**: < 5 seconds for full night PSG

### Implementation Details
- **Optimization**: Bayesian optimization for threshold selection
- **Interpolation**: 
  - EEG: Spherical splines
  - MEG: Minimum Norm Estimates with spherical harmonics
- **Code**: Available at https://autoreject.github.io

**Key Figures**:
- Cross-validation curve: `_page_22_Figure_2.jpeg`

---

## 3. YASA: Automated Sleep Staging

**Paper Path**: `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/literature/markdown/yasa/yasa.md`

### Overview
YASA is a free, open-source automated sleep staging tool trained on 30,000+ hours of PSG data.

### Technical Details
- **Algorithm Type**: Traditional features-based approach (not deep learning)
- **Training Data**: 3,163 nights from 7 datasets (NSRR)
- **Features**: EEG-based (11/20 top features), EOG (7/20), EMG (1/20)
- **Key Features**:
  - Fractal dimension
  - Permutation entropy
  - Absolute power spectrum
  - Temporal smoothing (7.5 min rolling average)

### Performance Metrics
- **Median accuracy**: 87.46%
- **Cohen's kappa**: 0.819
- **Stage-specific performance**:
  - N3: 83.2% sensitivity, F1=0.835
  - REM: 86.8% sensitivity, F1=0.868
  - N2: 85.7% sensitivity, F1=0.857
  - Wake: 93.0% sensitivity, F1=0.930
  - N1: 45.4% sensitivity, F1=0.432 (lowest)

### Implementation
- **Language**: Python
- **Processing speed**: < 5 seconds for full night
- **Installation**: `pip install yasa`
- **Input requirements**: 
  - Minimum: 1 EEG channel
  - Optimal: 1 EEG + 1 EOG + 1 EMG
- **Sampling rate**: Flexible (automatically handled)

### Unique Features
- Probability scores for each epoch
- Confidence metrics
- Works with standard EDF files
- No GPU required

**Key Figures**:
- Performance metrics: `_page_4_Figure_2.jpeg`
- Example hypnogram and spectrogram: `_page_9_Figure_2.jpeg`

---

## 4. Abnormal EEG Detection: Comparison of Foundation Model with Deep Learning

**Paper Path**: `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/literature/markdown/abnormal-eeg/abnormal-eeg.md`

### Overview
Compares BioSerenity-E1 (foundation model) with CNN-LSTM and Transformer models for classifying entire EEG recordings as normal/abnormal.

### Models Compared

#### 1. CNN-LSTM Model
- **Architecture**: 4 Conv1D layers → Average pooling → LSTM → Dense
- **Filter progression**: 16→32→64→128
- **Dropout**: 0.5 after LSTM
- **Input**: 20 min, 19 channels, 256 Hz

#### 2. Transformer Model
- **Architecture**: STFT → 2D-CNN → Transformer block → Classifier
- **STFT windows**: 1s with 10% overlap
- **Advantage**: Captures long-range dependencies

#### 3. BioSerenity-E1 Finetuned
- **Base model**: Pretrained on 4,000 hours EEG
- **Input windows**: 16 seconds × 16 channels
- **Architecture**: Frozen pretrained model → Trainable prediction head
- **Preprocessing**: 128 Hz, 16 channels, 0.5-45 Hz bandpass

### Performance Results

**Dataset A (n=4,480)**:
- CNN-LSTM: 86.34% balanced accuracy
- Transformer: 87.95% balanced accuracy
- BioSerenity-E1: **89.19% balanced accuracy**

**Dataset B (n=198, 3 experts)**:
- CNN-LSTM: 88.27% balanced accuracy
- Transformer: 90.92% balanced accuracy
- BioSerenity-E1: **94.63% balanced accuracy**

**TUAB (n=276)**:
- CNN-LSTM: 76.09% accuracy
- Transformer: 78.26% accuracy
- BioSerenity-E1: **82.25% accuracy**

### Key Finding
BioSerenity-E1 maintains high performance even with limited training data (74 recordings: 86.36% ± 1.27 balanced accuracy)

**Key Figures**:
- CNN-LSTM architecture: `_page_4_Figure_5.jpeg`
- Transformer architecture: `_page_5_Figure_4.jpeg`
- BioSerenity-E1 architecture: `_page_6_Figure_2.jpeg`

---

## 5. Epileptiform Discharge Detection Using Time-Series Classification

**Paper Path**: `/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/brain-go-brrr/literature/markdown/epileptiform-discharge-detection/epileptiform-discharge-detection.md`

### Overview
Explores InceptionTime and Minirocket algorithms for automated interictal epileptiform discharge (IED) detection.

### Algorithms Tested

#### InceptionTime
- **Architecture**: Multiple 1D conv kernels (8, 16, 32) → Concatenation
- **Key feature**: Large kernel sizes matching IED duration (30-125ms)
- **Ensemble**: 5 models with different seeds
- **Spatial dropout**: Applied after each conv block

#### Minirocket
- **Features**: 10k fixed kernels → PPV statistics
- **Classifier**: Ridge (< 10k samples) or Logistic Regression
- **Advantage**: Extremely fast training (< 10 min for entire UCR archive)

### Data Specifications
- **Window size**: 1 second
- **Sampling rate**: 256 Hz
- **Preprocessing**: 0.5-49 Hz bandpass
- **Montage**: TCP (Temporal Central Parasagittal)

### Performance Metrics

**Private datasets (IGE patients)**:
- InceptionTime (z-score): AUC=0.98, F1=0.77, AUPRC=0.80
- Minirocket: AUC=0.99, F1=0.74, AUPRC=0.79

**TUEV public dataset**:
- InceptionTime (log transform): F1=0.97, AUPRC=0.99
- Minirocket: F1=0.92, AUPRC=0.97

### Key Findings
- Optimal kernel size (32) correlates with IED duration (20-200ms)
- Models trained on private data generalize well to TUEV
- Models trained on TUEV don't generalize well to private data
- Artifact removal (Hurst exponent thresholding) improves cross-dataset performance

**Key Figures**:
- IED example: `_page_1_Figure_3.jpeg`
- InceptionTime architecture: `_page_4_Figure_11.jpeg`

---

## Integration and Synergies

### Common Technical Themes
1. **Preprocessing Standards**:
   - Resampling to 256 Hz (common across tools)
   - Bandpass filtering (varies by application)
   - Channel standardization (10-20 system)

2. **Validation Approaches**:
   - Cross-validation with held-out test sets
   - Multiple datasets for generalization testing
   - Clinical expert validation

3. **Performance Metrics**:
   - Balanced accuracy (handles class imbalance)
   - F1 scores (precision-recall balance)
   - Cohen's kappa (inter-rater agreement)
   - AUPRC (for imbalanced datasets)

### Potential Integration Pipeline
1. **Autoreject** → Clean raw EEG data
2. **YASA** → Sleep stage classification
3. **BioSerenity-E1 or EEGPT** → Extract universal features
4. **InceptionTime/Minirocket** → Specific event detection (IEDs)

### Key Takeaways
- Foundation models (EEGPT, BioSerenity-E1) show superior performance with less training data
- Artifact rejection is crucial for all downstream tasks
- Dataset diversity is essential for generalization
- Temporal features at multiple scales are important
- Open-source implementations facilitate adoption and benchmarking